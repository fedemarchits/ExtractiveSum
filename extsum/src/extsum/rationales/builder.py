"""Generate the one-shot reasoning traces with a strong reference model, once.

For the four reasoning techniques the one-shot example must SHOW reasoning, not
just an answer. To keep that honest the trace is never hand-written: one
reference model solves an exemplar and reveals its working, its answer is
compared with the exemplar's labels, and the best-overlapping trace is cached.
Every grid model then sees the same exemplar and trace.

A model will not reproduce the labels exactly, so acceptance is by overlap:
stop at the first exemplar with F1 >= accept_f1, else keep the best one seen if
it reaches min_f1, else write nothing (the technique's _trace variant then has
no exemplar).

    extsum rationales --experiment billsum

Output: data/rationales/<experiment>/<technique>__<unit>.json
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

from ..backends import DEFAULT_MODELS_YAML, GenConfig, get_backend
from ..config import Experiment, load_experiment
from ..metrics import prf
from ..postprocess import safe_extract_json
from ..prompts.rationale import (
    REASONING_TECHNIQUES,
    RationaleShot,
    configure_rationale_cache,
    load_rationale,
    save_rationale,
)

# ---------------------------------------------------------------------------
# Aspect task (ACLSum): free-form working, answer on a final marker line.
# ---------------------------------------------------------------------------

ASPECT_SYSTEM = "You are an expert in extractive summarization."

# How to ask each technique to REVEAL its working (the opposite of the run-time
# "do not include reasoning" instruction).
_ASPECT_ELICIT = {
    "chain_of_thought":
        "For each sentence, reason step-by-step: identify its key claim, map it "
        "to challenge/approach/outcome, and conclude Yes/No for '{aspect}'. Show "
        "all reasoning.",
    "self_ask":
        "For each sentence, write the question 'Does this primarily express "
        "\"{aspect}\"?' and answer Yes/No with a short justification. Show all Q/A.",
    "scoring_based":
        "For each sentence, give a 1-5 relevance score for '{aspect}' with a one-"
        "clause reason. Show every score.",
    "salience_inference":
        "State the salient '{aspect}' theme in one sentence, then score each "
        "sentence 0-2 for it with a brief reason. Show the theme and all scores.",
}

_ASPECT_END = "\nEnd with exactly one line: Selected indices: [comma-separated 1-based indices]"
_ASPECT_ANSWER_RE = re.compile(r"Selected indices:\s*\[([0-9,\s]*)\]")


def _aspect_prompt(technique: str, aspect: str, sentences: Sequence[str]) -> str:
    from ..prompts.sets.aspect.shared import header, numbered
    return (
        f"{header(aspect)}\n\nInput:\n{numbered(sentences)}\n\n"
        f"Instructions:\n{_ASPECT_ELICIT[technique].format(aspect=aspect)}{_ASPECT_END}"
    )


def _aspect_parse(raw: str, n_sentences: int) -> Tuple[List[int], str]:
    match = None
    for match in _ASPECT_ANSWER_RE.finditer(raw):
        pass  # keep the last match
    if not match:
        return [], ""
    indices = sorted(int(x) for x in match.group(1).split(",") if x.strip())
    return indices, _ASPECT_ANSWER_RE.split(raw)[0].rstrip()


# ---------------------------------------------------------------------------
# Document task (silver labels): "Reasoning:" section, then a final JSON.
# ---------------------------------------------------------------------------

_DOCUMENT_SYSTEM = {
    "document": (
        "You are an expert in extractive summarization. "
        "Provide concise reasoning and then a JSON answer."
    ),
    "bill": (
        "You are an expert in extractive summarization of legislation. "
        "Provide concise reasoning and then a JSON answer."
    ),
}

_DOCUMENT_ELICIT = {
    "document": {
        "chain_of_thought": (
            "Reason step by step about the document's central topic, key facts, "
            "coverage, redundancy, and which sentences are essential."
        ),
        "self_ask": (
            "For each sentence, ask whether removing it would cause the summary "
            "to lose important information, and whether the information is already "
            "covered elsewhere."
        ),
        "scoring_based": (
            "Assign each sentence an internal summary-importance score from 1 to 5. "
            "Explain which sentences are essential, useful, redundant, or minor."
        ),
        "salience_inference": (
            "Infer the document's central theme, assess sentence salience, and explain "
            "which sentences are necessary for a concise whole-document summary."
        ),
    },
    "bill": {
        "chain_of_thought": (
            "Reason step by step about the bill's main purpose, major provisions, "
            "requirements, affected parties, substantive effects, coverage, and redundancy."
        ),
        "self_ask": (
            "For each sentence, ask whether it contributes essential, non-redundant information "
            "about the bill's purpose, provisions, requirements, conditions, or effects."
        ),
        "scoring_based": (
            "Assign each sentence an internal summary-importance score from 1 to 5, considering "
            "legal/substantive importance, coverage, specificity, and redundancy."
        ),
        "salience_inference": (
            "Infer the bill's central purpose and salient provisions, then explain which sentences "
            "are necessary for a concise whole-bill summary."
        ),
    },
}


def _document_prompt(prompt_set: str, technique: str, sentences: Sequence[str]) -> str:
    numbered = "\n".join(
        f"Sentence {index}: {sentence}"
        for index, sentence in enumerate(sentences, start=1)
    )
    return (
        f"{_DOCUMENT_ELICIT[prompt_set][technique]}\n\n"
        "Document:\n"
        f"{numbered}\n\n"
        "Return your answer in exactly this structure:\n\n"
        "Reasoning:\n"
        "<your concise reasoning>\n\n"
        'Final JSON:\n'
        '{"selected_sentences": [1, 2, 3]}\n\n'
        "Use only valid 1-based sentence indices."
    )


def _document_parse(raw: str, n_sentences: int) -> Tuple[List[int], str]:
    text = str(raw or "").strip()

    indices: List[int] = []
    for value in safe_extract_json(text).get("selected_sentences", []):
        if isinstance(value, bool):
            continue
        try:
            index = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 1 <= index <= n_sentences and index not in indices:
            indices.append(index)

    match = re.search(
        r'Reasoning:\s*(.*?)(?:Final JSON:|\{\s*"selected_sentences")',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    rationale = match.group(1).strip() if match else ""
    if not rationale:
        # Fallback: everything before a trailing JSON object.
        rationale = re.sub(
            r'\{\s*"selected_sentences"\s*:\s*\[[^\]]*\]\s*\}\s*$',
            "",
            text,
            flags=re.DOTALL,
        ).strip()
    return indices, rationale


# ---------------------------------------------------------------------------
# Shared search loop
# ---------------------------------------------------------------------------

Parse = Callable[[str, int], Tuple[List[int], str]]


def best_trace(
    backend,
    exemplars,
    make_prompt: Callable[[Sequence[str]], str],
    parse: Parse,
    system: str,
    accept_f1: float,
    min_f1: float,
    label: str,
) -> Optional[Tuple[float, object, List[int], str]]:
    """Try exemplars in order; return (f1, exemplar, shown_indices, trace)."""
    best = None
    for number, exemplar in enumerate(exemplars, start=1):
        try:
            raw = backend.generate_batch([make_prompt(exemplar.sentences)], system=system)[0]
        except Exception as exc:  # timeout / rate limit / transient API error
            print(f"  [warn] {label}: exemplar {number} failed ({type(exc).__name__})", flush=True)
            continue
        shown, trace = parse(raw, len(exemplar.sentences))
        if not shown or not trace:
            continue
        f1 = float(prf(exemplar.gold_indices, shown)["f1"])
        print(f"  {label}: exemplar {number} F1={f1:.2f}", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, exemplar, shown, trace)
        if f1 >= accept_f1:
            break
    if best is not None and best[0] >= min_f1:
        return best
    return None


def _cells(exp: Experiment):
    """(unit, exemplars, make_prompt, parse, system) for every unit of the task."""
    seed = exp.seed
    max_tries = int(exp.rationale.get("max_exemplar_tries", 10))

    if exp.task == "aspect":
        from ..data import aclsum
        from ..fewshot.aspect import iter_exemplars as aspect_exemplars
        pool = aclsum.load_split(exp.rationale.get("exemplar_split", "train"))
        for aspect in exp.dataset.get("aspects", aclsum.ASPECTS):
            def exemplars(aspect=aspect):
                return aspect_exemplars(pool, aspect, seed=seed, limit=max_tries)

            def make_prompt(technique, aspect=aspect):
                return lambda sentences: _aspect_prompt(technique, aspect, sentences)

            yield aspect, exemplars, make_prompt, _aspect_parse, ASPECT_SYSTEM
        return

    from ..data import silver_io
    from ..fewshot.silver import iter_exemplars
    from ..paths import resolve
    from ..tasks.document import SUMMARY_UNIT

    if not exp.dataset.get("exemplar_silver"):
        raise RuntimeError("dataset.exemplar_silver is required to build rationales.")
    pool = silver_io.load_split(resolve(exp.dataset["exemplar_silver"]))
    if len(pool) == 0:
        raise ValueError("The exemplar silver file is empty.")

    def exemplars():
        return iter_exemplars(pool, SUMMARY_UNIT, seed=seed, limit=max_tries)

    def make_prompt(technique):
        return lambda sentences: _document_prompt(exp.prompt_set, technique, sentences)

    yield SUMMARY_UNIT, exemplars, make_prompt, _document_parse, _DOCUMENT_SYSTEM[exp.prompt_set]


def build(
    experiment: str,
    model: Optional[str] = None,
    models_yaml: str = DEFAULT_MODELS_YAML,
    overwrite: bool = False,
    backend=None,
) -> None:
    exp = load_experiment(experiment)
    cfg = exp.rationale
    accept_f1 = float(cfg.get("accept_f1", 0.7))
    min_f1 = float(cfg.get("min_f1", 0.4))
    if not 0.0 <= min_f1 <= accept_f1 <= 1.0:
        raise ValueError("Expected 0 <= min_f1 <= accept_f1 <= 1.")

    source_model = model or cfg.get("source_model")
    if not source_model:
        raise RuntimeError("No rationale source model was configured.")

    cache_dir = exp.rationale_dir
    configure_rationale_cache(cache_dir)

    if backend is None:
        backend = get_backend(
            source_model,
            models_yaml,
            # Traces need room to reason; the run-time budget would truncate them.
            GenConfig(
                max_new_tokens=int(cfg.get("max_new_tokens", 1024)),
                temperature=0.0,
                seed=exp.seed,
            ),
        )

    for unit, exemplars, make_prompt, parse, system in _cells(exp):
        for technique in sorted(REASONING_TECHNIQUES):
            label = f"{technique}/{unit}"
            # Resume-safe: a run interrupted by a quota error re-spends nothing.
            if not overwrite and load_rationale(technique, unit) is not None:
                print(f"[have] {label} (cached; use --overwrite to redo)", flush=True)
                continue

            best = best_trace(
                backend, exemplars(), make_prompt(technique), parse, system,
                accept_f1, min_f1, label,
            )
            if best is None:
                print(f"[skip] {label}: no trace reached F1 >= {min_f1}", flush=True)
                continue

            f1, exemplar, shown, trace = best
            path = save_rationale(
                RationaleShot(
                    technique=technique,
                    aspect=unit,
                    source_model=source_model,
                    exemplar_sentences=list(exemplar.sentences),
                    shown_indices=shown,
                    gold_indices=list(exemplar.gold_indices),
                    f1=f1,
                    rationale=trace,
                ),
                base_dir=cache_dir,
            )
            print(f"[ok]   {label} -> {path.name} (F1={f1:.2f})", flush=True)

    backend.free_memory()
