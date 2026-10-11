"""Orthogonal wrapper mechanisms, applied on top of any technique x variant.

self_consistency  : sample the same prompt N times (temperature > 0) and keep
                    indices chosen by majority. Reduces unstable selections.
dynamic_llm_capper: a second pass where the model prunes its own selection to
                    the essential sentences (no fixed K). When enabled it
                    replaces the post-hoc cap of capped variants.

Both are off by default (see the experiment configs) and are the only place
the runner deviates from a single greedy pass.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .postprocess import clean_indices, safe_extract_json

CapperPrompt = Callable[[Sequence[str], str, Sequence[int]], str]


def majority_vote(
    samples: Sequence[Sequence[int]], n_samples: int, threshold: Optional[int] = None
) -> List[int]:
    """Keep indices present in at least `threshold` samples (default > half)."""
    thr = threshold if threshold is not None else (n_samples // 2 + 1)
    counts: Counter = Counter()
    for s in samples:
        counts.update(set(s))
    return sorted(i for i, c in counts.items() if c >= thr)


def parse_batch(outs: Sequence[str], units: Sequence[str], n: int) -> Dict[str, List[int]]:
    return {
        u: clean_indices(safe_extract_json(o).get("selected_sentences", []), n)
        for u, o in zip(units, outs)
    }


def dynamic_cap(
    backend, capper_prompt: CapperPrompt, sentences, unit, selected, system
) -> List[int]:
    """Prune `selected` via a model pass; never adds indices. Falls back to the
    original selection if the prune pass yields nothing valid."""
    if not selected:
        return []
    out = backend.generate_batch(
        [capper_prompt(sentences, unit, selected)], system=system
    )[0]
    pruned = clean_indices(safe_extract_json(out).get("selected_sentences", []), len(sentences))
    keep = [i for i in pruned if i in set(selected)]
    return keep or list(selected)


def select_document(
    backend, prompts, units, sentences, sc, dyn, system,
    capper_prompt: Optional[CapperPrompt] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, object]]:
    """Produce per-unit predictions for one doc, honoring the wrappers.

    Returns (preds, raws): preds maps unit -> indices; raws maps unit -> the raw
    model text (a list of N strings when self_consistency is on). The fixed cap
    of capped variants is applied by the runner, not here.
    """
    n = len(sentences)
    if sc and sc.get("enabled"):
        ns = int(sc.get("n_samples", 5))
        passes = [backend.generate_batch(prompts, system=system) for _ in range(ns)]
        samples = [parse_batch(outs, units, n) for outs in passes]
        preds = {u: majority_vote([s[u] for s in samples], ns) for u in units}
        raws = {u: [outs[j] for outs in passes] for j, u in enumerate(units)}
    else:
        outs = backend.generate_batch(prompts, system=system)
        preds = parse_batch(outs, units, n)
        raws = {u: outs[j] for j, u in enumerate(units)}

    if dyn and dyn.get("enabled"):
        if capper_prompt is None:
            raise ValueError("dynamic_capper is enabled but no capper prompt was given.")
        for u in units:
            preds[u] = dynamic_cap(backend, capper_prompt, sentences, u, preds[u], system)
    return preds, raws
