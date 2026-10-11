"""Fail-fast checks before a run.

Catches the prerequisites that would otherwise crash mid-run, after a model is
loaded and GPU time is being paid for:

  1. the evaluation data exists (silver file for document experiments);
  2. the exemplar pool exists, if any one-shot variant is in the grid;
  3. the rationale cache holds a trace for every _trace variant in the grid;
  4. the model alias resolves for the chosen profile, and its credentials are
     present (OPENAI_API_KEY for an endpoint; an HF token is only warned about).

It resolves the real grid against the real prompt set, so it checks exactly the
variants that would run.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from .backends import DEFAULT_MODELS_YAML, resolve_model
from .config import load_experiment
from .grid import resolve_variants
from .paths import named_config, resolve
from .prompts import Shot, load_prompt_set

TRACE_SUFFIX = "_trace"


def check(
    experiment: str,
    model: str,
    grid: Optional[str] = None,
    models_yaml: str = DEFAULT_MODELS_YAML,
    profile: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). Errors block the run; warnings never do."""
    errors: List[str] = []
    warnings: List[str] = []

    exp = load_experiment(experiment)
    variants = resolve_variants(
        named_config("grids", grid or exp.grid), load_prompt_set(exp.prompt_set)
    )
    if not variants:
        errors.append("the grid resolves to zero variants")

    one_shot = any(v.shot is Shot.ONE for v in variants)

    if exp.task == "document":
        eval_path = resolve(exp.dataset["eval_silver"])
        if not eval_path.is_file():
            errors.append(
                f"missing evaluation silver file: {eval_path}\n"
                f"    fix: extsum silver --experiment {exp.name}"
            )
        if one_shot:
            exemplar = exp.dataset.get("exemplar_silver")
            if not exemplar or not resolve(exemplar).is_file():
                errors.append(
                    f"one-shot variants are in the grid but the exemplar silver "
                    f"file is missing: {exemplar!r}\n"
                    f"    fix: extsum silver --experiment {exp.name}"
                )
        try:
            from .tasks.document import load_cap_k
            load_cap_k(exp.cap)
        except (KeyError, FileNotFoundError) as exc:
            errors.append(f"cannot determine K: {exc}")

    units = exp.dataset.get("aspects", ["summary"]) if exp.task == "aspect" else ["summary"]
    missing = sorted(
        f"{technique}__{unit}.json"
        for technique in {
            v.technique[: -len(TRACE_SUFFIX)]
            for v in variants if v.technique.endswith(TRACE_SUFFIX)
        }
        for unit in units
        if not (exp.rationale_dir / f"{technique}__{unit}.json").is_file()
    )
    if missing:
        errors.append(
            f"_trace variants are in the grid but {len(missing)} rationale file(s) "
            f"are missing from {exp.rationale_dir}: {', '.join(missing)}\n"
            f"    fix A: extsum rationales --experiment {exp.name}   (needs OPENAI_*)\n"
            f"    fix B: set the *_trace techniques to [] in the grid to skip them"
        )

    try:
        settings = resolve_model(model, models_yaml, profile)
    except KeyError as exc:
        errors.append(str(exc.args[0]))
    else:
        backend = settings.get("backend")
        if backend == "endpoint":
            for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
                if not os.environ.get(var):
                    errors.append(f"model {model!r} runs on an endpoint but {var} is unset")
        elif not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            warnings.append("HF_TOKEN unset: gated Gemma/Qwen weight downloads may fail")
        if backend in ("hf", "local"):
            warnings.append(
                f"model {model!r} resolves to the Transformers backend "
                f"(profile {settings.get('profile')!r}): no batching, several s/doc"
            )

    return errors, warnings


def report(errors: List[str], warnings: List[str], label: str) -> bool:
    """Print the outcome; True when the run may proceed."""
    for warning in warnings:
        print(f"[preflight][warn] {warning}")
    if errors:
        print("\n[preflight] BLOCKED — fix these before running:\n")
        for number, error in enumerate(errors, 1):
            print(f"  {number}. {error}\n")
        return False
    print(f"[preflight] OK — {label} is ready to run.")
    return True
