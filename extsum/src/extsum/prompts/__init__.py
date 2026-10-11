"""Prompt techniques, grouped into interchangeable *prompt sets*.

A prompt set is one wording of the same technique grid for one kind of source:

    aspect    one ACLSum aspect of a scientific paper
    document  a whole news-style document
    bill      a whole legislative bill

Every set registers the same technique names, so grids, results and analyses
line up across sets. The types, registry, TF-IDF centrality and rationale cache
are shared; only the wording differs.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .base import Cap, Exemplar, RenderCtx, Shot, Technique, Variant, expand

PROMPT_SETS = ("aspect", "document", "bill")


@dataclass(frozen=True)
class PromptSet:
    name: str
    all_techniques: Callable[[], Dict[str, Technique]]
    all_variants: Callable[[], List[Variant]]
    dynamic_capper_prompt: Callable[[Sequence[str], str, Sequence[int]], str]


def load_prompt_set(name: str) -> PromptSet:
    if name not in PROMPT_SETS:
        raise ValueError(
            f"Unknown prompt set {name!r}. Available: {', '.join(PROMPT_SETS)}"
        )
    package = f"{__name__}.sets.{name}"
    registry = importlib.import_module(f"{package}.registry")
    wrappers = importlib.import_module(f"{package}.wrappers")
    return PromptSet(
        name=name,
        all_techniques=registry.all_techniques,
        all_variants=registry.all_variants,
        dynamic_capper_prompt=wrappers.dynamic_capper_prompt,
    )


__all__ = [
    "PROMPT_SETS", "PromptSet", "load_prompt_set",
    "Cap", "Exemplar", "RenderCtx", "Shot", "Technique", "Variant", "expand",
]
