"""Resolve a grid YAML + a prompt set's technique registry into active Variants.

Grid YAML, per technique:
  <name>: all                          -> the technique's full shot x cap grid
  <name>: [{shot: .., cap: ..}, ...]   -> only those cells
  <name>: []                           -> disabled
  (absent)                             -> default to the full grid
Requested cells are always intersected with what the technique declares valid
(e.g. negative_aware is one-shot only), so an over-broad grid can't create an
invalid variant. Because absent means "all", a partial grid must zero the
techniques it drops.
"""
from __future__ import annotations

from typing import Dict, List

import yaml

from .paths import PathLike, resolve
from .prompts import Cap, PromptSet, Shot, Variant, expand


def resolve_variants(grid_yaml: PathLike, prompt_set: PromptSet) -> List[Variant]:
    with resolve(grid_yaml).open(encoding="utf-8") as fh:
        spec: Dict = (yaml.safe_load(fh) or {}).get("techniques", {})

    techniques = prompt_set.all_techniques()
    unknown = sorted(set(spec) - set(techniques))
    if unknown:
        raise KeyError(
            f"Grid {grid_yaml} names techniques the {prompt_set.name!r} prompt "
            f"set does not define: {unknown}"
        )

    out: List[Variant] = []
    for name, tech in techniques.items():
        valid = {(v.shot, v.cap) for v in expand(tech)}
        entry = spec.get(name, "all")
        if entry == "all" or entry is None:
            cells = valid
        else:
            cells = {
                (Shot(item["shot"]), Cap(item["cap"])) for item in entry
            } & valid
        out.extend(Variant(name, s, c) for (s, c) in valid if (s, c) in cells)
    return out
