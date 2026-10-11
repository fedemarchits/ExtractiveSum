"""Prompt wording is frozen.

Every published number was produced by a specific prompt text. These hashes
were recorded from the original Part A / Part C prompt code for a fixed input
(tests/prompt_fixture.py) rendered with the checked-in rationale caches; any
refactor that changes a single character of a prompt — or of a technique's
system prompt — fails here.

If a wording change is intended, it defines a new experiment: regenerate the
golden file deliberately and say so in the commit.
"""
import hashlib
import json
from pathlib import Path

import pytest

import prompt_fixture as F
from extsum.paths import resolve
from extsum.prompts import Cap, Exemplar, RenderCtx, Shot, load_prompt_set
from extsum.prompts.rationale import configure_rationale_cache

GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.parametrize("name", sorted(F.SETS))
def test_prompts_match_golden(name):
    cache_dir, units = F.SETS[name]
    configure_rationale_cache(resolve(cache_dir))
    prompt_set = load_prompt_set(name)
    techniques = prompt_set.all_techniques()
    exemplar = Exemplar(
        sentences=F.EXEMPLAR_SENTENCES,
        gold_indices=F.EXEMPLAR_GOLD,
        near_miss_indices=F.EXEMPLAR_NEAR_MISS if name == "bill" else (),
    )

    rendered = {}
    for v in prompt_set.all_variants():
        ctx = RenderCtx(
            shot=v.shot,
            cap=v.cap,
            k=F.CAP_K if v.cap is Cap.CAPPED else None,
            exemplar=exemplar if v.shot is Shot.ONE else None,
        )
        technique = techniques[v.technique]
        for unit in units:
            text = technique.build(F.SENTENCES, unit, ctx)
            payload = repr(technique.system_override) + "\x00" + text
            rendered[f"{v.slug}::{unit}"] = hashlib.sha256(payload.encode()).hexdigest()

    golden = json.loads((GOLDEN / f"prompts_{name}.json").read_text())
    assert set(rendered) == set(golden)
    changed = sorted(key for key in golden if rendered[key] != golden[key])
    assert not changed, f"{len(changed)} prompts changed, e.g. {changed[:5]}"
