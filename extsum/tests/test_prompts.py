"""Prompt sets: registry integrity, rendering of every variant, rationale use."""
import json

import pytest

from extsum.prompts import PROMPT_SETS, Cap, Exemplar, RenderCtx, Shot, expand, load_prompt_set
from extsum.prompts import rationale

EX = Exemplar(
    sentences=["The model is slow.", "We propose a fast solver.", "It wins on all metrics."],
    gold_indices=[2],
    near_miss_indices=[3],
)
DOC = ["We study X.", "Prior work fails on Y.", "Our method fixes Y.", "Results improve by 10%."]
UNIT = {"aspect": "challenge", "document": "summary", "bill": "summary"}
TRACED = ("chain_of_thought", "self_ask", "scoring_based", "salience_inference")


def _ctx(v):
    return RenderCtx(
        shot=v.shot,
        cap=v.cap,
        k=3 if v.cap is Cap.CAPPED else None,
        exemplar=EX if v.shot is Shot.ONE else None,
    )


@pytest.fixture
def trace_cache(tmp_path):
    for unit in set(UNIT.values()) | {"outcome"}:
        for technique in TRACED:
            (tmp_path / f"{technique}__{unit}.json").write_text(json.dumps({
                "technique": technique, "aspect": unit, "source_model": "reference",
                "exemplar_sentences": ["A is slow.", "We propose B.", "B wins."],
                "shown_indices": [3], "gold_indices": [3], "f1": 1.0,
                "rationale": "Sentence 1: a problem -> No\nSentence 3: a result -> Yes",
            }))
    rationale.configure_rationale_cache(tmp_path)
    yield tmp_path
    rationale.configure_rationale_cache("data/rationales")


def test_unknown_prompt_set_is_rejected():
    with pytest.raises(ValueError):
        load_prompt_set("nope")


@pytest.mark.parametrize("name", PROMPT_SETS)
def test_registry_integrity(name):
    prompt_set = load_prompt_set(name)
    slugs = [v.slug for v in prompt_set.all_variants()]
    assert len(slugs) == len(set(slugs))
    vanilla = expand(prompt_set.all_techniques()["vanilla"])
    assert {(v.shot, v.cap) for v in vanilla} == {
        (Shot.ZERO, Cap.UNCAPPED), (Shot.ZERO, Cap.CAPPED),
        (Shot.ONE, Cap.UNCAPPED), (Shot.ONE, Cap.CAPPED),
    }
    for one_shot_only in ("negative_aware",) + tuple(f"{t}_trace" for t in TRACED):
        shots = {v.shot for v in expand(prompt_set.all_techniques()[one_shot_only])}
        assert shots == {Shot.ONE}, one_shot_only


def test_sets_register_their_own_techniques():
    # Separate registries: loading one set must not leak into another.
    assert len(load_prompt_set("aspect").all_techniques()) == 16
    assert len(load_prompt_set("document").all_techniques()) == 20
    assert len(load_prompt_set("bill").all_techniques()) == 20


@pytest.mark.parametrize("name", PROMPT_SETS)
def test_all_variants_render(name, trace_cache):
    for tech in load_prompt_set(name).all_techniques().values():
        for v in expand(tech):
            out = tech.build(DOC, UNIT[name], _ctx(v))
            assert isinstance(out, str) and len(out) > 50
            assert "selected_sentences" in out
            if v.cap is Cap.CAPPED:
                assert "3" in out


def test_trace_technique_shows_the_cached_reasoning(trace_cache):
    techniques = load_prompt_set("aspect").all_techniques()
    ctx = RenderCtx(shot=Shot.ONE, cap=Cap.UNCAPPED, exemplar=EX)
    traced = techniques["chain_of_thought_trace"].build(DOC, "outcome", ctx)
    assert "Reasoning:" in traced and "-> Yes" in traced
    plain = techniques["chain_of_thought"].build(DOC, "outcome", ctx)
    assert "Reasoning:" not in plain  # the base technique stays answer-only


def test_aspect_trace_falls_back_to_answer_only_without_a_cache(tmp_path):
    rationale.configure_rationale_cache(tmp_path)  # empty
    try:
        ctx = RenderCtx(shot=Shot.ONE, cap=Cap.UNCAPPED, exemplar=EX)
        out = load_prompt_set("aspect").all_techniques()["self_ask_trace"].build(DOC, "challenge", ctx)
        assert "Reasoning:" not in out
    finally:
        rationale.configure_rationale_cache("data/rationales")


@pytest.mark.parametrize("name", ["document", "bill"])
def test_document_trace_refuses_to_run_without_a_cache(name, tmp_path):
    # A silent fallback would make a _trace variant identical to its base one.
    rationale.configure_rationale_cache(tmp_path)  # empty
    try:
        ctx = RenderCtx(shot=Shot.ONE, cap=Cap.UNCAPPED, exemplar=EX)
        with pytest.raises(FileNotFoundError):
            load_prompt_set(name).all_techniques()["self_ask_trace"].build(DOC, "summary", ctx)
    finally:
        rationale.configure_rationale_cache("data/rationales")
