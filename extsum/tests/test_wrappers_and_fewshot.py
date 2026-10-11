"""Wrapper mechanisms and exemplar selection — no models."""
from extsum.fewshot import aspect as aspect_shots
from extsum.fewshot import silver as silver_shots
from extsum.wrappers import dynamic_cap, majority_vote, select_document


class Scripted:
    """Returns a fixed list per generate_batch call; cycles through `scripts`."""
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = 0

    def generate_batch(self, prompts, system=None):
        out = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        return out


def _capper(sentences, unit, selected):
    return f"prune {list(selected)}"


def test_majority_vote_threshold():
    samples = [[1, 2], [1, 3], [1, 2]]  # 1 x3, 2 x2, 3 x1
    assert majority_vote(samples, 3) == [1, 2]
    assert majority_vote(samples, 3, threshold=3) == [1]


def test_dynamic_cap_never_adds_and_falls_back():
    sents = ["s1", "s2", "s3", "s4"]
    # prunes [1,2,3] -> keeps [2]; the hallucinated 4 is dropped
    be = Scripted([['{"selected_sentences": [2, 4]}']])
    assert dynamic_cap(be, _capper, sents, "challenge", [1, 2, 3], None) == [2]
    assert dynamic_cap(be, _capper, sents, "challenge", [], None) == []
    assert dynamic_cap(Scripted([["no json"]]), _capper, sents, "challenge", [1, 3], None) == [1, 3]


def test_self_consistency_takes_the_majority_and_keeps_every_sample():
    be = Scripted([
        ['{"selected_sentences": [1, 2]}'],
        ['{"selected_sentences": [1]}'],
        ['{"selected_sentences": [1, 2]}'],
    ])
    preds, raws = select_document(
        be, ["p"], ["challenge"], ["a", "b", "c"], {"enabled": True, "n_samples": 3}, None, None
    )
    assert preds["challenge"] == [1, 2]
    assert isinstance(raws["challenge"], list) and len(raws["challenge"]) == 3


def test_plain_pass_returns_one_raw_string_and_does_not_cap():
    be = Scripted([['{"selected_sentences": [1, 2, 3, 4]}']])
    preds, raws = select_document(be, ["p"], ["challenge"], ["a", "b", "c", "d"], None, None, None)
    assert preds["challenge"] == [1, 2, 3, 4]  # the fixed cap is the runner's job
    assert isinstance(raws["challenge"], str)


def test_dynamic_capper_prunes_through_the_prompt_set_prompt():
    be = Scripted([['{"selected_sentences": [1, 2, 3]}'], ['{"selected_sentences": [2]}']])
    preds, _ = select_document(
        be, ["p"], ["summary"], ["a", "b", "c"], None, {"enabled": True}, None, capper_prompt=_capper
    )
    assert preds["summary"] == [2]


_ACL = [
    {"source_sentences": ["a", "b", "c"], "challenge_labels": [0, 0, 0]},
    {"source_sentences": ["d", "e"],      "challenge_labels": [1, 0]},
]


def test_aspect_exemplar_prefers_a_doc_with_gold():
    ex = aspect_shots.select_exemplar(_ACL, "challenge", seed=42)
    assert list(ex.gold_indices) == [1] and list(ex.sentences) == ["d", "e"]
    assert aspect_shots.gold_indices_1based({"challenge_labels": [0, 1, 1]}, "challenge") == [2, 3]


def test_aspect_exemplars_are_deterministic_and_always_have_gold():
    a = [list(e.gold_indices) for e in aspect_shots.iter_exemplars(_ACL, "challenge", seed=1)]
    b = [list(e.gold_indices) for e in aspect_shots.iter_exemplars(_ACL, "challenge", seed=1)]
    assert a == b and all(a)


_SILVER = [
    {"source_sentences": ["x"], "silver_indices": [], "reference_summary": "x"},
    {"source_sentences": ["rates rose", "weather was mild", "rates may rise again"],
     "silver_indices": [1], "reference_summary": "rates rose and may rise again"},
]


def test_silver_exemplar_skips_unlabelled_docs_and_adds_near_misses():
    ex = silver_shots.select_exemplar(_SILVER, "summary", seed=42)
    assert list(ex.gold_indices) == [1]
    # hard negatives: unselected sentences ranked by overlap with the reference
    assert list(ex.near_miss_indices)[0] == 3 and 1 not in ex.near_miss_indices
