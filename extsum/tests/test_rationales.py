"""Rationale builder: best-overlap acceptance and resume, with a scripted model."""
import json

from extsum.config import load_experiment
from extsum.rationales.builder import build


class Reference:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def generate_batch(self, prompts, system=None):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return [answer]

    def free_memory(self):
        pass


GOOD = 'Reasoning:\nSentences 1, 2 and 5 carry the topic.\n\nFinal JSON:\n{"selected_sentences": [1, 2, 5]}'
WEAK = 'Reasoning:\nOnly the first matters.\n\nFinal JSON:\n{"selected_sentences": [1, 7, 8]}'
BAD = 'Reasoning:\nNone of them.\n\nFinal JSON:\n{"selected_sentences": [7, 8]}'


def _clear(experiment):
    cache = load_experiment(str(experiment)).rationale_dir
    for path in cache.glob("*.json"):
        path.unlink()
    return cache


def test_first_trace_above_accept_f1_is_cached_and_runs_resume(toy_experiment):
    cache = _clear(toy_experiment)
    backend = Reference([WEAK, GOOD])
    build(str(toy_experiment), model="ref", backend=backend)

    shot = json.loads((cache / "chain_of_thought__summary.json").read_text())
    assert shot["shown_indices"] == [1, 2, 5] and shot["f1"] == 1.0
    assert shot["rationale"] == "Sentences 1, 2 and 5 carry the topic."
    assert shot["source_model"] == "ref" and shot["aspect"] == "summary"
    assert len(list(cache.glob("*.json"))) == 4

    calls = backend.calls
    build(str(toy_experiment), model="ref", backend=backend)
    assert backend.calls == calls  # everything cached: no new calls


def test_nothing_is_written_when_no_trace_reaches_the_floor(toy_experiment):
    cache = _clear(toy_experiment)
    build(str(toy_experiment), model="ref", backend=Reference([BAD]))
    assert list(cache.glob("*.json")) == []
