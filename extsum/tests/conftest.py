"""Shared fixtures. Nothing here needs a GPU, a model, or the network."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Tests address the checked-in configs and data wherever pytest is started from.
os.environ.setdefault("EXTSUM_ROOT", str(PROJECT))


class FakeBackend:
    """Deterministic stand-in for a model: the answer depends only on the
    prompt, so two runs agree and different prompts disagree."""

    def __init__(self, full_batch: bool = False, max_index: int = 12):
        self.wants_full_batch = full_batch
        self.max_index = max_index
        self.calls = 0

    def generate_batch(self, prompts, system=None):
        self.calls += 1
        out = []
        for prompt in prompts:
            digest = hashlib.sha256(((system or "") + "\x00" + prompt).encode()).hexdigest()
            rng = random.Random(digest)
            picks = [rng.randint(0, self.max_index) for _ in range(rng.randint(0, 9))]
            out.append("Some reasoning.\n" + json.dumps({"selected_sentences": picks}))
        return out

    def free_memory(self) -> None:
        pass


@pytest.fixture
def fake_backend():
    return FakeBackend


@pytest.fixture
def toy_experiment(tmp_path):
    """A self-contained document experiment: 6 tiny documents with silver
    labels, an exemplar pool, a rationale cache and its own results directory.
    Returns the path of its experiment YAML."""
    import pandas as pd
    import yaml

    def frame(prefix, n_docs):
        rows = []
        for d in range(n_docs):
            sentences = [f"{prefix} document {d} sentence {i} about topic {i % 3}." for i in range(1, 9)]
            rows.append({
                "id": f"{prefix}-{d}",
                "dataset": "toy",
                "source_sentences": sentences,
                "reference_summary": f"{prefix} document {d} about topic 1 and topic 2.",
                "silver_indices": [1, 2, 5],
            })
        return pd.DataFrame(rows)

    frame("eval", 6).to_parquet(tmp_path / "eval.parquet")
    frame("pool", 4).to_parquet(tmp_path / "pool.parquet")

    rationales = tmp_path / "rationales"
    rationales.mkdir()
    for technique in ("chain_of_thought", "self_ask", "scoring_based", "salience_inference"):
        (rationales / f"{technique}__summary.json").write_text(json.dumps({
            "technique": technique, "aspect": "summary", "source_model": "reference",
            "exemplar_sentences": ["A is slow.", "We propose B.", "B wins."],
            "shown_indices": [3], "gold_indices": [3], "f1": 1.0,
            "rationale": "Sentence 3 states the result.",
        }))

    config = {
        "name": "toy",
        "task": "document",
        "prompt_set": "document",
        "grid": "document_full",
        "dataset": {
            "name": "toy",
            "eval_silver": str(tmp_path / "eval.parquet"),
            "exemplar_silver": str(tmp_path / "pool.parquet"),
        },
        "cap": {"k": 2},
        "fewshot": {"seed": 42},
        "rationale": {"cache_dir": str(rationales)},
        "inference": {"max_new_tokens": 64, "temperature": 0.0, "seed": 42},
        "output": {
            "results_dir": str(tmp_path / "results"),
            "reports_dir": str(tmp_path / "reports"),
            "resume": True,
        },
    }
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(config))
    return path
