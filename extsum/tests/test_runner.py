"""End-to-end on a toy experiment with a fake model: run, resume, cap, report,
re-score, preflight. No GPU, no network."""
import json

from extsum import preflight
from extsum.config import load_experiment
from extsum.report import write_summary
from extsum.rescore import rescore
from extsum.runner import run

GRID = "billsum_large"  # 12 variants: zero/one-shot, capped/uncapped, tool variants


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_writes_one_row_per_document_and_respects_the_cap(toy_experiment, fake_backend):
    out_dir = run("fake", str(toy_experiment), grid=GRID, backend=fake_backend())
    files = sorted(out_dir.glob("*.jsonl"))
    assert len(files) == 12 and not list(out_dir.glob("*.partial"))

    for path in files:
        rows = _rows(path)
        assert [r["doc_idx"] for r in rows] == list(range(6))
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        assert meta["n_docs"] == 6 and meta["cap_k"] == 2 and meta["variant"] == path.stem
        for r in rows:
            assert r["silver_indices"] == [1, 2, 5]
            assert r["pred_indices"] == sorted(set(r["pred_indices"]))
            assert all(1 <= i <= r["num_sentences"] for i in r["pred_indices"])
            assert isinstance(r["raw_response"], str)
            if path.stem.endswith("__capped"):
                assert len(r["pred_indices"]) <= 2
            tp = len(set(r["pred_indices"]) & {1, 2, 5})
            assert r["tp"] == tp


def test_batched_and_per_document_paths_agree(toy_experiment, fake_backend, tmp_path):
    import yaml

    per_doc = run("a", str(toy_experiment), grid=GRID, backend=fake_backend(full_batch=False))
    batched = run("b", str(toy_experiment), grid=GRID, backend=fake_backend(full_batch=True))
    for path in sorted(per_doc.glob("*.jsonl")):
        assert _rows(path) == _rows(batched / path.name), path.name


def test_resume_skips_finished_variants_and_limit_truncates(toy_experiment, fake_backend):
    backend = fake_backend()
    out_dir = run("fake", str(toy_experiment), grid=GRID, backend=backend, limit=2)
    assert all(len(_rows(p)) == 2 for p in out_dir.glob("*.jsonl"))
    calls = backend.calls
    run("fake", str(toy_experiment), grid=GRID, backend=backend, limit=2)
    assert backend.calls == calls  # nothing re-run


def test_report_and_rescore_roundtrip(toy_experiment, fake_backend):
    import pandas as pd

    out_dir = run("fake", str(toy_experiment), grid=GRID, backend=fake_backend())
    summary = pd.read_csv(write_summary(str(toy_experiment)))
    assert len(summary) == 12 and set(summary["model"]) == {"fake"}
    assert summary["n"].eq(6).all() and summary["cap_k"].eq(2).all()

    # Re-scoring freshly written results changes nothing.
    assert rescore(str(toy_experiment)) == 0

    # Corrupt one stored selection: re-scoring restores the row exactly.
    path = sorted(out_dir.glob("*__capped.jsonl"))[0]
    rows = _rows(path)
    target = next(i for i, r in enumerate(rows) if r["pred_indices"])
    broken = dict(rows[target], pred_indices=[], pred_length=0, f1=0.0, pred_text="")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows[:target] + [broken] + rows[target + 1:]))
    assert rescore(str(toy_experiment), dry_run=True) == 1
    assert _rows(path)[target]["pred_indices"] == []  # dry run wrote nothing
    assert rescore(str(toy_experiment)) == 1
    assert _rows(path) == rows


def test_preflight_passes_on_a_complete_experiment_and_names_what_is_missing(toy_experiment, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    errors, _ = preflight.check(str(toy_experiment), "gemma4_31b")
    assert errors == []

    exp = load_experiment(str(toy_experiment))
    (exp.rationale_dir / "self_ask__summary.json").unlink()
    errors, _ = preflight.check(str(toy_experiment), "gemma4_31b")
    assert len(errors) == 1 and "self_ask__summary.json" in errors[0]

    # ...but a grid without _trace variants does not need the cache
    errors, _ = preflight.check(str(toy_experiment), "gemma4_31b", grid=GRID)
    assert errors == []

    monkeypatch.delenv("OPENAI_API_KEY")
    errors, _ = preflight.check(str(toy_experiment), "gemma4_31b", grid=GRID)
    assert any("OPENAI_API_KEY" in e for e in errors)
