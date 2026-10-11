"""Positional bias: where selected sentences sit in the document.

Measured for the reference labels (human gold on ACLSum, `local_score` silver
elsewhere) and for every model, on the same scale, so a model's bias can be
read against the bias its dataset already rewards.

All shares are computed PER ANSWER and then averaged, so a model that returns
forty sentences for one document does not outweigh forty normal answers.

    lead20    mean share of an answer's sentences in the first fifth of the
              document (0.20 if position played no role)
    mean_pos  mean normalised position of a selected sentence (0.50 if none)
    tv        total-variation distance between two position histograms over
              ten equal bins (0 = identical, 1 = disjoint)

For capped variants a model's answer is measured twice: as the model gave it
(`as_answered`) and after the runner kept only the first K sentences
(`after_cap`). The difference is the part of the bias the truncation adds.

Tables are written to reports/position/ when the raw results are present and
read from there otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..paths import resolve

N_BINS = 10
OUT_DIR = "reports/position"
BIN_COLUMNS = [f"bin{i}" for i in range(N_BINS)]


def histogram(indices: Sequence[int], n_sentences: int) -> Optional[List[float]]:
    """Share of the selected sentences in each tenth of the document."""
    valid = [i for i in indices if 1 <= i <= n_sentences]
    if not valid or n_sentences <= 0:
        return None
    counts = [0.0] * N_BINS
    for i in valid:
        counts[min(N_BINS - 1, int((i - 1) / n_sentences * N_BINS))] += 1
    return [c / len(valid) for c in counts]


def mean_position(indices: Sequence[int], n_sentences: int) -> float:
    valid = [i for i in indices if 1 <= i <= n_sentences]
    return sum((i - 0.5) / n_sentences for i in valid) / len(valid)


class Accumulator:
    """Averages per-answer histograms and positions."""

    def __init__(self) -> None:
        self.bins = [0.0] * N_BINS
        self.position = 0.0
        self.length = 0.0
        self.answers = 0
        self.empty = 0

    def add(self, indices: Sequence[int], n_sentences: int) -> None:
        shares = histogram(indices, n_sentences)
        if shares is None:
            self.empty += 1
            return
        self.bins = [a + b for a, b in zip(self.bins, shares)]
        self.position += mean_position(indices, n_sentences)
        self.length += len([i for i in indices if 1 <= i <= n_sentences])
        self.answers += 1

    def shares(self) -> List[float]:
        return [b / max(self.answers, 1) for b in self.bins]

    def row(self) -> Dict[str, float]:
        shares = self.shares()
        row = dict(zip(BIN_COLUMNS, shares))
        row.update(
            lead10=shares[0],
            lead20=shares[0] + shares[1],
            last20=shares[-2] + shares[-1],
            mean_pos=self.position / max(self.answers, 1),
            tv_uniform=0.5 * sum(abs(s - 1 / N_BINS) for s in shares),
            mean_selected=self.length / max(self.answers, 1),
            n_answers=self.answers,
            n_empty=self.empty,
        )
        return row


def tv_distance(a: Iterable[float], b: Iterable[float]) -> float:
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def _rows(path: Path) -> Iterable[Dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _silver_reference(name: str, path: Path, label: str) -> Optional[Dict]:
    import pandas as pd

    if not path.is_file():
        return None
    acc = Accumulator()
    sentences = 0
    frame = pd.read_parquet(path)
    for row in frame.itertuples():
        n = len(row.source_sentences)
        acc.add([int(i) for i in row.silver_indices], n)
        sentences += n
    out = {"dataset": name, "subset": "all", "labels": label,
           "mean_sentences": sentences / max(len(frame), 1)}
    out.update(acc.row())
    return out


def build_tables():
    """Recompute reports/position/{reference,models}.csv from the raw results."""
    import pandas as pd

    from ..config import load_experiment
    from ..postprocess import parse_selection

    reference: List[Dict] = []
    models: List[Dict] = []

    # --- references ---------------------------------------------------------
    for name in ("cnndm", "billsum"):
        exp = load_experiment(name)
        row = _silver_reference(name, resolve(exp.dataset["eval_silver"]), "silver (local_score)")
        if row:
            reference.append(row)
    acl = load_experiment("aclsum")
    acl_files = sorted(acl.results_dir.glob("*/*.jsonl"))
    if acl_files:
        per_aspect: Dict[str, Accumulator] = {}
        pooled = Accumulator()
        sentences = docs = 0
        for row in _rows(acl_files[0]):  # gold labels are the same in every file
            if row["aspect"] == "union":
                sentences += row["num_sentences"]
                docs += 1
                continue
            per_aspect.setdefault(row["aspect"], Accumulator()).add(
                row["gold_indices"], row["num_sentences"]
            )
            pooled.add(row["gold_indices"], row["num_sentences"])
        for subset, acc in [("all", pooled), *sorted(per_aspect.items())]:
            out = {"dataset": "aclsum", "subset": subset, "labels": "human",
                   "mean_sentences": sentences / max(docs, 1)}
            out.update(acc.row())
            reference.append(out)

    reference_frame = pd.DataFrame(reference)

    def reference_bins(dataset: str) -> List[float]:
        match = reference_frame[(reference_frame.dataset == dataset) & (reference_frame.subset == "all")]
        return match[BIN_COLUMNS].iloc[0].tolist()

    # --- models -------------------------------------------------------------
    for name in ("aclsum", "cnndm", "billsum"):
        exp = load_experiment(name)
        for model_dir in sorted(p for p in exp.results_dir.iterdir() if p.is_dir()):
            files = sorted(model_dir.glob("*.jsonl"))
            if not files:
                continue
            acc = {mode: Accumulator() for mode in ("as_answered", "after_cap", "uncapped")}
            over_cap = run_from_one = capped_answers = 0
            for path in files:
                capped = path.stem.endswith("__capped")
                for row in _rows(path):
                    if row.get("aspect") == "union":
                        continue
                    n = row["num_sentences"]
                    if not capped:
                        acc["uncapped"].add(row["pred_indices"], n)
                        continue
                    raw = row.get("raw_response")
                    answered = parse_selection(raw, n) if isinstance(raw, str) else row["pred_indices"]
                    acc["as_answered"].add(answered, n)
                    acc["after_cap"].add(row["pred_indices"], n)
                    capped_answers += 1
                    kept = row["pred_indices"]
                    # the answer was longer than K and the runner cut it
                    over_cap += bool(kept) and len(answered) > len(kept) and answered[:len(kept)] == kept
                    run_from_one += bool(answered) and answered == list(range(1, len(answered) + 1))
            target = reference_bins(name)
            for mode, a in acc.items():
                out = {"dataset": name, "model": model_dir.name, "mode": mode}
                out.update(a.row())
                out["tv_reference"] = tv_distance(a.shares(), target)
                out["excess_lead20"] = out["lead20"] - (target[0] + target[1])
                if mode != "uncapped":
                    out["over_cap_pct"] = 100 * over_cap / max(capped_answers, 1)
                    out["run_from_first_sentence_pct"] = 100 * run_from_one / max(capped_answers, 1)
                models.append(out)

    out_dir = resolve(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_frame.to_csv(out_dir / "reference.csv", index=False)
    models_frame = pd.DataFrame(models)
    models_frame.to_csv(out_dir / "models.csv", index=False)
    return reference_frame, models_frame


def load_tables(rebuild: bool = True):
    """(reference, models) tables; rebuilt when raw results are available."""
    import pandas as pd

    out_dir = resolve(OUT_DIR)
    have_raw = any(resolve("results").glob("*/*/*.jsonl"))
    if rebuild and have_raw:
        return build_tables()
    return pd.read_csv(out_dir / "reference.csv"), pd.read_csv(out_dir / "models.csv")
