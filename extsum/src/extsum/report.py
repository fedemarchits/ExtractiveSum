"""Aggregate per-document JSONL results into one tidy summary table.

    results/<experiment>/<model>/<variant>.jsonl  ->  reports/<experiment>/summary.csv

One row per (model, variant) — and per aspect for the aspect task — with mean
P/R/F1 and ROUGE. Raw JSONL files are large and not kept in git, so a machine
may hold only some models' files: models that are in the existing summary but
have no raw files here are carried over unchanged instead of being dropped.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .config import Experiment, load_experiment

METRICS = [
    "precision", "recall", "f1",
    "rouge1_model", "rouge2_model", "rougeL_model",
    "rouge1_oracle", "rouge2_oracle", "rougeL_oracle", "oracle_gap_rougeL",
]


def read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_slug(slug: str):
    parts = slug.split("__") + ["", ""]
    return parts[0], parts[1], parts[2]


def summarize(exp: Experiment):
    import pandas as pd

    rows: List[Dict] = []
    for jsonl in sorted(exp.results_dir.glob("*/*.jsonl")):
        records = read_jsonl(jsonl)
        if not records:
            continue
        technique, shot, cap = split_slug(jsonl.stem)
        meta_path = jsonl.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        df = pd.DataFrame(records)
        key = {"model": jsonl.parent.name, "technique": technique, "shot": shot, "cap": cap}

        if exp.task == "aspect":
            for aspect, group in df.groupby("aspect"):
                row = dict(key, aspect=aspect, n=len(group),
                           mean_latency_s=meta.get("mean_latency_s"))
                row.update({m: float(group[m].mean()) for m in METRICS if m in group})
                rows.append(row)
        else:
            row = dict({"dataset": exp.name}, **key, n=len(df),
                       mean_latency_s=meta.get("mean_latency_s"), cap_k=meta.get("cap_k"))
            row.update({m: float(df[m].mean()) for m in METRICS if m in df.columns})
            rows.append(row)

    return pd.DataFrame(rows)


def write_summary(experiment: str) -> Path:
    import pandas as pd

    exp = load_experiment(experiment)
    out_path = exp.reports_dir / "summary.csv"
    fresh = summarize(exp)

    if out_path.exists():
        previous = pd.read_csv(out_path)
        have = set(fresh["model"]) if len(fresh) else set()
        carried = previous[~previous["model"].isin(have)]
        if len(carried):
            print(
                "[report] no raw files here for: "
                f"{', '.join(sorted(carried['model'].unique()))} "
                "(rows carried over from the existing summary)"
            )
            fresh = pd.concat([fresh, carried], ignore_index=True)

    if fresh.empty:
        raise FileNotFoundError(f"No result files under {exp.results_dir}")

    order = [c for c in ("dataset", "model", "technique", "shot", "cap", "aspect") if c in fresh]
    fresh = fresh.sort_values(order).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(out_path, index=False)
    print(f"[report] wrote {out_path} ({len(fresh)} rows)")
    return out_path
