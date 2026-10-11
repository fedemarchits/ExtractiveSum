"""Build an experiment's silver files and derive its sentence budget K.

Both read everything from the experiment config, so one command reproduces the
exact evaluation and exemplar sets every model was run on:

    extsum silver --experiment billsum        # eval + exemplar parquet files
    extsum compute-k --experiment billsum     # K from the compression ratio

K (compute-k) is the number of source sentences a reference-length extractive
summary needs. It is estimated on a development split, never on the evaluated
one:

    compression ratio  CR_d = reference-summary words / source-document words
    per-document K     K_d  = CR_d * number of source sentences
    dataset K               = round(median over documents of K_d), at least 1

The median is used because document and summary lengths are both long-tailed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..config import Experiment, load_experiment
from ..paths import resolve

#: Heuristic settings (which heuristic, which ROUGE variant) inherited by compute-k.
BASE_HEURISTIC = "configs/heuristics/aclsum_local_score.json"


def _document_experiment(experiment: str) -> Experiment:
    exp = load_experiment(experiment)
    if exp.task != "document":
        raise ValueError(
            f"{exp.name!r} is an {exp.task!r} task: it uses the dataset's own "
            "labels and needs no silver files."
        )
    return exp


def _heuristic_path(exp: Experiment) -> Path:
    if "heuristic" not in exp.cap:
        raise KeyError(f"cap.heuristic is not set in {exp.path}")
    return resolve(exp.cap["heuristic"])


def build_silver(experiment: str, which: str = "both", overwrite: bool = False) -> List[Path]:
    """Write the eval and/or exemplar silver parquet of a document experiment."""
    from .labelling import generate_silver_dataset, load_best_heuristic

    exp = _document_experiment(experiment)
    dataset = exp.dataset
    heuristic_path = _heuristic_path(exp)
    heuristic = load_best_heuristic(heuristic_path)

    targets = []
    if which in ("eval", "both"):
        targets.append(("eval", dataset.get("eval_split", "test"), dataset["eval_silver"]))
    if which in ("exemplar", "both") and dataset.get("exemplar_silver"):
        targets.append(
            ("exemplar", dataset.get("exemplar_split", "validation"), dataset["exemplar_silver"])
        )

    written: List[Path] = []
    for role, split, out in targets:
        out_path = resolve(out)
        if out_path.exists() and not overwrite:
            print(f"[silver] {role}: {out_path} exists (use --overwrite to rebuild)")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[silver] {role}: {dataset['name']}/{split}  "
            f"{heuristic['best_heuristic']} K={heuristic['best_k']} "
            f"metric={heuristic['extract_metric']}  "
            f"max_docs={dataset.get('max_docs')} seed={dataset.get('sample_seed')}",
            flush=True,
        )
        frame = generate_silver_dataset(
            dataset_name=dataset["name"],
            split=split,
            heuristic_path=heuristic_path,
            out_path=out_path,
            max_docs=dataset.get("max_docs"),
            seed=dataset.get("sample_seed"),
        )
        print(f"[silver] wrote {len(frame)} documents to {out_path}")
        written.append(out_path)
    return written


def compression_stats(
    dataset_name: str,
    split: str,
    max_docs: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict:
    """Compression-ratio statistics of one split and the K they imply."""
    import numpy as np

    from ..data.segmentation import sentence_split
    from .labelling import load_abstractive_dataset

    dataset, article_column, summary_column = load_abstractive_dataset(
        dataset_name=dataset_name, split=split
    )
    if max_docs is not None:
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(max_docs, len(dataset))))

    ratios: List[float] = []
    sentence_counts: List[int] = []
    per_doc_k: List[float] = []
    for example in dataset:
        article = str(example.get(article_column) or "")
        summary = str(example.get(summary_column) or "")
        doc_words, summary_words = len(article.split()), len(summary.split())
        sentences = sentence_split(article, dataset_name=dataset_name)
        if not doc_words or not summary_words or not sentences:
            continue
        ratio = summary_words / doc_words
        ratios.append(ratio)
        sentence_counts.append(len(sentences))
        per_doc_k.append(ratio * len(sentences))

    if not per_doc_k:
        raise RuntimeError(f"No usable documents in {dataset_name}/{split}.")

    return {
        "best_k": max(1, int(round(float(np.median(per_doc_k))))),
        "n_docs": len(per_doc_k),
        "compression_ratio_mean": round(float(np.mean(ratios)), 4),
        "compression_ratio_median": round(float(np.median(ratios)), 4),
        "sentences_per_doc_mean": round(float(np.mean(sentence_counts)), 2),
        "sentences_per_doc_median": float(np.median(sentence_counts)),
        "k_per_doc_mean": round(float(np.mean(per_doc_k)), 3),
        "k_per_doc_median": round(float(np.median(per_doc_k)), 3),
        "k_per_doc_p25": round(float(np.percentile(per_doc_k, 25)), 3),
        "k_per_doc_p75": round(float(np.percentile(per_doc_k, 75)), 3),
    }


def compute_k(
    experiment: str,
    split: Optional[str] = None,
    max_docs: Optional[int] = None,
    seed: Optional[int] = None,
    dry_run: bool = False,
) -> Dict:
    """Derive K on the development split and write it to the experiment's
    heuristic JSON (the file both `extsum silver` and the runner read)."""
    exp = _document_experiment(experiment)
    name = exp.dataset["name"]
    # The exemplar split is the development split: it is never evaluated.
    split = split or exp.dataset.get("exemplar_split", "validation")
    if split == exp.dataset.get("eval_split", "test"):
        raise ValueError("K must not be estimated on the evaluated split.")

    stats = compression_stats(name, split, max_docs, seed)
    print(f"[k] dataset: {name}  split: {split}")
    for key, value in stats.items():
        print(f"[k] {key}: {value}")
    if dry_run:
        return stats

    out_path = _heuristic_path(exp)
    if out_path == resolve(BASE_HEURISTIC):
        raise ValueError(
            f"{exp.name!r} uses the shared {BASE_HEURISTIC}; point cap.heuristic "
            "at a dataset-specific file before deriving its own K."
        )
    base = json.loads(resolve(BASE_HEURISTIC).read_text(encoding="utf-8"))
    heuristic = {
        "best_heuristic": base["best_heuristic"],
        "best_k": stats["best_k"],
        "extract_metric": base.get("extract_metric", "rouge1"),
        "mode": base.get("mode", "singular"),
        "k_selection": {
            "method": "compression_ratio",
            "dataset": name,
            "split": split,
            "max_docs": max_docs,
            "seed": seed,
            **{key: value for key, value in stats.items() if key != "best_k"},
        },
    }
    out_path.write_text(json.dumps(heuristic, indent=2) + "\n", encoding="utf-8")
    print(f"[k] wrote {out_path} (best_k = {stats['best_k']})")
    return stats
