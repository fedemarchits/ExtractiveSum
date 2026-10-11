"""Re-score saved predictions from their raw model responses (no GPU needed).

Every result row keeps the model's raw response. When the answer parser changes
(e.g. accepting ["1", "13"] as well as [1, 13]), this re-parses each response
with the current parser and rebuilds the affected rows through the same task
code the runner uses, so a re-scored file is exactly what a fresh run would
have written for those responses. Rows whose selection is unchanged are left
byte-for-byte alone.

Rebuilding rows needs the source sentences, i.e. the task's dataset. With
--dry-run only the F1 change is reported, from the labels stored in the rows.

    extsum rescore --experiment billsum --dry-run
    extsum rescore --experiment billsum
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional

from . import metrics as M
from .config import load_experiment
from .postprocess import cap_indices_in_order, parse_selection
from .report import read_jsonl
from .tasks import Task, get_task

_DERIVED_UNITS = ("union",)  # rows computed from other rows, with no response


def _unit(row: Dict) -> str:
    return row.get("aspect", "summary")


def _labels(row: Dict) -> List[int]:
    return row["silver_indices"] if "silver_indices" in row else row["gold_indices"]


def _reparse(row: Dict, cap: Optional[int]) -> Optional[List[int]]:
    """The selection the current parser gives for this row (None: not re-parsable)."""
    raw = row.get("raw_response")
    if not isinstance(raw, str):  # derived row, or a self-consistency sample list
        return None
    return cap_indices_in_order(parse_selection(raw, int(row["num_sentences"])), cap)


def rescore(experiment: str, dry_run: bool = False) -> int:
    """Returns the number of rows whose selection changed."""
    exp = load_experiment(experiment)
    files = sorted(exp.results_dir.glob("*/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No result files under {exp.results_dir}")

    task: Optional[Task] = None

    def loaded_task() -> Task:
        nonlocal task
        if task is None:
            task = get_task(exp)
            task.load(needs_exemplars=False)
        return task

    per_model: Dict[str, List[float]] = {}
    total_rows = total_changed = 0
    oracle_cache: Dict = {}

    for path in files:
        capped = path.stem.endswith("__capped")
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        def cap_for(unit: str) -> Optional[int]:
            if not capped:
                return None
            if meta.get("cap_k"):
                return int(meta["cap_k"])
            if unit in (meta.get("caps") or {}):
                return int(meta["caps"][unit])
            return loaded_task().cap(unit)

        rows = read_jsonl(path)
        by_doc: Dict[int, List[int]] = defaultdict(list)
        for position, row in enumerate(rows):
            by_doc[int(row["doc_idx"])].append(position)

        old_f1 = sum(r["f1"] for r in rows if _unit(r) not in _DERIVED_UNITS)
        new_f1 = old_f1
        changed = 0
        rebuilt = list(rows)

        for doc_idx, positions in by_doc.items():
            preds, raws, doc_changed = {}, {}, False
            for position in positions:
                row = rows[position]
                unit = _unit(row)
                if unit in _DERIVED_UNITS:
                    continue
                new = _reparse(row, cap_for(unit))
                if new is None:
                    new = row["pred_indices"]
                elif new != row["pred_indices"]:
                    doc_changed = True
                    changed += 1
                    new_f1 += M.prf(_labels(row), new)["f1"] - row["f1"]
                preds[unit], raws[unit] = new, row.get("raw_response")

            if not doc_changed or dry_run:
                continue

            t = loaded_task()
            doc = t.doc(doc_idx)
            first = rows[positions[0]]
            if t.doc_id(doc) != str(first["doc_id"]) or len(t.sentences(doc)) != first["num_sentences"]:
                raise ValueError(
                    f"{path}: document {first['doc_id']!r} does not match the "
                    f"dataset loaded for experiment {exp.name!r}"
                )
            fresh = t.rows(doc, doc_idx, preds, raws, oracle_cache)
            if len(fresh) != len(positions):
                raise ValueError(f"{path}: row count changed for document {doc_idx}")
            for position, row in zip(positions, fresh):
                rebuilt[position] = row

        n = sum(1 for r in rows if _unit(r) not in _DERIVED_UNITS)
        stats = per_model.setdefault(path.parent.name, [0.0, 0.0, 0, 0])
        stats[0] += old_f1
        stats[1] += new_f1
        stats[2] += n
        stats[3] += changed
        total_rows += n
        total_changed += changed

        if changed and not dry_run:
            tmp = path.with_suffix(".jsonl.rescore")
            tmp.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rebuilt),
                encoding="utf-8",
            )
            tmp.replace(path)

    for model, (old_f1, new_f1, n, changed) in sorted(per_model.items()):
        print(
            f"[rescore] {exp.name}/{model}: rows={n} changed={changed} "
            f"({100 * changed / max(n, 1):.1f}%)  mean F1 {old_f1 / max(n, 1):.3f} -> {new_f1 / max(n, 1):.3f}"
        )
    print(
        f"[rescore] total rows={total_rows} changed={total_changed}"
        + ("  (dry run, nothing written)" if dry_run else "")
    )
    return total_changed
