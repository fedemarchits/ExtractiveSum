"""Whole-document extractive summarization on silver labels.

One selection per document. The labels are silver: sentences picked by the
`local_score` heuristic from an abstractive dataset (see extsum.silver). K comes
from the dataset's heuristic JSON; exemplars come from a separate silver file
built from a split that is never evaluated.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .. import metrics as M
from ..config import Experiment
from ..data import silver_io as S
from ..fewshot.silver import select_exemplar
from ..paths import resolve
from ..prompts import Exemplar
from .base import SelectionMap, Task

#: Prompt builders take an aspect argument; a whole-document task has one unit.
SUMMARY_UNIT = "summary"


def load_cap_k(cap_cfg: Dict[str, Any]) -> int:
    """K from an explicit ``k`` or from the heuristic JSON's ``best_k``."""
    if "k" in cap_cfg:
        return int(cap_cfg["k"])
    if "heuristic" not in cap_cfg:
        raise KeyError("cap needs either 'k' or 'heuristic' (path to a JSON with best_k).")
    path = resolve(cap_cfg["heuristic"])
    if not path.is_file():
        raise FileNotFoundError(f"Heuristic configuration not found: {path}")
    heuristic = json.loads(path.read_text(encoding="utf-8"))
    if "best_k" not in heuristic:
        raise KeyError(f"'best_k' is missing from heuristic configuration: {path}")
    return int(heuristic["best_k"])


class DocumentTask(Task):
    kind = "document"
    units = (SUMMARY_UNIT,)

    def __init__(self, experiment: Experiment):
        dataset = experiment.dataset
        self.dataset_name = str(dataset["name"])
        self._eval_path = resolve(dataset["eval_silver"])
        self._exemplar_path = (
            resolve(dataset["exemplar_silver"]) if dataset.get("exemplar_silver") else None
        )
        self._cap_k = load_cap_k(experiment.cap)
        self._eval = None
        self._pool = None

    def load(self, needs_exemplars: bool) -> None:
        if not self._eval_path.is_file():
            raise FileNotFoundError(
                f"Evaluation silver file not found: {self._eval_path}. "
                f"Build it with: extsum silver --dataset {self.dataset_name}"
            )
        self._eval = S.load_split(self._eval_path)

        if not needs_exemplars:
            return
        if self._exemplar_path is None:
            raise RuntimeError(
                "The grid contains one-shot variants, but dataset.exemplar_silver "
                "is not set in the experiment config."
            )
        if not self._exemplar_path.is_file():
            raise FileNotFoundError(
                f"Exemplar silver file not found: {self._exemplar_path}"
            )
        self._pool = S.load_split(self._exemplar_path)
        if len(self._pool) == 0:
            raise ValueError(f"Exemplar silver file is empty: {self._exemplar_path}")

    def __len__(self) -> int:
        return len(self._eval)

    def doc(self, index: int) -> Any:
        return self._eval.iloc[index]

    def doc_id(self, doc: Any) -> str:
        return S.get_doc_id(doc)

    def sentences(self, doc: Any) -> List[str]:
        return S.get_source_sentences(doc)

    def cap(self, unit: str) -> int:
        return self._cap_k

    def exemplar(self, unit: str, seed: int) -> Optional[Exemplar]:
        """A one-shot variant never silently falls back to zero-shot."""
        if self._pool is None:
            raise RuntimeError("Exemplar pool was not loaded.")
        exemplar = select_exemplar(self._pool, unit, seed=seed)
        if exemplar is None:
            raise RuntimeError("No valid one-shot exemplar could be selected.")
        return exemplar

    def rows(self, doc, doc_idx, preds: SelectionMap, raws, oracle_cache) -> List[Dict]:
        doc_id = S.get_doc_id(doc)
        sentences = S.get_source_sentences(doc)
        reference = S.get_reference_summary(doc)
        silver = S.gold_for_doc(doc)
        pred = preds[SUMMARY_UNIT]

        pred_text = S.indices_to_text(sentences, pred)
        exact = M.prf(silver, pred)

        rouge_model: Dict[str, float] = {}
        rouge_oracle: Dict[str, float] = {}
        oracle_indices: List[int] = []
        if reference:
            rouge_model = M.rouge(reference, pred_text)
            key = (self.dataset_name, doc_id, "oracle")
            if key not in oracle_cache:
                oracle_cache[key] = M.build_oracle_indices(sentences, reference)
            oracle_indices = oracle_cache[key]
            rouge_oracle = M.rouge(reference, S.indices_to_text(sentences, oracle_indices))

        return [{
            "doc_id": doc_id,
            "doc_idx": doc_idx,
            "dataset": self.dataset_name,
            "num_sentences": len(sentences),
            "pred_length": len(pred),
            "silver_length": len(silver),
            "pred_indices": pred,
            "silver_indices": sorted(silver),
            # Reconstructed summaries, kept for semantic metrics and inspection.
            "pred_text": pred_text,
            "silver_text": S.indices_to_text(sentences, silver),
            "reference_summary": reference,
            "precision": exact["precision"],
            "recall": exact["recall"],
            "f1": exact["f1"],
            "tp": exact["tp"],
            "fp": exact["fp"],
            "fn": exact["fn"],
            "rouge1_model": rouge_model.get("rouge1", 0.0),
            "rouge2_model": rouge_model.get("rouge2", 0.0),
            "rougeL_model": rouge_model.get("rougeL", 0.0),
            "rouge1_oracle": rouge_oracle.get("rouge1", 0.0),
            "rouge2_oracle": rouge_oracle.get("rouge2", 0.0),
            "rougeL_oracle": rouge_oracle.get("rougeL", 0.0),
            "oracle_gap_rougeL": (
                rouge_oracle.get("rougeL", 0.0) - rouge_model.get("rougeL", 0.0)
                if rouge_oracle else 0.0
            ),
            "oracle_indices": oracle_indices,
            # Raw model output, kept so answers can be re-parsed later.
            "raw_response": raws[SUMMARY_UNIT],
        }]

    def meta(self) -> Dict[str, Any]:
        return {"dataset": self.dataset_name, "cap_k": self._cap_k}
