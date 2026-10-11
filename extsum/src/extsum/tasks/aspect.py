"""Aspect-based extractive summarization on ACLSum (human gold labels).

Three selections per paper (challenge / approach / outcome) plus a derived
per-document `union` row. K per aspect is the median gold count on a held-out
split; exemplars come from that same split.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .. import metrics as M
from ..config import Experiment
from ..data import aclsum
from ..fewshot.aspect import select_exemplar
from ..prompts import Exemplar
from .base import SelectionMap, Task


def _row(doc_id, doc_idx, aspect, n, pred, gold, m, rm, ro, oracle_idx, raw=None) -> Dict:
    gap = (ro["rougeL"] - rm["rougeL"]) if ro else 0.0
    return {
        "doc_id": doc_id, "doc_idx": doc_idx, "aspect": aspect, "num_sentences": n,
        "pred_indices": pred, "gold_indices": sorted(gold),
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "rouge1_model": rm.get("rouge1", 0.0), "rouge2_model": rm.get("rouge2", 0.0),
        "rougeL_model": rm.get("rougeL", 0.0),
        "rouge1_oracle": ro.get("rouge1", 0.0) if ro else 0.0,
        "rouge2_oracle": ro.get("rouge2", 0.0) if ro else 0.0,
        "rougeL_oracle": ro.get("rougeL", 0.0) if ro else 0.0,
        "oracle_gap_rougeL": gap, "oracle_indices": oracle_idx,
        # raw model text (str, or list[str] under self_consistency); None for the
        # derived union row, which makes no direct model call.
        "raw_response": raw,
    }


class AspectTask(Task):
    kind = "aspect"

    def __init__(self, experiment: Experiment):
        dataset = experiment.dataset
        self.units = tuple(dataset.get("aspects", aclsum.ASPECTS))
        self._eval_split = dataset.get("eval_split", "test")
        self._exemplar_split = dataset.get("exemplar_split", "validation")
        self._cap_split = experiment.cap.get("split", self._exemplar_split)
        self._eval = None
        self._pool = None
        self._refs: Dict[str, Dict[str, str]] = {}
        self._caps: Dict[str, int] = {}

    def load(self, needs_exemplars: bool) -> None:
        self._eval = aclsum.load_split(self._eval_split)
        self._pool = aclsum.load_split(self._exemplar_split)
        self._refs = aclsum.load_abstractive_refs()
        self._caps = aclsum.aspect_caps_from_gold(
            aclsum.load_split(self._cap_split), self.units
        )

    def __len__(self) -> int:
        return len(self._eval)

    def doc(self, index: int) -> Any:
        return self._eval[index]

    def doc_id(self, doc: Any) -> str:
        return doc["id"]

    def sentences(self, doc: Any) -> List[str]:
        return doc["source_sentences"]

    def cap(self, unit: str) -> int:
        return self._caps[unit]

    def exemplar(self, unit: str, seed: int) -> Optional[Exemplar]:
        return select_exemplar(self._pool, unit, seed=seed)

    def _scored(self, doc_id, sents, ref, pred, key, oracle_cache):
        """ROUGE of the selection and of the greedy oracle against `ref`."""
        rm = M.rouge(ref, aclsum.indices_to_text(sents, pred))
        if key not in oracle_cache:
            oracle_cache[key] = M.build_oracle_indices(sents, ref)
        oracle_idx = oracle_cache[key]
        ro = M.rouge(ref, aclsum.indices_to_text(sents, oracle_idx))
        return rm, ro, oracle_idx

    def rows(self, doc, doc_idx, preds: SelectionMap, raws, oracle_cache) -> List[Dict]:
        doc_id, sents = doc["id"], doc["source_sentences"]
        n = len(sents)
        gold = aclsum.gold_for_doc(doc, self.units)
        abs_ref = self._refs.get(doc_id)
        out: List[Dict] = []

        for a in self.units:
            m = M.prf(gold[a], preds[a])
            rm, ro, oracle_idx = {}, {}, []
            if abs_ref and abs_ref.get(a):
                rm, ro, oracle_idx = self._scored(
                    doc_id, sents, abs_ref[a], preds[a], (doc_id, a), oracle_cache
                )
            out.append(_row(doc_id, doc_idx, a, n, preds[a], gold[a], m, rm, ro,
                            oracle_idx, raw=raws[a]))

        # union across aspects
        gold_u = sorted(set().union(*[set(gold[a]) for a in self.units]))
        pred_u = sorted(set().union(*[set(preds[a]) for a in self.units]))
        m_u = M.prf(gold_u, pred_u)
        rm_u, ro_u, oracle_u = {}, {}, []
        if abs_ref:
            ref_u = M.concat_abs_refs(abs_ref, self.units)
            if ref_u:
                rm_u, ro_u, oracle_u = self._scored(
                    doc_id, sents, ref_u, pred_u, (doc_id, "union"), oracle_cache
                )
        out.append(_row(doc_id, doc_idx, "union", n, pred_u, gold_u, m_u, rm_u, ro_u, oracle_u))
        return out

    def meta(self) -> Dict[str, Any]:
        return {"caps": dict(self._caps)}
