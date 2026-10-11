"""Metrics: exact-match P/R/F1 on sentence indices, ROUGE of the extracted
text against the abstractive reference, and the greedy ROUGE-L oracle used as
an upper bound.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def prf(gold_indices: Sequence[int], pred_indices: Sequence[int]) -> Dict[str, float]:
    """Precision, recall, F1 (+ tp/fp/fn) for two index sets."""
    gs = {int(i) for i in gold_indices}
    ps = {int(i) for i in pred_indices}
    tp, fp, fn = len(gs & ps), len(ps - gs), len(gs - ps)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


# Scorers are created lazily so importing this module needs no heavy deps.
_full = None
_oracle = None


def _full_scorer():
    global _full
    if _full is None:
        from rouge_score import rouge_scorer
        _full = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return _full


def _oracle_scorer():
    global _oracle
    if _oracle is None:
        from rouge_score import rouge_scorer
        _oracle = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _oracle


def rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """ROUGE-1/2/L F1 for a reference/hypothesis pair (zeros if either is empty)."""
    reference = str(reference or "").strip()
    hypothesis = str(hypothesis or "").strip()
    if not reference or not hypothesis:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    rs = _full_scorer().score(reference, hypothesis)
    return {k: rs[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")}


def build_oracle_indices(
    sentences: Sequence[str], abstr_ref: str, max_len: Optional[int] = None
) -> List[int]:
    """Greedy oracle: add the sentence that most improves ROUGE-L F1 until no
    gain (or max_len). Returns 1-based indices in selection order."""
    abstr_ref = str(abstr_ref or "").strip()
    if not abstr_ref or not sentences:
        return []

    scorer = _oracle_scorer()
    selected: List[int] = []
    remaining = list(range(len(sentences)))
    best_f = 0.0
    while remaining and (max_len is None or len(selected) < max_len):
        best_f_next, chosen = best_f, None
        for i in remaining:
            hyp = " ".join(sentences[j] for j in selected + [i])
            f = scorer.score(abstr_ref, hyp)["rougeL"].fmeasure
            if f > best_f_next:
                best_f_next, chosen = f, i
        if chosen is None:
            break
        best_f = best_f_next
        selected.append(chosen)
        remaining.remove(chosen)
    return [i + 1 for i in selected]


def concat_abs_refs(abs_ref: Dict[str, str], aspects: Sequence[str]) -> str:
    """Concatenate abstractive references across aspects (for the union setting)."""
    if not abs_ref:
        return ""
    return " ".join(abs_ref[a] for a in aspects if abs_ref.get(a))
