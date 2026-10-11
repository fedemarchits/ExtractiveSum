"""ROUGE metric wrappers — direct google-research rouge_score (no HF evaluate hot path).

For single-pair calls these return scores numerically equivalent to
HF `evaluate.load("rouge")` (HF wraps the same google rouge_score library
underneath; bootstrap aggregation is degenerate on a single sample), but
without per-call wrapper overhead → 5-10× faster.
"""

_SCORERS = {}


def _get_scorer(rouge_type):
    """Lazy per-process RougeScorer singleton. Keyed by rouge_type for caching."""
    if rouge_type not in _SCORERS:
        from rouge_score import rouge_scorer
        _SCORERS[rouge_type] = rouge_scorer.RougeScorer(
            [rouge_type], use_stemmer=True
        )
    return _SCORERS[rouge_type]


def rouge_score(candidate, reference, rougex):
    """Single F1 score. Drop-in replacement for the old HF-evaluate wrapper."""
    scorer = _get_scorer(rougex)
    return scorer.score(reference, candidate)[rougex].fmeasure


def rouge_pr_f(candidate, reference, rougex):
    """Return {'precision': p, 'recall': r, 'fmeasure': f} for one pair."""
    scorer = _get_scorer(rougex)
    s = scorer.score(reference, candidate)[rougex]
    return {"precision": s.precision, "recall": s.recall, "fmeasure": s.fmeasure}


def rouge_score_batch(candidates, reference, rougex):
    """Score many candidates against one reference. Same per-pair tokenization
    cost (RougeScorer doesn't expose pre-tokenized refs cleanly), but skips
    Python-level wrapper overhead via a single function-local scorer reference.
    """
    scorer = _get_scorer(rougex)
    return [scorer.score(reference, c)[rougex].fmeasure for c in candidates]
