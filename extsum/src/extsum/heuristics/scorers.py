"""Unified scorer registry. Every scorer is callable: f(candidate, reference) -> float.

String identifiers used here flow through joblib without pickle issues
(the actual closures are recreated inside each worker via get_scorer).
"""

from .metrics import rouge_score


def _make_rouge_scorer(rouge_type):
    def _fn(candidate, reference):
        return rouge_score(candidate, reference, rouge_type)
    _fn.__name__ = f"rouge_{rouge_type}_scorer"
    return _fn


def _make_mover_scorer():
    def _fn(candidate, reference):
        from .metrics_semantic import mover_score
        return mover_score(candidate, reference)
    _fn.__name__ = "mover_scorer"
    return _fn


_REGISTRY = {
    "rouge1": _make_rouge_scorer("rouge1"),
    "rouge2": _make_rouge_scorer("rouge2"),
    "rougeL": _make_rouge_scorer("rougeL"),
    "moverscore": _make_mover_scorer(),
}


def get_scorer(name):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown scorer '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_scorers():
    return list(_REGISTRY)
