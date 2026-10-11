"""MoverScore semantic similarity wrapper.

Uses moverscore_v2 (install via:
  pip install git+https://github.com/AIPHES/emnlp19-moverscore.git
). Default backbone is `distilbert-base-uncased` per env var
MOVERSCORE_MODEL inside moverscore_v2.
"""

from collections import defaultdict

import numpy as np

# moverscore_v2 still uses the deprecated `np.float` alias (removed in numpy>=1.24).
# Restore it so the library doesn't crash inside `word_mover_score`.
if not hasattr(np, "float"):
    np.float = float


_IDF_DICT = None


def _get_idf_dict():
    global _IDF_DICT
    if _IDF_DICT is None:
        _IDF_DICT = defaultdict(lambda: 1.0)
    return _IDF_DICT


def mover_score(candidate, reference):
    """Single MoverScore (uniform IDF, n-gram=1)."""
    from moverscore_v2 import word_mover_score
    idf = _get_idf_dict()
    scores = word_mover_score(
        [reference],
        [candidate],
        idf_dict_ref=idf,
        idf_dict_hyp=idf,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
        batch_size=8,
    )
    return float(scores[0])


def mover_score_batch(candidates, references):
    """Batched MoverScore — much faster than per-pair calls."""
    from moverscore_v2 import word_mover_score
    idf = _get_idf_dict()
    scores = word_mover_score(
        list(references),
        list(candidates),
        idf_dict_ref=idf,
        idf_dict_hyp=idf,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
        batch_size=32,
    )
    return [float(s) for s in scores]
