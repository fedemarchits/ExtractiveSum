"""Heuristics that turn an abstractive reference into extractive labels.

The study that produced `local_score` (Part B): greedy, beam, local and global
search over ROUGE, a transformer-scored variant, MMR over sentence embeddings
and MoverScore as the objective, compared on ACLSum for K = 1..32.

    extsum heuristics sweep             the full suite (sections are --skip-*-able)
    extsum heuristics hybrid            score-vs-MMR hybrid grid
    extsum heuristics moverscore-eval   MoverScore over saved ROUGE-sweep picks

Outputs go to reports/heuristics/. The GPU sections need the `heuristics`
extra (sentence-transformers, moverscore).

This package is self-contained: it keeps its own ROUGE/PRF helpers so the
published Part B numbers stay reproducible. The winning heuristic as applied to
new datasets lives in extsum.silver.
"""
