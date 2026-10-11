"""What a task is.

A task turns a dataset into *units of work*: for each evaluation document, one
or more named selections the model has to make (ACLSum: one per aspect; a
whole-document dataset: a single "summary"). It owns everything dataset-shaped
— loading, labels, caps, exemplars and the result-row schema — so the runner
stays the same for every dataset.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..prompts import Exemplar

SelectionMap = Dict[str, List[int]]


class Task:
    kind: str = ""
    #: the named selections made per document, in prompt order
    units: Sequence[str] = ()

    def load(self, needs_exemplars: bool) -> None:
        """Load evaluation documents, caps and (if needed) the exemplar pool."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def doc(self, index: int) -> Any:
        raise NotImplementedError

    def doc_id(self, doc: Any) -> str:
        raise NotImplementedError

    def sentences(self, doc: Any) -> List[str]:
        raise NotImplementedError

    def cap(self, unit: str) -> int:
        """Sentence budget K applied to capped variants of this unit."""
        raise NotImplementedError

    def exemplar(self, unit: str, seed: int) -> Optional[Exemplar]:
        """The one-shot exemplar for this unit (never an evaluation document)."""
        raise NotImplementedError

    def rows(
        self,
        doc: Any,
        doc_idx: int,
        preds: SelectionMap,
        raws: Dict[str, Any],
        oracle_cache: Dict,
    ) -> List[Dict]:
        """Score one document's selections and return its JSONL rows."""
        raise NotImplementedError

    def meta(self) -> Dict[str, Any]:
        """Task-specific fields for the per-variant .meta.json sidecar."""
        return {}
