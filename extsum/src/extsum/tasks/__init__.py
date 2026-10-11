"""Tasks: the dataset-shaped half of an experiment (see tasks.base.Task)."""
from __future__ import annotations

from ..config import Experiment
from .base import Task


def get_task(experiment: Experiment) -> Task:
    if experiment.task == "aspect":
        from .aspect import AspectTask
        return AspectTask(experiment)
    if experiment.task == "document":
        from .document import DocumentTask
        return DocumentTask(experiment)
    raise ValueError(f"Unknown task {experiment.task!r}")


__all__ = ["Task", "get_task"]
