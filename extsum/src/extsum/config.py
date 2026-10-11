"""Experiment configuration.

One YAML per experiment under configs/experiments/. It names the task (what a
unit of work is), the prompt set (how it is worded) and the data, so the same
runner, grid and analysis serve every dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from .paths import named_config, resolve

TASKS = ("aspect", "document")


@dataclass(frozen=True)
class Experiment:
    name: str
    task: str
    prompt_set: str
    grid: str
    dataset: Dict[str, Any]
    cap: Dict[str, Any]
    fewshot: Dict[str, Any] = field(default_factory=dict)
    rationale: Dict[str, Any] = field(default_factory=dict)
    inference: Dict[str, Any] = field(default_factory=dict)
    self_consistency: Dict[str, Any] = field(default_factory=dict)
    dynamic_capper: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    path: Path = Path()

    @property
    def seed(self) -> int:
        return int(self.fewshot.get("seed", 42))

    @property
    def results_dir(self) -> Path:
        """results/<experiment>/ — one sub-directory per model below it."""
        return resolve(self.output.get("results_dir", "results")) / self.name

    @property
    def reports_dir(self) -> Path:
        """reports/<experiment>/ — summary.csv, tables/ and figures/."""
        return resolve(self.output.get("reports_dir", "reports")) / self.name

    @property
    def rationale_dir(self) -> Path:
        return resolve(
            self.rationale.get("cache_dir", f"data/rationales/{self.name}")
        )

    @property
    def resume(self) -> bool:
        return bool(self.output.get("resume", True))


def load_experiment(name_or_path: str) -> Experiment:
    path = named_config("experiments", name_or_path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    for key in ("name", "task", "prompt_set", "grid", "dataset", "cap"):
        if key not in raw:
            raise KeyError(f"'{key}' is missing from {path}")
    if raw["task"] not in TASKS:
        raise ValueError(
            f"task must be one of {TASKS}, got {raw['task']!r} in {path}"
        )

    known = set(Experiment.__dataclass_fields__) - {"path"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise KeyError(f"Unknown keys in {path}: {unknown}")

    return Experiment(path=path, **raw)
