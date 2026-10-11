"""Technique registry.

Each prompt set owns one Registry. Every file under the set's techniques/
package calls ``register`` on its Technique; the first lookup imports them all.
Lookups are by technique name; variant expansion lives in base.expand.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List

from .base import Technique, Variant, expand


class Registry:
    def __init__(self, techniques_package: str):
        self._techniques_package = techniques_package
        self._techniques: Dict[str, Technique] = {}

    def register(self, tech: Technique) -> Technique:
        if tech.name in self._techniques:
            raise ValueError(f"duplicate technique: {tech.name}")
        self._techniques[tech.name] = tech
        return tech

    def _discover(self) -> None:
        package = importlib.import_module(self._techniques_package)
        for mod in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{package.__name__}.{mod.name}")

    def all_techniques(self) -> Dict[str, Technique]:
        if not self._techniques:
            self._discover()
        return dict(self._techniques)

    def all_variants(self) -> List[Variant]:
        out: List[Variant] = []
        for t in self.all_techniques().values():
            out.extend(expand(t))
        return out
