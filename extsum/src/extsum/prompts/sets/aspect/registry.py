"""Technique registry for this prompt set (see extsum.prompts.registry)."""
from __future__ import annotations

from extsum.prompts.registry import Registry

_REGISTRY = Registry(__package__ + ".techniques")

register = _REGISTRY.register
all_techniques = _REGISTRY.all_techniques
all_variants = _REGISTRY.all_variants
