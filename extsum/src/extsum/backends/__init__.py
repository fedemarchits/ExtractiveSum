"""Generation backends behind one interface.

    backend.generate_batch(prompts, system=None) -> list[str]

    hf        Hugging Face Transformers on one GPU.
    vllm      vLLM with continuous batching (the fast path on a rented GPU).
    endpoint  OpenAI-compatible chat API (OPENAI_BASE_URL / OPENAI_API_KEY).

configs/models.yaml lists each model once. Where a model runs is decided by a
*profile* (``vast``, ``cluster``, ...): the profile supplies the backend and
its settings for every local model, a model may override them per profile, and
models that pin ``backend`` themselves (the endpoint-only large tier) ignore
the profile. Select with --profile or EXTSUM_PROFILE.

Heavy dependencies (torch, transformers, vllm, openai) are imported lazily, so
this package can be imported for offline configuration inspection.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import yaml

from ..paths import PathLike, resolve
from .base import GenConfig

DEFAULT_MODELS_YAML = "configs/models.yaml"
_RESERVED = ("alias", "hf_id", "family", "tier", "profiles")


def _load(models_yaml: PathLike) -> Dict:
    path = resolve(models_yaml)
    if not path.is_file():
        raise FileNotFoundError(f"Model configuration file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def active_profile(profile: Optional[str], config: Dict) -> str:
    return (
        profile
        or os.environ.get("EXTSUM_PROFILE")
        or config.get("default_profile")
        or "vast"
    )


def model_aliases(models_yaml: PathLike = DEFAULT_MODELS_YAML) -> List[str]:
    return [m["alias"] for m in _load(models_yaml).get("models", [])]


def resolve_model(
    alias: str,
    models_yaml: PathLike = DEFAULT_MODELS_YAML,
    profile: Optional[str] = None,
) -> Dict:
    """Merge profile defaults, per-model profile overrides and fixed settings."""
    config = _load(models_yaml)
    models = {m["alias"]: m for m in config.get("models", [])}
    if alias not in models:
        raise KeyError(
            f"Model alias {alias!r} was not found in {models_yaml}. "
            f"Available aliases: {sorted(models)}"
        )
    model = models[alias]
    fixed = {k: v for k, v in model.items() if k not in _RESERVED}

    settings: Dict = {}
    if "backend" not in fixed:
        name = active_profile(profile, config)
        profiles = config.get("profiles", {})
        if name not in profiles:
            raise KeyError(
                f"Unknown profile {name!r} in {models_yaml}. "
                f"Available: {sorted(profiles)}"
            )
        settings.update(profiles[name])
        settings.update((model.get("profiles") or {}).get(name, {}))
        settings["profile"] = name
    settings.update(fixed)
    settings.update(alias=alias, hf_id=model["hf_id"])
    return settings


def get_backend(
    alias: str,
    models_yaml: PathLike = DEFAULT_MODELS_YAML,
    gen: Optional[GenConfig] = None,
    profile: Optional[str] = None,
):
    """Construct the generation backend configured for one model alias."""
    m = resolve_model(alias, models_yaml, profile)
    gen = gen or GenConfig()
    kind = m.get("backend", "hf")

    if kind in ("hf", "local"):
        from .hf import LocalHFBackend
        return LocalHFBackend(
            hf_id=m["hf_id"],
            gen=gen,
            device_map=m.get("device_map"),
            load_in_4bit=bool(m.get("load_in_4bit", False)),
        )

    if kind == "vllm":
        from .vllm import VLLMBackend
        return VLLMBackend(
            hf_id=m["hf_id"],
            gen=gen,
            quantization=m.get("quantization"),
            tensor_parallel_size=int(m.get("tensor_parallel_size", 1)),
            max_model_len=m.get("max_model_len"),
            gpu_memory_utilization=float(m.get("gpu_memory_utilization", 0.90)),
            disable_reasoning=bool(m.get("disable_reasoning", True)),
        )

    if kind == "endpoint":
        from .endpoint import EndpointBackend
        return EndpointBackend(
            model_id=m.get("endpoint_model", m["hf_id"]),
            gen=gen,
            disable_reasoning=bool(m.get("disable_reasoning", True)),
        )

    raise ValueError(f"Unknown backend {kind!r} for model alias {alias!r}.")


__all__ = ["GenConfig", "get_backend", "resolve_model", "model_aliases"]
