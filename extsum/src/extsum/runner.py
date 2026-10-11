"""Drive the grid: model x prompt variant x document -> per-document JSONL.

    results/<experiment>/<model>/<variant>.jsonl        one row per selection
    results/<experiment>/<model>/<variant>.meta.json    latency, caps, counts

Resume-safe: a variant whose JSONL exists is skipped, and a variant is written
to a .partial file and renamed only when complete, so an interrupted run never
leaves a complete-looking file behind.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .backends import DEFAULT_MODELS_YAML, GenConfig, get_backend
from .config import Experiment, load_experiment
from .grid import resolve_variants
from .paths import named_config
from .postprocess import cap_indices_in_order
from .prompts import Cap, PromptSet, RenderCtx, Shot, Variant, load_prompt_set
from .prompts.rationale import configure_rationale_cache
from .tasks import Task, get_task
from .wrappers import parse_batch, select_document

SYSTEM = "You are an expert in extractive summarization."


def _enabled(wrapper_cfg: Optional[Dict]) -> bool:
    return bool(wrapper_cfg and wrapper_cfg.get("enabled"))


def render_contexts(variant: Variant, task: Task, seed: int) -> Dict[str, RenderCtx]:
    """One RenderCtx per unit: its cap (capped variants) and exemplar (one-shot)."""
    return {
        unit: RenderCtx(
            shot=variant.shot,
            cap=variant.cap,
            k=task.cap(unit) if variant.cap is Cap.CAPPED else None,
            exemplar=task.exemplar(unit, seed) if variant.shot is Shot.ONE else None,
        )
        for unit in task.units
    }


def run_variant(
    variant: Variant,
    prompt_set: PromptSet,
    task: Task,
    backend,
    seed: int,
    oracle_cache: Dict,
    out_path: Path,
    sc: Optional[Dict] = None,
    dyn: Optional[Dict] = None,
    limit: Optional[int] = None,
) -> None:
    """Run one prompt variant over the task's evaluation documents."""
    technique = prompt_set.all_techniques()[variant.technique]
    system = SYSTEM if technique.system_override is None else technique.system_override
    system = system or None
    units = list(task.units)
    ctxs = render_contexts(variant, task, seed)
    n_docs = len(task) if limit is None else min(limit, len(task))

    def prompts_for(doc) -> List[str]:
        sentences = task.sentences(doc)
        return [technique.build(sentences, u, ctxs[u]) for u in units]

    latencies: List[float] = []

    # Fast path: vLLM-style backends want EVERY prompt of the variant at once so
    # continuous batching kicks in, instead of one call per document. Only when
    # no per-document wrapper is active (each needs its own extra passes).
    precomputed = None
    if getattr(backend, "wants_full_batch", False) and not _enabled(sc) and not _enabled(dyn):
        flat: List[str] = []
        for doc_idx in range(n_docs):
            flat.extend(prompts_for(task.doc(doc_idx)))
        start = time.perf_counter()
        flat_outs = backend.generate_batch(flat, system=system)
        elapsed = time.perf_counter() - start
        if len(flat_outs) != len(flat):
            raise RuntimeError(
                "Backend returned an unexpected number of outputs: "
                f"expected {len(flat)}, got {len(flat_outs)}."
            )
        nu = len(units)
        precomputed = [flat_outs[i * nu:(i + 1) * nu] for i in range(n_docs)]
        latencies = [elapsed / max(n_docs, 1)] * n_docs  # mean per document

    partial_path = out_path.with_suffix(".jsonl.partial")
    with partial_path.open("w", encoding="utf-8") as fh:
        for doc_idx in range(n_docs):
            doc = task.doc(doc_idx)
            sentences = task.sentences(doc)

            if precomputed is not None:
                outs = precomputed[doc_idx]
                preds = parse_batch(outs, units, len(sentences))
                raws = {u: outs[j] for j, u in enumerate(units)}
            else:
                start = time.perf_counter()
                preds, raws = select_document(
                    backend, prompts_for(doc), units, sentences, sc, dyn, system,
                    capper_prompt=prompt_set.dynamic_capper_prompt,
                )
                latencies.append(time.perf_counter() - start)

            # The dynamic capper replaces the fixed cap; otherwise capped
            # variants keep the first K selected sentences in document order.
            if variant.cap is Cap.CAPPED and not _enabled(dyn):
                for u in units:
                    preds[u] = cap_indices_in_order(preds[u], task.cap(u))

            for row in task.rows(doc, doc_idx, preds, raws, oracle_cache):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    partial_path.replace(out_path)

    meta = {
        "variant": variant.slug,
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "n_docs": n_docs,
    }
    meta.update(task.meta())
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def generation_config(experiment: Experiment) -> GenConfig:
    """Greedy and deterministic, unless self-consistency asks for sampling."""
    inf, sc = experiment.inference, experiment.self_consistency
    temperature = (
        float(sc.get("temperature", 0.7)) if _enabled(sc)
        else float(inf.get("temperature", 0.0))
    )
    return GenConfig(
        max_new_tokens=int(inf.get("max_new_tokens", 512)),
        temperature=temperature,
        seed=int(inf.get("seed", experiment.seed)),
    )


def run(
    model_alias: str,
    experiment: str,
    grid: Optional[str] = None,
    models_yaml: str = DEFAULT_MODELS_YAML,
    profile: Optional[str] = None,
    limit: Optional[int] = None,
    backend=None,
) -> Path:
    """Run every variant of the grid for one model on one experiment.

    `grid` overrides the experiment's default grid; `limit` runs only the first
    N evaluation documents (smoke tests); `backend` injects a ready backend.
    Returns the directory the results were written to.
    """
    exp = load_experiment(experiment)
    prompt_set = load_prompt_set(exp.prompt_set)
    configure_rationale_cache(exp.rationale_dir)

    variants = resolve_variants(named_config("grids", grid or exp.grid), prompt_set)
    task = get_task(exp)
    task.load(needs_exemplars=any(v.shot is Shot.ONE for v in variants))

    if backend is None:
        backend = get_backend(model_alias, models_yaml, generation_config(exp), profile)

    out_dir = exp.results_dir / model_alias
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_cache: Dict = {}

    for variant in variants:
        out_path = out_dir / f"{variant.slug}.jsonl"
        label = f"{exp.name}/{model_alias}/{variant.slug}"
        if exp.resume and out_path.exists():
            print(f"[skip] {label} (exists)", flush=True)
            continue
        print(f"[run]  {label}", flush=True)
        run_variant(
            variant, prompt_set, task, backend, exp.seed, oracle_cache, out_path,
            sc=exp.self_consistency, dyn=exp.dynamic_capper, limit=limit,
        )
        backend.free_memory()  # release cached VRAM between variants

    return out_dir
