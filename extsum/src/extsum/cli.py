"""Command-line interface.

    extsum list        show experiments, grids, models, or a grid's variants
    extsum preflight   check that a run has everything it needs
    extsum run         run a grid for one model on one experiment
    extsum silver      build an experiment's silver label files
    extsum compute-k   derive the sentence budget K from the compression ratio
    extsum rationales  build the cached one-shot reasoning traces
    extsum rescore     re-parse saved raw responses with the current parser
    extsum report      aggregate raw results into reports/<experiment>/summary.csv
    extsum analyze     tables and figures for one experiment
    extsum figures     overview figures for the README
    extsum heuristics  the heuristic study that selected `local_score`

Experiments and grids are referred to by name (configs/experiments/<name>.yaml,
configs/grids/<name>.yaml) or by path.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _add_run_target(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-e", "--experiment", required=True,
                        help="experiment name or YAML path (see `extsum list`)")
    parser.add_argument("-m", "--model", required=True,
                        help="model alias from configs/models.yaml")
    parser.add_argument("-g", "--grid", default=None,
                        help="grid name or YAML path (default: the experiment's grid)")
    parser.add_argument("--profile", default=None,
                        help="machine profile from configs/models.yaml "
                             "(default: EXTSUM_PROFILE, else the file's default_profile)")
    parser.add_argument("--models", default="configs/models.yaml",
                        help="model configuration YAML")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extsum",
        description="Extractive summarization with LLM prompting: runs, data and analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="show what is configured")
    p.add_argument("what", nargs="?", default="experiments",
                   choices=["experiments", "grids", "models", "variants"])
    p.add_argument("-e", "--experiment", help="for `variants`: the experiment")
    p.add_argument("-g", "--grid", help="for `variants`: a grid other than the experiment's")
    p.add_argument("--profile", default=None)

    p = sub.add_parser("preflight", help="check that a run has everything it needs")
    _add_run_target(p)

    p = sub.add_parser("run", help="run a grid for one model on one experiment")
    _add_run_target(p)
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N evaluation documents (smoke test)")
    p.add_argument("--skip-preflight", action="store_true")

    p = sub.add_parser("silver", help="build an experiment's silver label files")
    p.add_argument("-e", "--experiment", required=True)
    p.add_argument("--which", choices=["eval", "exemplar", "both"], default="both")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("compute-k", help="derive K from the compression ratio")
    p.add_argument("-e", "--experiment", required=True)
    p.add_argument("--split", default=None,
                   help="development split (default: the experiment's exemplar split)")
    p.add_argument("--max-docs", type=int, default=None,
                   help="subsample size (default: the whole split)")
    p.add_argument("--seed", type=int, default=None, help="seed for the subsample")
    p.add_argument("--dry-run", action="store_true",
                   help="print the statistics without writing the heuristic JSON")

    p = sub.add_parser("rationales", help="build the cached one-shot reasoning traces")
    p.add_argument("-e", "--experiment", required=True)
    p.add_argument("-m", "--model", default=None,
                   help="reference model alias (default: rationale.source_model)")
    p.add_argument("--models", default="configs/models.yaml")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("rescore", help="re-parse saved raw responses with the current parser")
    p.add_argument("-e", "--experiment", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="report the F1 change without writing anything")

    p = sub.add_parser("report", help="aggregate raw results into summary.csv")
    p.add_argument("-e", "--experiment", required=True, nargs="+")

    p = sub.add_parser("analyze", help="tables and figures for one experiment")
    p.add_argument("-e", "--experiment", required=True)
    p.add_argument("-m", "--model", default="qwen35_4b",
                   help="model used for the single-model tables")

    p = sub.add_parser("figures", help="overview figures for the README")
    p.add_argument("--out", default="reports/figures")

    p = sub.add_parser(
        "heuristics", help="the heuristic study that selected local_score",
        description="Arguments after the sub-command are passed through; add --help to see them.",
    )
    p.add_argument("study", choices=["sweep", "hybrid", "moverscore-eval"])
    p.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def _cmd_list(args) -> int:
    from .paths import named_config, resolve

    if args.what in ("experiments", "grids"):
        for path in sorted(resolve(f"configs/{args.what}").glob("*.yaml")):
            if args.what == "experiments":
                from .config import load_experiment
                exp = load_experiment(path.stem)
                print(f"{exp.name:<10} task={exp.task:<9} prompts={exp.prompt_set:<9} grid={exp.grid}")
            else:
                print(path.stem)
        return 0

    if args.what == "models":
        from .backends import model_aliases, resolve_model
        for alias in model_aliases():
            m = resolve_model(alias, profile=args.profile)
            where = m.get("endpoint_model") or m["hf_id"]
            print(f"{alias:<12} backend={m['backend']:<9} {where}")
        return 0

    if not args.experiment:
        print("extsum list variants needs --experiment", file=sys.stderr)
        return 2
    from .config import load_experiment
    from .grid import resolve_variants
    from .prompts import load_prompt_set

    exp = load_experiment(args.experiment)
    variants = resolve_variants(
        named_config("grids", args.grid or exp.grid), load_prompt_set(exp.prompt_set)
    )
    for variant in sorted(variants, key=lambda v: v.slug):
        print(variant.slug)
    print(f"\n{len(variants)} variants across "
          f"{len({v.technique for v in variants})} techniques")
    return 0


def _preflight(args) -> bool:
    from . import preflight
    errors, warnings = preflight.check(
        args.experiment, args.model, args.grid, args.models, args.profile
    )
    return preflight.report(errors, warnings, f"{args.model} on {args.experiment}")


def _cmd_heuristics(args) -> int:
    from .paths import resolve

    module = {
        "sweep": "sweep", "hybrid": "hybrid_sweep", "moverscore-eval": "moverscore_eval",
    }[args.study]
    passthrough = [a for a in args.args if a != "--"]
    if args.study != "moverscore-eval" and "--out-dir" not in passthrough:
        default = "reports/heuristics" + ("/hybrid" if args.study == "hybrid" else "")
        passthrough += ["--out-dir", str(resolve(default))]

    import importlib
    main = importlib.import_module(f"extsum.heuristics.{module}").main
    sys.argv = [f"extsum heuristics {args.study}"] + passthrough
    main()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    from .env import load_dotenv
    load_dotenv()

    if args.command == "list":
        return _cmd_list(args)

    if args.command == "preflight":
        return 0 if _preflight(args) else 1

    if args.command == "run":
        if not args.skip_preflight and not _preflight(args):
            return 1
        from .runner import run
        out_dir = run(args.model, args.experiment, args.grid, args.models,
                      args.profile, args.limit)
        print(f"[run] done -> {out_dir}")
        return 0

    if args.command == "silver":
        from .silver.build import build_silver
        build_silver(args.experiment, args.which, args.overwrite)
        return 0

    if args.command == "compute-k":
        from .silver.build import compute_k
        compute_k(args.experiment, args.split, args.max_docs, args.seed, args.dry_run)
        return 0

    if args.command == "rationales":
        from .rationales.builder import build
        build(args.experiment, args.model, args.models, args.overwrite)
        return 0

    if args.command == "rescore":
        from .rescore import rescore
        rescore(args.experiment, args.dry_run)
        return 0

    if args.command == "report":
        from .report import write_summary
        for experiment in args.experiment:
            write_summary(experiment)
        return 0

    if args.command == "analyze":
        from .analysis.pipeline import run_all
        run_all(args.experiment, args.model)
        return 0

    if args.command == "figures":
        from .analysis.overview import build_figures
        build_figures(args.out)
        return 0

    if args.command == "heuristics":
        return _cmd_heuristics(args)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
