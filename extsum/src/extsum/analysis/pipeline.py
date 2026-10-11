"""Run every analysis for one experiment.

    reports/<experiment>/tables/    ablations, bootstrap CIs, significance,
                                    runtime, position bias (+ MoverScore
                                    correlation when moverscore is installed)
    reports/<experiment>/figures/   one position-bias plot per model

The analyses read the whole-document row schema, so they apply to
document-task experiments (cnndm, xsum, billsum).
"""
from __future__ import annotations

from ..config import load_experiment
from .ablations import build_ablation_tables
from .bootstrap import build_bootstrap_table
from .correlation import run_correlation_analysis
from .position_bias import build_position_table, plot_position_bias
from .runtime import build_runtime_table
from .significance import paired_strategy_tests


def run_all(experiment: str, model: str = "qwen35_4b") -> None:
    """`model` is the one used for the single-model tables (10, 11, 12)."""
    exp = load_experiment(experiment)
    if exp.task != "document":
        raise ValueError(
            f"The analysis suite covers document-task experiments; "
            f"{exp.name!r} is an {exp.task!r} task."
        )

    results = str(exp.results_dir)
    tables = str(exp.reports_dir / "tables")
    figures = str(exp.reports_dir / "figures")

    print("[analysis] tool and scoring ablations (Tables 10, 11)")
    build_ablation_tables(results_dir=results, output_dir=tables, dataset=exp.name, model=model)

    print("[analysis] metric correlation (Table 12)")
    try:
        run_correlation_analysis(
            results_dir=results,
            strategy_out=f"{tables}/metric_correlation.csv",
            spearman_out=f"{tables}/spearman_correlation.csv",
            table12_out=f"{tables}/table12_rank_invariance.csv",
            table12_dataset=exp.name,
            table12_model=model,
            table12_shot="zero_shot",
            table12_cap="capped",
        )
    except RuntimeError as exc:
        # MoverScore is an optional dependency; the rest does not need it.
        print(f"[analysis] skipped: {exc}")

    print("[analysis] bootstrap confidence intervals")
    build_bootstrap_table(
        results_dir=results, out_path=f"{tables}/bootstrap_confidence_intervals.csv"
    )

    print("[analysis] paired significance tests")
    paired_strategy_tests(results_dir=results, metric="f1", out_path=f"{tables}/significance_f1.csv")

    print("[analysis] runtime")
    build_runtime_table(results_dir=results, out_path=f"{tables}/runtime.csv")

    print("[analysis] position bias")
    positions = build_position_table(results_dir=results, out_path=f"{tables}/position_bias.csv")
    if not positions.empty:
        # The silver-reference curve is drawn inside each model's figure.
        models = positions.loc[positions["selection_source"] == "model", "model"].unique()
        for name in sorted(models):
            plot_position_bias(
                positions,
                dataset=exp.name,
                model=name,
                out_path=f"{figures}/position_bias_{exp.name}_{name}.pdf",
            )

    print(f"[analysis] complete -> {exp.reports_dir}")
