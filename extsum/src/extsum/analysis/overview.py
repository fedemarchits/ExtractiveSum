"""Overview figures for the project README.

    extsum figures            # -> reports/figures/*.png

Each figure is drawn from the tracked report files (reports/<experiment>/
summary.csv and tables/), so it can be rebuilt without the raw answers. The one
exception is the selection-length table, which needs the raw JSONL: it is
written to reports/<experiment>/tables/selection_length.csv when the raw files
are present and read from there otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from ..paths import resolve

QWEN, GEMMA = "#2F6DB5", "#E0792B"
INK, MUTED, GRID = "#1F2933", "#6B7785", "#E3E7EC"

#: alias -> (family, label, nominal parameters in billions)
MODELS = {
    "qwen35_2b": ("Qwen3.5", "2B", 2),
    "qwen35_4b": ("Qwen3.5", "4B", 4),
    "qwen35_9b": ("Qwen3.5", "9B", 9),
    "qwen35_27b": ("Qwen3.5", "27B", 27),
    "gemma4_e2b": ("Gemma4", "E2B", 2),
    "gemma4_e4b": ("Gemma4", "E4B", 4),
    "gemma4_12b": ("Gemma4", "12B", 12),
    "gemma4_31b": ("Gemma4", "31B", 31),
}
FAMILY_COLOR = {"Qwen3.5": QWEN, "Gemma4": GEMMA}
LOCAL = [m for m in MODELS if m not in ("qwen35_27b", "gemma4_31b")]

#: the five techniques the large tier ran on ACLSum
ACLSUM_FIVE = ["self_ask", "salience_inference", "self_critique", "contrastive_joint", "negative_aware"]

#: techniques present in every prompt set (no _trace, no ablations)
BASE_TECHNIQUES = [
    "vanilla", "chain_of_thought", "least_to_most", "self_ask", "explanation_based",
    "salience_inference", "scoring_based", "simulated_tool_augmented", "self_critique",
    "contrastive_joint", "negative_aware", "tool_augmented",
]


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK, "text.color": INK,
        "axes.edgecolor": "#B8C0CA", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False,
    })


def _summary(experiment: str):
    import pandas as pd
    return pd.read_csv(resolve(f"reports/{experiment}/summary.csv"))


def _label(alias: str) -> str:
    family, size, _ = MODELS[alias]
    return f"{family}-{size}"


def baselines(experiment: str) -> Dict[str, float]:
    """F1 of two no-model selections: the first K sentences, and every sentence."""
    import numpy as np
    import pandas as pd

    from ..config import load_experiment
    from ..metrics import prf
    from ..tasks.document import load_cap_k

    exp = load_experiment(experiment)
    out_path = exp.reports_dir / "tables" / "baselines.json"
    silver_path = resolve(exp.dataset["eval_silver"])
    if silver_path.is_file():
        k = load_cap_k(exp.cap)
        silver = pd.read_parquet(silver_path)
        first, everything = [], []
        for row in silver.itertuples():
            n = len(row.source_sentences)
            first.append(prf(row.silver_indices, range(1, min(k, n) + 1))["f1"])
            everything.append(prf(row.silver_indices, range(1, n + 1))["f1"])
        values = {"first_k": float(np.mean(first)), "select_all": float(np.mean(everything)), "k": k}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    return json.loads(out_path.read_text(encoding="utf-8"))


def selection_length(experiment: str):
    """Mean number of selected sentences per model, capped vs uncapped."""
    import pandas as pd

    from ..config import load_experiment

    exp = load_experiment(experiment)
    out_path = exp.reports_dir / "tables" / "selection_length.csv"
    files = sorted(exp.results_dir.glob("*/*.jsonl"))
    if files:
        rows: List[Dict] = []
        for path in files:
            lengths = [
                len(json.loads(line)["pred_indices"])
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            rows.append({
                "model": path.parent.name, "variant": path.stem,
                "cap": path.stem.split("__")[2], "n": len(lengths),
                "mean_selected": sum(lengths) / max(len(lengths), 1),
            })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
    return pd.read_csv(out_path)


def fig_scaling(out: Path) -> None:
    import matplotlib.pyplot as plt

    acl = _summary("aclsum")
    acl = acl[acl.technique.isin(ACLSUM_FIVE)].groupby("model").f1.mean()

    def shared_variants(experiment: str):
        """Mean F1 per model over the variants the large tier also ran."""
        df = _summary(experiment)
        shared = set(zip(*[df[df.model == "gemma4_31b"][c] for c in ("technique", "shot", "cap")]))
        keep = [t in shared for t in zip(df.technique, df.shot, df.cap)]
        return df[keep].groupby("model").f1.mean(), len(shared)

    cnn, n_cnn = shared_variants("cnndm")
    bill, n_bill = shared_variants("billsum")

    panels = [
        ("ACLSum · aspect-based, human labels", "5 best techniques (18 variants), all 8 models", acl, None),
        ("CNN/DailyMail · whole document, silver", f"5 best techniques ({n_cnn} variants), all 8 models", cnn, baselines("cnndm")),
        ("BillSum · whole bill, silver", f"5 best techniques ({n_bill} variants), all 8 models", bill, baselines("billsum")),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, (title, subtitle, series, base) in zip(axes, panels):
        by_family = {
            family: sorted(
                (MODELS[m][2], series[m], MODELS[m][1]) for m in series.index if MODELS[m][0] == family
            )
            for family in FAMILY_COLOR
        }
        for family, color in FAMILY_COLOR.items():
            points = by_family[family]
            other = by_family["Qwen3.5" if family == "Gemma4" else "Gemma4"]
            xs, ys, names = zip(*points)
            ax.plot(xs, ys, "-o", color=color, lw=2.2, ms=7, label=family, zorder=3)
            for x, y, name in points:
                # Label above the point if it is the higher of the two families
                # at (roughly) this size, so labels never sit on each other.
                near = [oy for ox, oy, _ in other if max(x, ox) / min(x, ox) <= 1.5]
                above = (y > near[0] or (y == near[0] and family == "Gemma4")) if near else family == "Gemma4"
                ax.annotate(
                    f"{name}\n{y:.3f}" if above else f"{y:.3f}\n{name}", (x, y),
                    textcoords="offset points", xytext=(0, 9 if above else -9),
                    ha="center", va="bottom" if above else "top", fontsize=8.5, color=color,
                )
        if base:
            for key in ("first_k", "select_all"):
                ax.axhline(base[key], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(0.03, 0.03,
                    f"dashed lines, no model:\nfirst {base['k']} sentences {base['first_k']:.3f}\n"
                    f"every sentence {base['select_all']:.3f}",
                    transform=ax.transAxes, color=MUTED, fontsize=8.5, va="bottom", ha="left",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
        ax.set_xscale("log")
        ax.set_xticks([2, 4, 9, 12, 29])
        ax.set_xticklabels(["2", "4", "9", "12", "27 / 31"])
        ax.minorticks_off()
        ax.set_xlim(1.6, 40)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.035, hi + 0.035)
        ax.set_xlabel("model size (billion parameters, log scale)")
        ax.set_title(f"{title}\n", pad=4)
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    axes[0].set_ylabel("mean F1 on selected sentences")
    axes[0].legend(loc="lower right")
    fig.suptitle("Past about 4B parameters, bigger models add little; on CNN/DailyMail, nothing",
                 x=0.012, ha="left", fontsize=14, fontweight="bold", y=1.04)
    fig.savefig(out)
    plt.close(fig)


def fig_tfidf(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    df = _summary("billsum")
    df = df[(df.shot == "zero_shot") & (df.cap == "capped")]
    bars = [
        ("vanilla", "plain prompt", "#B8C0CA"),
        ("tool_no_meta", "tool prompt, scores removed", "#8FA3B8"),
        ("tool_augmented", "tool prompt with TF-IDF scores", "#2A9D8F"),
        ("tool_no_roleplay", "TF-IDF scores, no role-play", "#16695F"),
    ]
    table = df.pivot_table(index="model", columns="technique", values="f1").loc[LOCAL]
    base = baselines("billsum")

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    width = 0.2
    x = np.arange(len(LOCAL))
    for i, (technique, label, color) in enumerate(bars):
        values = table[technique].values
        rects = ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color, zorder=3)
        for rect, value in zip(rects, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + 0.006, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(base["first_k"], color=INK, lw=1.1, ls=(0, (4, 3)), zorder=4,
               label=f"no model: first {base['k']} sentences ({base['first_k']:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in LOCAL])
    ax.set_ylabel("F1 (zero-shot, capped at K=6)")
    ax.set_ylim(0, 0.62)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left", fontsize=9, columnspacing=1.4)
    ax.set_title("BillSum: showing a TF-IDF centrality score next to each sentence is what helps\n", pad=4)
    ax.text(0, 1.015, "Same prompt family, four versions. Removing the scores removes the gain; "
            "removing the role-play does not.", transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig_techniques(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    columns = [("aclsum", "ACLSum"), ("cnndm", "CNN/DailyMail"), ("billsum", "BillSum")]
    delta = {}
    for experiment, label in columns:
        df = _summary(experiment)
        df = df[df.model.isin(LOCAL) & df.technique.isin(BASE_TECHNIQUES)]
        means = df.groupby("technique").f1.mean()
        delta[label] = means - means["vanilla"]

    import pandas as pd
    table = pd.DataFrame(delta).loc[[t for t in BASE_TECHNIQUES if t != "vanilla"]]
    table = table.loc[table["BillSum"].sort_values(ascending=False).index]
    limit = float(np.abs(table.values).max())

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    image = ax.imshow(table.values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(value) > 0.55 * limit else INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.outline.set_visible(False)
    bar.set_label("mean F1 minus the plain prompt's F1")
    fig.suptitle("No technique wins everywhere", x=0.02, ha="left", fontsize=14, fontweight="bold", y=1.03)
    fig.text(0.02, 0.965, "Gain over the plain (vanilla) prompt, mean over local models and variants. "
             "tool_augmented\ngains least on ACLSum and most on both whole-document datasets.",
             fontsize=9, color=MUTED, va="top")
    fig.subplots_adjust(top=0.82)
    fig.savefig(out)
    plt.close(fig)


def fig_overselection(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for ax, (experiment, title) in zip(axes, (("cnndm", "CNN/DailyMail"), ("billsum", "BillSum"))):
        lengths = selection_length(experiment)
        lengths = lengths[lengths.cap == "uncapped"].groupby("model").mean_selected.mean()
        models = [m for m in MODELS if m in lengths.index]
        values = [lengths[m] for m in models]
        colors = [FAMILY_COLOR[MODELS[m][0]] for m in models]
        rects = ax.bar(np.arange(len(models)), values, color=colors, zorder=3, width=0.7)
        for rect, value in zip(rects, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + 0.7, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
        k = baselines(experiment)["k"]
        ax.axhline(k, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([MODELS[m][1] for m in models])
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    axes[0].set_ylabel("sentences selected when no limit is given")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    handles.append(plt.Line2D([0], [0], color=INK, lw=1.2, ls=(0, (4, 3))))
    axes[1].legend(handles, [*FAMILY_COLOR, "silver summary length (K = 3 and K = 6)"], loc="upper right")
    fig.suptitle("Without a cap, Qwen3.5 selects far too much; Gemma4 limits itself",
                 x=0.012, ha="left", fontsize=14, fontweight="bold", y=1.03)
    fig.savefig(out)
    plt.close(fig)


def fig_position(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    shown = [("qwen35_2b", QWEN, "-"), ("qwen35_9b", QWEN, (0, (5, 2))),
             ("gemma4_e4b", GEMMA, "-")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for ax, (experiment, title) in zip(axes, (("cnndm", "CNN/DailyMail"), ("billsum", "BillSum"))):
        table = pd.read_csv(resolve(f"reports/{experiment}/tables/position_bias.csv"))
        centers = np.arange(10) + 0.5

        def share(rows):
            counts = rows.groupby("position_bin").selection_count.sum().reindex(range(10), fill_value=0)
            return (counts / counts.sum()).values

        silver = table[table.selection_source == "silver"]
        silver = silver[silver.variant_slug == silver.variant_slug.iloc[0]]
        ax.bar(centers, share(silver), width=0.9, color="#D5DAE0", zorder=2,
               label="silver labels (what the reference rewards)")
        for model, color, dash in shown:
            rows = table[(table.selection_source == "model") & (table.model == model) & (table.cap == "capped")]
            ax.plot(centers, share(rows), color=color, lw=2.2, ls=dash, marker="o", ms=4.5,
                    zorder=3, label=_label(model))
        ax.axhline(0.1, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xticklabels(["start", "20%", "40%", "60%", "80%", "end"])
        ax.set_xlabel("position in the document")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of selected sentences")
    axes[1].legend(loc="upper right")
    fig.suptitle("Where the selected sentences sit: CNN/DailyMail rewards the opening, BillSum does not",
                 x=0.012, ha="left", fontsize=14, fontweight="bold", y=1.03)
    fig.savefig(out)
    plt.close(fig)


def fig_heuristics(out: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    table = pd.read_csv(resolve("reports/heuristics/k_sweep_singular_rouge1.csv"))
    styles = {
        "local_score": ("local_score (independent top-K)", "#2A9D8F", 2.8),
        "greedy_search": ("greedy search", "#8FA3B8", 1.8),
        "beam_search": ("beam search", "#B8C0CA", 1.8),
        "global_score": ("global score", "#6B7785", 1.8),
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for heuristic, (label, color, width) in styles.items():
        rows = table[table.heuristic == heuristic].sort_values("k")
        ax.plot(rows.k, rows.rouge_avg, "-o", color=color, lw=width, ms=5, label=label, zorder=3)
    local = table[table.heuristic == "local_score"].set_index("k").rouge_avg
    best = int(local.idxmax())
    ax.annotate(f"best: K = {best}\n{local[best]:.3f}", (best, local[best]), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, color="#16695F", fontweight="bold")
    ax.set_xlabel("K (sentences kept)")
    ax.set_ylabel("mean of ROUGE-1/2/L vs the human extractive gold")
    ax.set_ylim(0.33, 0.62)
    ax.legend(loc="lower right")
    ax.set_title("Heuristics on ACLSum: scoring sentences independently wins\n", pad=4)
    ax.text(0, 1.015, "Singular mode, extraction by ROUGE-1, averaged over the three aspects.",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig_aspects(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    df = _summary("aclsum")
    table = df.pivot_table(index="model", columns="aspect", values="f1").loc[list(MODELS)]
    aspects = [("challenge", "#B8C0CA"), ("approach", "#6B9AC4"), ("outcome", "#2A9D8F")]

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    width = 0.26
    x = np.arange(len(table))
    for i, (aspect, color) in enumerate(aspects):
        values = table[aspect].values
        rects = ax.bar(x + (i - 1) * width, values, width, label=aspect, color=color, zorder=3)
        for rect, value in zip(rects, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + 0.008, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in table.index])
    ax.set_ylabel("mean F1 over the variants each model ran")
    ax.set_ylim(0, 0.82)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("ACLSum: 'challenge' is the hard aspect for every model\n", pad=4)
    ax.text(0, 1.015, "Local models: all 54 variants. Qwen3.5-27B and Gemma4-31B: the 18 variants they ran.",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.savefig(out)
    plt.close(fig)


def fig_position_reference(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from .position import BIN_COLUMNS, load_tables

    reference, _ = load_tables()
    panels = [
        ("cnndm", "CNN/DailyMail", "silver labels"),
        ("aclsum", "ACLSum", "human labels, three aspects pooled"),
        ("billsum", "BillSum", "silver labels"),
    ]
    aspect_style = {"challenge": ("#B8C0CA", "-"), "approach": ("#6B9AC4", "-"), "outcome": ("#2A9D8F", "-")}
    centers = np.arange(10) + 0.5

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, (dataset, title, subtitle) in zip(axes, panels):
        row = reference[(reference.dataset == dataset) & (reference.subset == "all")].iloc[0]
        ax.bar(centers, row[BIN_COLUMNS].values.astype(float), width=0.9, color="#C9D1DA", zorder=2)
        if dataset == "aclsum":
            for aspect, (color, dash) in aspect_style.items():
                values = reference[(reference.dataset == dataset) & (reference.subset == aspect)]
                ax.plot(centers, values[BIN_COLUMNS].iloc[0].values.astype(float), color=color, lw=2,
                        ls=dash, marker="o", ms=3.5, zorder=3, label=aspect)
            ax.legend(loc="upper right", fontsize=8.5)
        ax.axhline(0.1, color=INK, lw=1, ls=(0, (2, 3)), zorder=4)
        ax.text(0.03, 0.97, f"first fifth: {100 * row.lead20:.0f}%\nmean position: {row.mean_pos:.2f}",
                transform=ax.transAxes, va="top", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xticks([0, 5, 10])
        ax.set_xticklabels(["start", "middle", "end"])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{title}\n", pad=4)
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    axes[0].set_ylabel("share of reference sentences")
    fig.suptitle("Where the reference sentences sit (dotted line: no positional preference)",
                 x=0.012, ha="left", fontsize=14, fontweight="bold", y=1.06)
    fig.savefig(out)
    plt.close(fig)


def fig_position_models(out: Path) -> None:
    import matplotlib.pyplot as plt

    from .position import load_tables

    reference, models = load_tables(rebuild=False)
    panels = [("cnndm", "CNN/DailyMail"), ("billsum", "BillSum"), ("aclsum", "ACLSum")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharex=True)
    for ax, (dataset, title) in zip(axes, panels):
        target = reference[(reference.dataset == dataset) & (reference.subset == "all")].lead20.iloc[0]
        rows = models[models.dataset == dataset]
        names = [m for m in MODELS if m in set(rows.model)][::-1]
        for y, model in enumerate(names):
            answered = rows[(rows.model == model) & (rows["mode"] == "as_answered")].lead20.iloc[0]
            capped = rows[(rows.model == model) & (rows["mode"] == "after_cap")].lead20.iloc[0]
            color = FAMILY_COLOR[MODELS[model][0]]
            ax.plot([target, answered], [y, y], color=color, lw=2.5, alpha=0.35, zorder=2,
                    solid_capstyle="round")
            if abs(capped - answered) > 0.01:
                ax.plot([answered, capped], [y, y], color=color, lw=1.2, ls=(0, (2, 2)), zorder=2)
                ax.plot(capped, y, "o", mfc="white", mec=color, mew=1.6, ms=8, zorder=3)
            ax.plot(answered, y, "o", color=color, ms=9, zorder=4)
            if max(answered, capped) > 0.8:  # no room to the right: label above
                ax.text(answered, y + 0.28, f"{100 * (answered - target):+.0f}",
                        va="bottom", ha="center", fontsize=9, color=color)
            elif answered >= target:
                ax.text(max(answered, capped) + 0.025, y, f"{100 * (answered - target):+.0f}",
                        va="center", ha="left", fontsize=9, color=color)
            else:
                ax.text(answered - 0.025, y, f"{100 * (answered - target):+.0f}",
                        va="center", ha="right", fontsize=9, color=color)
        ax.axvline(target, color=INK, lw=1.6, zorder=1)
        ax.text(target, len(names) - 0.35, f" reference {100 * target:.0f}%", fontsize=9, color=INK,
                va="bottom", ha="left")
        ax.axvline(0.2, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([_label(m) for m in names])
        ax.set_ylim(-0.6, len(names) + 0.1)
        ax.set_xlim(0.12, 1.0)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["20%", "40%", "60%", "80%", "100%"])
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("share of an answer's sentences in the first fifth")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", mfc=MUTED, mec=MUTED, ms=9),
        plt.Line2D([0], [0], marker="o", color="none", mfc="white", mec=MUTED, mew=1.6, ms=8),
        plt.Line2D([0], [0], color=INK, lw=1.6),
        plt.Line2D([0], [0], color=MUTED, lw=1, ls=(0, (2, 3))),
    ]
    fig.legend(handles, ["as the model answered", "after the cap kept the first K",
                         "reference labels", "no positional preference (20%)"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=9.5)
    fig.suptitle("Lead bias of each model against its dataset's labels (numbers: percentage points from the labels)",
                 x=0.012, ha="left", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(out)
    plt.close(fig)


def fig_position_shift(out: Path) -> None:
    import matplotlib.pyplot as plt

    from .position import load_tables

    reference, models = load_tables(rebuild=False)
    answered = models[models["mode"] == "as_answered"]
    both = [m for m in MODELS
            if {"cnndm", "billsum"} <= set(answered[answered.model == m].dataset)]

    def lead(frame, **where):
        for key, value in where.items():
            frame = frame[frame[key] == value]
        return float(frame.lead20.iloc[0])

    def spread(values, gap=0.03):
        """Label heights: the values, nudged apart so no two labels overlap."""
        order = sorted(range(len(values)), key=lambda i: values[i])
        placed = list(values)
        for previous, current in zip(order, order[1:]):
            placed[current] = max(placed[current], placed[previous] + gap)
        return placed

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ref = [lead(reference, dataset=d, subset="all") for d in ("cnndm", "billsum")]
    series = [("reference labels", INK, ref, True)]
    for model in both:
        values = [lead(answered, dataset=d, model=model) for d in ("cnndm", "billsum")]
        series.append((_label(model), FAMILY_COLOR[MODELS[model][0]], values, False))

    left = spread([values[0] for _, _, values, _ in series])
    right = spread([values[1] for _, _, values, _ in series])
    for (name, color, values, bold), y_left, y_right in zip(series, left, right):
        ax.plot([0, 1], values, "-o", color=color, lw=3 if bold else 2, ms=9 if bold else 6.5,
                zorder=5 if bold else 3)
        weight = "bold" if bold else "normal"
        ax.text(1.04, y_right, f"{name}  {100 * values[1]:.0f}%", va="center", fontsize=9.5,
                color=color, fontweight=weight)
        ax.text(-0.04, y_left, f"{100 * values[0]:.0f}%", va="center", ha="right", fontsize=9.5,
                color=color, fontweight=weight)
    ax.axhline(0.2, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(0.5, 0.205, "no positional preference", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CNN/DailyMail", "BillSum"], fontsize=11, color=INK)
    ax.set_xlim(-0.28, 1.75)
    ax.set_ylim(0.15, 0.95)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"])
    ax.set_ylabel("share of an answer's sentences in the first fifth")
    ax.grid(axis="x", visible=False)
    for side in ("bottom",):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_title("On BillSum the labels are near uniform; every model still leans on the opening\n", pad=4)
    ax.text(0, 1.015, "Capped prompts, as the model answered. All eight models.",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.savefig(out)
    plt.close(fig)


FIGURES = {
    "scaling": fig_scaling,
    "billsum_tfidf": fig_tfidf,
    "techniques": fig_techniques,
    "overselection": fig_overselection,
    "position_bias": fig_position,
    "heuristics_k": fig_heuristics,
    "aclsum_aspects": fig_aspects,
    "position_reference": fig_position_reference,
    "position_models": fig_position_models,
    "position_shift": fig_position_shift,
}


def build_figures(out_dir: str = "reports/figures") -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")

    _style()
    target = resolve(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name, draw in FIGURES.items():
        path = target / f"{name}.png"
        draw(path)
        print(f"[figures] {path}")
        written.append(path)
    return written
