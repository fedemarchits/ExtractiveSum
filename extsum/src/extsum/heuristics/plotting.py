"""Plotting helpers — all save to PNG (Agg backend, no display)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save(out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_avg_score(result, title, out_dir, filename):
    max_sent_list = sorted(result.keys(), reverse=True)
    heuristics = list(next(iter(result.values())).keys())

    plt.figure(figsize=(10, 6))

    for heur in heuristics:
        y_values = [result[max_sent][heur] for max_sent in max_sent_list]
        plt.plot(max_sent_list, y_values, label=heur)

    avg_values = [
        sum(result[max_sent][heur] for heur in heuristics) / len(heuristics)
        for max_sent in max_sent_list
    ]
    plt.plot(max_sent_list, avg_values, marker="o", color="black", label="Average", linewidth=2)

    max_idx = avg_values.index(max(avg_values))
    max_x = max_sent_list[max_idx]
    max_y = avg_values[max_idx]

    plt.annotate(
        f"{max_x} sentences",
        xy=(max_x, max_y),
        xytext=(max_x, max_y + 0.02),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        ha="center",
    )

    plt.xlabel("Max Sentences")
    plt.ylabel("Average Score")
    plt.title(title)
    plt.xticks(max_sent_list)
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _save(out_dir, filename)


def plot_total_time(results, title, out_dir, filename):
    heuristic_times = {}
    for summary_type in results:
        for heuristic, data in results[summary_type].items():
            if data["time"] is not None:
                heuristic_times[heuristic] = heuristic_times.get(heuristic, 0) + data["time"]

    df = pd.DataFrame(
        list(heuristic_times.items()),
        columns=["Heuristic", "Total Time (s)"],
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x="Heuristic", y="Total Time (s)", data=df,
        hue="Heuristic", palette="viridis", legend=False,
    )
    plt.title(title)
    plt.ylabel("Total Time (seconds)")
    plt.xlabel("Heuristic")
    plt.xticks(rotation=45, ha="right")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    return _save(out_dir, filename)


def plot_histogram_with_values(
    data, out_dir, filename, x_label="Heuristic", y_label="R1", title="Histogram"
):
    df = pd.DataFrame(list(data.items()), columns=[x_label, y_label])

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=x_label, y=y_label, data=df,
        hue=x_label, palette="viridis", legend=False,
    )
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(rotation=45, ha="right")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    return _save(out_dir, filename)


def plot_prf_grouped(prf_data, out_dir, filename, title="P / R / F1"):
    """prf_data: {heuristic: {'precision': p, 'recall': r, 'fmeasure': f}}.

    Renders 3 grouped bars per heuristic.
    """
    heurs = list(prf_data.keys())
    metrics = ["precision", "recall", "fmeasure"]
    x = np.arange(len(heurs))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for i, m in enumerate(metrics):
        vals = [prf_data[h][m] for h in heurs]
        bars = plt.bar(x + i * width, vals, width, label=m)
        for b, v in zip(bars, vals):
            plt.annotate(
                f"{v:.2f}",
                (b.get_x() + b.get_width() / 2.0, v),
                ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points",
                fontsize=8,
            )

    plt.xticks(x + width, heurs, rotation=45, ha="right")
    plt.ylabel("score")
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    return _save(out_dir, filename)


def plot_prf_per_summary(prf_singular, out_dir, filename, title="P / R / F1 by summary"):
    """prf_singular: {summary: {heuristic: {p, r, f}}} — one subplot per summary type."""
    summaries = list(prf_singular.keys())
    fig, axes = plt.subplots(1, len(summaries), figsize=(6 * len(summaries), 5), sharey=True)
    if len(summaries) == 1:
        axes = [axes]

    metrics = ["precision", "recall", "fmeasure"]
    width = 0.25

    for ax, summ in zip(axes, summaries):
        heurs = list(prf_singular[summ].keys())
        x = np.arange(len(heurs))
        for i, m in enumerate(metrics):
            vals = [prf_singular[summ][h][m] for h in heurs]
            ax.bar(x + i * width, vals, width, label=m)
        ax.set_xticks(x + width)
        ax.set_xticklabels(heurs, rotation=45, ha="right")
        ax.set_title(summ)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    axes[-1].legend()
    fig.suptitle(title)
    return _save(out_dir, filename)
