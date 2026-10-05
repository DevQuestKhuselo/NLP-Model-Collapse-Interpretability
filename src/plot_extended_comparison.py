"""
Compare extended collapse runs: model-generated continuations against the
matched real-data control, across seeds.

This is the figure the experiment's conclusions rest on. The training pool
grows every generation in both conditions, so whatever the two share is the
effect of having more data; only the gap between them is the effect of
training on model output. Each seed is drawn as its own thin line and the mean
over seeds as a thick one, so the reader can see how much of a trend survives
replication before reading anything into it.

Usage:
    python src/plot_extended_comparison.py "results/extended_collapse_metrics*_test_summary.csv"

Arguments are per-run summary CSVs or glob patterns (expanded here, so quoted
patterns work in PowerShell too). Runs are grouped into conditions by their
continuation_source and decode columns; each file is one replicate.

Produces:
  results/extended_comparison.png           six panels, conditions overlaid
  results/extended_comparison_summary.csv   per-condition, per-generation mean,
                                            min and max over seeds
"""

import argparse
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from plot_extended_collapse_metrics import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    accuracy_baselines,
    spread_labels,
    style_axes,
)

# Colour follows the condition, never its position in the subset being
# plotted. Categorical slots 1-3 -- the set that stays separable across every
# pair -- so hybrid and argmax share slot 3 and cannot appear together.
CONDITIONS = {
    ("model", "sample"): ("Model-generated", "model", "#2a78d6"),
    ("real", None): ("Real-data control", "control", "#eb6834"),
    ("model", "hybrid"): ("Model-generated, hybrid decode", "hybrid", "#1baf7a"),
    ("model", "argmax"): ("Model-generated, argmax decode", "argmax", "#1baf7a"),
}

METRICS = [
    # column, panel title, y label
    ("test_acc_t1", "y_q accuracy", "Test accuracy"),
    ("test_acc_t3", "y_next accuracy", "Test accuracy"),
    ("test_consistency", "Self-consistency on own rollouts", "Share of sequences"),
    ("test_next_context_entropy", "Next-symbol choice entropy", "Entropy (nats)"),
    ("test_next_off_context_mass", "Probability outside the context", "Probability"),
    ("test_next_kl_to_truth", "KL from truth at x_next", "KL (nats)"),
]


def condition_key(run):
    source = run["continuation_source"].iloc[0]
    if source == "real":
        return ("real", None)
    return ("model", run["decode"].iloc[0])


def load_runs(patterns):
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    runs = {}
    for path in dict.fromkeys(paths):
        run = pd.read_csv(path)
        key = condition_key(run)
        if key not in CONDITIONS:
            raise ValueError(f"{path}: unknown condition {key}")
        runs.setdefault(key, []).append(run)

    if ("model", "hybrid") in runs and ("model", "argmax") in runs:
        raise ValueError("hybrid and argmax share a colour slot; plot them in "
                         "separate figures")
    # fixed order, so the legend and the colours never reshuffle
    return {key: runs[key] for key in CONDITIONS if key in runs}


def aggregate(runs):
    """Per-condition, per-generation mean / min / max over seeds."""
    rows = []
    for key, replicates in runs.items():
        stacked = pd.concat(replicates)
        grouped = stacked.groupby("generation")
        for gen, group in grouped:
            row = {"condition": CONDITIONS[key][0], "generation": gen,
                   "n_seeds": len(group)}
            for column, _, _ in METRICS:
                row[f"{column}_mean"] = group[column].mean()
                row[f"{column}_min"] = group[column].min()
                row[f"{column}_max"] = group[column].max()
            rows.append(row)
    return pd.DataFrame(rows)


def reference(column, cfg_like):
    """(value, description) of the analytic reference drawn in each panel."""
    K = cfg_like["context_len"]
    _, no_binding = accuracy_baselines(cfg_like)
    if column in ("test_acc_t1", "test_acc_t3"):
        return no_binding, f"no binding, {no_binding:.2f}"
    if column == "test_consistency":
        return 1.0, "perfect, 1.0"
    if column == "test_next_context_entropy":
        return math.log(K), f"truth, log {K} = {math.log(K):.3f}"
    return 0.0, "truth, 0"


def add_reference(ax, value):
    """
    The dashed rule alone. Its value goes in the panel title instead of beside
    the line, where it would collide with the direct labels at the line ends.
    """
    ax.axhline(value, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
               zorder=1)


def set_limits(ax, column, values, cfg_like):
    K = cfg_like["context_len"]
    top = float(values.max())
    low = float(values.min())
    if column in ("test_acc_t1", "test_acc_t3", "test_consistency"):
        # never hide the 0.60 no-binding line, and never clip a real drop
        ax.set_ylim(min(0.5, math.floor(low * 10) / 10), 1.02)
    elif column == "test_next_context_entropy":
        ax.set_ylim(0, max(math.log(K), top) * 1.25)
    elif column == "test_next_off_context_mass":
        ax.set_ylim(0, max(0.05, top * 1.3))
    else:
        ax.set_ylim(0, max(0.05, top * 1.2))


def plot_comparison(runs, cfg_like, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), facecolor=SURFACE)
    generations = sorted(pd.concat(
        [run for reps in runs.values() for run in reps])["generation"].unique())

    for ax, (column, title, ylabel) in zip(axes.flat, METRICS):
        all_values = []
        ends, names = [], []
        for key, replicates in runs.items():
            label, short, color = CONDITIONS[key]
            for run in replicates:
                ax.plot(run["generation"], run[column], color=color,
                        linewidth=1, alpha=0.35, zorder=2)
                all_values.append(run[column])
            mean = pd.concat(replicates).groupby("generation")[column].mean()
            ax.plot(mean.index, mean.values, color=color, linewidth=2,
                    marker="o", markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3,
                    label=f"{label}  ({len(replicates)} "
                          f"seed{'s' if len(replicates) != 1 else ''})")
            ends.append(float(mean.values[-1]))
            names.append(short)

        ref_value, ref_text = reference(column, cfg_like)
        add_reference(ax, ref_value)
        style_axes(ax, "Generation", ylabel, f"{title}   (dashed: {ref_text})")
        set_limits(ax, column, pd.concat(all_values), cfg_like)
        ax.set_xticks(generations)

        # direct labels at the line ends, in text ink; the line end beside
        # each one carries the colour
        lo, hi = ax.get_ylim()
        for name, y in zip(names, spread_labels(ends, min_gap=0.07 * (hi - lo))):
            ax.annotate(name, xy=(generations[-1], y), xytext=(8, 0),
                        textcoords="offset points", color=INK_SECONDARY,
                        fontsize=8, va="center", annotation_clip=False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper left", ncol=len(labels),
                        frameon=False, fontsize=9, bbox_to_anchor=(0.005, 0.95))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.suptitle("Extended model collapse: model-generated data vs real-data "
                 "control  (thick: mean over seeds, thin: each seed)",
                 color=INK_PRIMARY, fontsize=12, x=0.005, ha="left")
    # leave room above the panels for the title and the shared legend; the
    # shared save() helper re-runs tight_layout without that margin
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+",
                        help="per-run *_test_summary.csv files or glob patterns")
    parser.add_argument("--config", default="config_extended_collapse.yaml",
                        help="used only for the analytic reference lines")
    parser.add_argument("--out", default="results/extended_comparison.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    runs = load_runs(args.summaries)
    for key, replicates in runs.items():
        print(f"{CONDITIONS[key][0]}: {len(replicates)} run(s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_comparison(runs, cfg, args.out)

    table_path = os.path.splitext(args.out)[0] + "_summary.csv"
    aggregate(runs).to_csv(table_path, index=False)
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
