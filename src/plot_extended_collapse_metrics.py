"""
Plot Extended Model Collapse metrics.

Produces:
  results/extended_train_loss.png            train loss, one line per generation
  results/extended_val_loss.png              val loss, one line per generation
  results/extended_val_acc_per_token.png     val accuracy, one panel per token
  results/extended_collapse_summary.png      the main result, four panels
  results/extended_symbol_rank_frequency.png symbol rank-frequency per generation

Usage:
    python src/plot_extended_collapse_metrics.py --csv results/extended_collapse_metrics.csv

Colour decisions
----------------
Generation is an ORDINAL variable, so generations are encoded with a single-hue
light-to-dark blue ramp rather than arbitrary categorical colours -- the reader
sees the progression without consulting the legend. The ramp starts at a step
that still clears 2:1 against the chart surface so generation 0 is never a ghost.

The three predicted tokens are an IDENTITY variable, so they take the first
three categorical slots, which are the ones that stay separable under colour
vision deficiency when every pair can appear together. They also carry direct
labels, so identity is never colour-alone.

Analytic reference values are drawn as muted dashed rules, not as series -- they
are properties of the task, not measurements.

These figures are for a printed report, so they commit to the light surface.
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --- chart chrome -----------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# single-hue blue ramp, steps 250..700; ordinal-safe against the light surface
GEN_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

# categorical slots 1-3, the set that stays separable across all pairs
TOKEN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TOKEN_NAMES = ["y_q  (query label)", "x_next  (next symbol)", "y_next  (its value)"]

SERIES_COLOR = "#2a78d6"  # single-series panels


def generation_colors(generations):
    """
    Map ordered generations onto the blue ramp, evenly spaced across it.
    Colour follows the generation, never its position in a filtered subset.
    """
    n = len(generations)
    if n == 1:
        return {generations[0]: GEN_RAMP[len(GEN_RAMP) // 2]}
    step = (len(GEN_RAMP) - 1) / (n - 1)
    return {g: GEN_RAMP[round(i * step)] for i, g in enumerate(generations)}


def style_axes(ax, xlabel, ylabel, title):
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=10, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)


def reference_rule(ax, y, label):
    """An analytic property of the task, not a measurement."""
    ax.axhline(y, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(label, xy=(0.995, y), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7.5, color=INK_MUTED)


def spread_labels(values, min_gap):
    """
    Nudge direct labels apart vertically so overlapping series stay readable.

    Series that converge -- which is exactly what we expect y_q and y_next to do
    once the circuit has formed -- would otherwise stack their labels on top of
    each other. Returns label positions in the same order as `values`; the lines
    themselves are untouched.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for slot, i in enumerate(order):
        if slot == 0:
            continue
        below = placed[order[slot - 1]]
        if placed[i] - below < min_gap:
            placed[i] = below + min_gap
    return placed


def new_figure(ncols=1, width=7.2, height=4.4):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)
    return fig, ([axes] if ncols == 1 else list(axes))


def save(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {out_path}")


# --- figures ----------------------------------------------------------------

def plot_step_metric(df, metric, ylabel, title, out_path, reference=None):
    generations = sorted(df["generation"].unique())
    colors = generation_colors(generations)
    fig, (ax,) = new_figure()

    for gen in generations:
        sub = df[df["generation"] == gen]
        ax.plot(sub["step"], sub[metric], color=colors[gen], linewidth=2,
                label=f"Gen {gen}", zorder=3)

    if reference is not None:
        reference_rule(ax, reference[0], reference[1])

    style_axes(ax, "Training step", ylabel, title)
    legend = ax.legend(title="Generation", frameon=False, fontsize=8,
                       title_fontsize=8, loc="best")
    legend.get_title().set_color(INK_SECONDARY)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save(fig, out_path)


def plot_val_acc_per_token(df, cfg_like, out_path):
    """Small multiples: one panel per predicted token, one line per generation."""
    generations = sorted(df["generation"].unique())
    colors = generation_colors(generations)
    chance = 1.0 / cfg_like["num_labels"]
    ceiling = 1.0 / cfg_like["context_len"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURFACE,
                             sharey=True)
    refs = [
        (chance, f"chance {chance:.2f}"),
        (ceiling, f"ceiling {ceiling:.2f}  (1/K)"),
        (chance, f"chance {chance:.2f}"),
    ]

    for t, ax in enumerate(axes):
        for gen in generations:
            sub = df[df["generation"] == gen]
            ax.plot(sub["step"], sub[f"val_acc_t{t + 1}"], color=colors[gen],
                    linewidth=2, label=f"Gen {gen}", zorder=3)
        reference_rule(ax, refs[t][0], refs[t][1])
        style_axes(ax, "Training step", "Validation accuracy" if t == 0 else "",
                   TOKEN_NAMES[t])
        ax.set_ylim(0, 1.02)

    legend = axes[-1].legend(title="Generation", frameon=False, fontsize=8,
                             title_fontsize=8, loc="center right")
    legend.get_title().set_color(INK_SECONDARY)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.suptitle("Validation accuracy per predicted token",
                 color=INK_PRIMARY, fontsize=12, x=0.008, ha="left")
    save(fig, out_path)


def plot_collapse_summary(summary, cfg_like, out_path):
    """
    The main result: what happens across generations.

    Four panels rather than one dual-axis chart -- accuracy, nats, symbol counts
    and a fraction do not share a scale and must not be forced onto one.
    """
    gens = summary["generation"]
    num_classes = cfg_like["num_classes"]
    chance = 1.0 / cfg_like["num_labels"]
    ceiling = 1.0 / cfg_like["context_len"]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3), facecolor=SURFACE)

    # 1. per-token test accuracy -- identity, so categorical + direct labels
    ax = axes[0]
    end_values = [summary[f"test_acc_t{t + 1}"].iloc[-1] for t in range(3)]
    label_y = spread_labels(end_values, min_gap=0.075)
    for t in range(3):
        ax.plot(gens, summary[f"test_acc_t{t + 1}"], color=TOKEN_COLORS[t],
                linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=TOKEN_NAMES[t], zorder=3)
        ax.annotate(TOKEN_NAMES[t].split()[0],
                    xy=(gens.iloc[-1], label_y[t]),
                    xytext=(6, 0), textcoords="offset points",
                    color=TOKEN_COLORS[t], fontsize=8, va="center",
                    annotation_clip=False)
    reference_rule(ax, chance, f"chance {chance:.2f}")
    reference_rule(ax, ceiling, f"1/K {ceiling:.2f}")
    style_axes(ax, "Generation", "Test accuracy", "In-context accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(gens)
    legend = ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    # 2. symbol entropy
    ax = axes[1]
    ax.plot(gens, summary["data_symbol_entropy"], color=SERIES_COLOR,
            linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    reference_rule(ax, math.log(num_classes), f"true  log {num_classes} = {math.log(num_classes):.2f}")
    style_axes(ax, "Generation", "Entropy (nats)", "Symbol distribution entropy")
    ax.set_ylim(0, math.log(num_classes) * 1.15)
    ax.set_xticks(gens)

    # 3. symbol coverage -- the tail
    ax = axes[2]
    ax.plot(gens, summary["data_symbol_coverage"], color=SERIES_COLOR,
            linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    reference_rule(ax, num_classes, f"all {num_classes} symbols")
    style_axes(ax, "Generation", "Symbols used at least once",
               "Tail coverage")
    ax.set_ylim(0, num_classes * 1.15)
    ax.set_xticks(gens)

    # 4. the mechanistic probe
    ax = axes[3]
    ax.plot(gens, summary["data_context_consistency"], color=SERIES_COLOR,
            linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    reference_rule(ax, 1.0, "perfect")
    style_axes(ax, "Generation", "Self-consistency",
               "Value matches own emitted symbol")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(gens)

    fig.suptitle("Extended model collapse across generations",
                 color=INK_PRIMARY, fontsize=12, x=0.005, ha="left")
    save(fig, out_path)


def plot_symbol_rank_frequency(counts_df, out_path):
    """
    Rank-frequency of emitted symbols, one line per generation.

    This is the shape-of-the-distribution view that an accuracy number cannot
    show: under collapse the curve steepens and the tail falls away as symbols
    stop being emitted at all.
    """
    generations = sorted(counts_df["generation"].unique())
    colors = generation_colors(generations)
    fig, (ax,) = new_figure(width=7.6, height=4.6)

    for gen in generations:
        sub = counts_df[counts_df["generation"] == gen]
        ranked = sub["count"].sort_values(ascending=False).to_numpy()
        ax.plot(range(1, len(ranked) + 1), ranked, color=colors[gen],
                linewidth=2, label=f"Gen {gen}", zorder=3)

    gen0 = counts_df[counts_df["generation"] == generations[0]]
    uniform = gen0["count"].sum() / max(len(gen0), 1)
    reference_rule(ax, uniform, f"uniform  {uniform:.0f} per symbol")

    ax.set_yscale("symlog", linthresh=1)
    style_axes(ax, "Symbol rank (most to least emitted)",
               "Times emitted (symlog)",
               "Symbol rank-frequency by generation")
    legend = ax.legend(title="Generation", frameon=False, fontsize=8,
                       title_fontsize=8, loc="best")
    legend.get_title().set_color(INK_SECONDARY)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save(fig, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/extended_collapse_metrics.csv")
    parser.add_argument("--config", default="config_extended_collapse.yaml",
                        help="used only for the analytic reference lines")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(args.csv)
    out_dir = os.path.dirname(args.csv) or "."
    summary_path = args.csv.replace(".csv", "_test_summary.csv")
    counts_path = args.csv.replace(".csv", "_symbol_counts.csv")

    token2_floor = math.log(cfg["context_len"])

    plot_step_metric(
        df, "train_loss", "Train loss (mean over 3 tokens)",
        "Train loss, all generations",
        os.path.join(out_dir, "extended_train_loss.png"),
    )
    plot_step_metric(
        df, "val_loss", "Validation loss (mean over 3 tokens)",
        "Validation loss, all generations",
        os.path.join(out_dir, "extended_val_loss.png"),
    )
    plot_step_metric(
        df, "val_loss_t2", "Cross-entropy (nats)",
        "Next-symbol loss -- irreducible floor is log K",
        os.path.join(out_dir, "extended_val_loss_next_symbol.png"),
        reference=(token2_floor, f"floor  log {cfg['context_len']} = {token2_floor:.3f}"),
    )
    plot_val_acc_per_token(
        df, cfg, os.path.join(out_dir, "extended_val_acc_per_token.png"),
    )

    if os.path.exists(summary_path):
        plot_collapse_summary(
            pd.read_csv(summary_path), cfg,
            os.path.join(out_dir, "extended_collapse_summary.png"),
        )
    else:
        print(f"Warning: {summary_path} not found, skipping summary figure.")

    if os.path.exists(counts_path):
        plot_symbol_rank_frequency(
            pd.read_csv(counts_path),
            os.path.join(out_dir, "extended_symbol_rank_frequency.png"),
        )
    else:
        print(f"Warning: {counts_path} not found, skipping rank-frequency figure.")


if __name__ == "__main__":
    main()
