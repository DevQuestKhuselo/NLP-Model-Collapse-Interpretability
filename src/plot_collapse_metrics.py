"""
Plot Base Model Collapse metrics across all generations.

Produces:
  results/collapse_train_loss.png       - train loss curves, one line per generation
  results/collapse_val_loss.png         - val loss curves, one line per generation
  results/collapse_val_acc.png          - val accuracy curves, one line per generation
  results/collapse_test_summary.png     - test acc / test loss / label agreement
                                           vs generation (the main collapse result)

Usage:
    python src/plot_collapse_metrics.py --csv results/collapse_metrics.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_per_step_metric(df, metric, ylabel, title, out_path):
    plt.figure(figsize=(8, 5))
    for gen in sorted(df["generation"].unique()):
        sub = df[df["generation"] == gen]
        plt.plot(sub["step"], sub[metric], label=f"Gen {gen}")
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Generation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_test_summary(summary_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(summary_df["generation"], summary_df["test_acc"], marker="o")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Test accuracy vs generation")
    axes[0].set_xticks(summary_df["generation"])

    axes[1].plot(summary_df["generation"], summary_df["test_loss"], marker="o", color="orange")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Test loss")
    axes[1].set_title("Test loss vs generation")
    axes[1].set_xticks(summary_df["generation"])

    axes[2].plot(
        summary_df["generation"],
        summary_df["train_label_agreement_with_truth"],
        marker="o",
        color="green",
    )
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Agreement with true label")
    axes[2].set_title("Model-generated label agreement\nwith ground truth")
    axes[2].set_xticks(summary_df["generation"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default="results/collapse_metrics.csv",
        help="Path to the step-level collapse metrics CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    summary_path = args.csv.replace(".csv", "_test_summary.csv")

    out_dir = os.path.dirname(args.csv) or "."

    plot_per_step_metric(
        df, "train_loss", "Train loss", "Train loss over training (all generations)",
        os.path.join(out_dir, "collapse_train_loss.png"),
    )
    plot_per_step_metric(
        df, "val_loss", "Validation loss", "Validation loss over training (all generations)",
        os.path.join(out_dir, "collapse_val_loss.png"),
    )
    plot_per_step_metric(
        df, "val_acc", "Validation accuracy", "Validation accuracy over training (all generations)",
        os.path.join(out_dir, "collapse_val_acc.png"),
    )

    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        plot_test_summary(
            summary_df, os.path.join(out_dir, "collapse_test_summary.png")
        )
    else:
        print(f"Warning: {summary_path} not found, skipping test summary plot.")


if __name__ == "__main__":
    main()