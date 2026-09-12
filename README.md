# ICL Induction-Head Baseline (PyTorch)

This repository is a compact baseline for the Wits NLP Project 1 task:
build a Transformer that learns a symbol-based in-context learning task,
then move toward reproducing the Singh et al. (2024) induction-head setup.

## What the model learns

Each sequence is:

    exemplar_1, label_1, exemplar_2, label_2, ..., query_exemplar

The query repeats one of the context classes. Labels are randomly reassigned
for every sequence, so the network cannot solve the task by memorising a
global class->label map. It has to recover the mapping from the current context.

For K=2:

    A, 3, B, 1, A  -> target 3

A different sequence may use:

    A, 0, C, 4, A  -> target 0

## Architecture

- causal Transformer
- 2 layers
- 8 heads
- d_model = 64
- RoPE
- attention-only blocks (no MLP)
- LayerNorm + residual connections
- classify from final query position

These choices match the important defaults used by the Singh et al. codebase.

## Install

    python -m venv .venv
    # Linux/macOS
    source .venv/bin/activate
    # Windows
    .venv\Scripts\activate

    pip install -r requirements.txt

## 1. Fast symbolic sanity check

    python src/train.py --config config_symbolic.yaml

Expected behaviour:
- with 5 labels, chance accuracy is 20%
- validation accuracy should rise well above chance
- use this mode to debug the pipeline before expensive reproduction runs

Then:

    python src/plot_metrics.py --csv results/symbolic_metrics.csv
    python src/inspect_attention.py --checkpoint checkpoints/symbolic_baseline.pt

## 2. Singh-style Omniglot-feature run

Download/copy the authors' supplied file:

    omniglot_resnet18_randomized_order_s0.h5

to:

    data/omniglot_resnet18_randomized_order_s0.h5

Then run:

    python src/train.py --config config_singh.yaml

The Singh main-paper command uses:
- 1,000,000 iterations
- batch size 32
- learning rate 1e-5
- context length 2
- 5 random labels
- d_model 64
- default depth 2
- default 8 attention heads
- default RoPE
- attention-only model

IMPORTANT:
This project is a compact PyTorch reimplementation of the core task/architecture,
not a bit-for-bit port of the authors' JAX/Equinox code. For strict numerical
reproduction of their paper figures, run their official repository's
`ih_paper_runs.sh`.

## Outputs

Training creates:

    results/<mode>_metrics.csv
    checkpoints/<mode>_baseline.pt

Plotting creates:

    results/loss_curve.png
    results/accuracy_curve.png

Attention inspection creates one heatmap per layer/head.

## What to report to the group

1. Verify the model exceeds 20% chance accuracy.
2. Show loss and accuracy curves.
3. Show that labels are random across sequences.
4. Save the baseline checkpoint.
5. Inspect layer/head attention maps.
6. Hand the checkpoint + config to the collapse-pipeline teammate.

## Scientific caution

High accuracy alone does not prove that a specific induction-head circuit has
formed. Attention patterns, head ablations, or other mechanistic interventions
are needed before claiming a particular circuit.
