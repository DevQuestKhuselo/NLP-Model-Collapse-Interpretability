# ICL Induction-Head Baseline (PyTorch)

This repository is a compact baseline for the Wits NLP Project 1 task:
build a Transformer that learns a symbol-based in-context learning task,
then move toward reproducing the Singh et al. (2024) induction-head setup.
It also includes the Base Model Collapse pipeline , which recursively
retrains this Transformer on its own model-generated data to study model collapse.

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
The same architecture and hyperparameters are reused, unchanged, for the
Model Collapse experiment below.

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

## 3. Base Model Collapse 

This experiment recursively retrains the same Transformer on data it generated
itself, to study whether/how in-context learning behaviour degrades over
generations.

Pipeline:

    Data^0 (real, true labels)          -> train Model_0 -> eval on fixed val/test
    Data^1 (labeled by Model_0)         -> pool = D0+D1   -> train Model_1 (fresh init)
    Data^2 (labeled by Model_1)         -> pool = D0+D1+D2 -> train Model_2 (fresh init)
    ...

Key design points:
- Only the **target** is model-generated each generation; the context (classes
  and true random label assignment) is always sampled from the original true
  distribution. This isolates model error as the sole source of collapse.
- The training pool **accumulates** across generations (D0 ∪ D1 ∪ ... ∪ Di),
  matching the assignment's model-collapse pipeline figure.
- Validation and test sets are sampled **once** from the true distribution and
  reused unchanged for every generation, so results are directly comparable
  across generations.
- Each generation's model is **freshly initialized** and trained from scratch
  on that generation's accumulated pool (no warm-starting from the previous
  generation's weights).
- Architecture and all baseline hyperparameters (d_model, num_heads, depth,
  batch_size, lr, train_steps, etc.) are identical to `config_symbolic.yaml`.

Run it with:

    python src/train_collapse.py --config config_collapse.yaml

Number of generations, per-generation dataset size, and fixed val/test sizes
are set in `config_collapse.yaml` (`num_generations`, `train_pool_size`,
`val_size`, `test_size`).

Outputs:

    data/collapse/gen_<i>_train.pt              - each generation's training data
    data/collapse/fixed_val.pt                  - fixed validation set (all generations)
    data/collapse/fixed_test.pt                 - fixed test set (all generations)
    checkpoints/collapse/collapse_gen_<i>.pt     - model weights per generation
    results/collapse_metrics.csv                - step-level train/val metrics
    results/collapse_metrics_test_summary.csv    - per-generation test loss/acc/
                                                    perplexity and label agreement
                                                    with the true labels

Outputs:
 
    data/collapse/gen_<i>_train.pt              - each generation's training data
    data/collapse/fixed_val.pt                  - fixed validation set (all generations)
    data/collapse/fixed_test.pt                 - fixed test set (all generations)
    checkpoints/collapse/collapse_gen_<i>.pt     - model weights per generation
    results/collapse_metrics.csv                - step-level train/val metrics
    results/collapse_metrics_test_summary.csv    - per-generation test loss/acc/
                                                    perplexity and label agreement
                                                    with the true labels
 
### Plotting collapse results
 
    python src/plot_collapse_metrics.py --csv results/collapse_metrics.csv
 
Requires matplotlib and pandas (`pip install matplotlib pandas` if not already
in your environment). This reads both `results/collapse_metrics.csv` and
`results/collapse_metrics_test_summary.csv` and produces:
 
    results/collapse_train_loss.png       - train loss curves, all generations overlaid
    results/collapse_val_loss.png         - val loss curves, all generations overlaid
    results/collapse_val_acc.png          - val accuracy curves, all generations overlaid
    results/collapse_test_summary.png     - test accuracy / test loss / label agreement
                                             vs. generation (the main collapse result,
                                             three panels side by side)
 
`collapse_test_summary.png` is the figure most likely to go in the extended
abstract, since it directly shows whether accuracy degrades, stays flat, or
improves as synthetic data accumulates across generations.
 
## Outputs
 
Training creates:
 
    results/<mode>_metrics.csv
    checkpoints/<mode>_baseline.pt
 
Plotting creates:
 
    results/loss_curve.png
    results/accuracy_curve.png
 
Attention inspection creates one heatmap per layer/head.
 
## Scientific caution

High accuracy alone does not prove that a specific induction-head circuit has
formed. Attention patterns, head ablations, or other mechanistic interventions
are needed before claiming a particular circuit. Similarly, a lack of visible
collapse in the base model does not mean collapse isn't happening — small
distributional shifts in symbol/label usage can appear before accuracy drops,
so distributional analysis (not just accuracy) matters for this experiment.