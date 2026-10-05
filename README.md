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
 
## 4. Extended Model Collapse

The base task asks for one token, and that token is deterministic given the
context. Recursive training can therefore only corrupt *correctness* — there is
almost no distributional information to lose, which is why the base pipeline is
not expected to collapse much.

The extended task asks for **three** tokens. Given the same context, the model
predicts the query's label, then **the next symbol in the sequence**, then
**that symbol's corresponding value**:

    pos:   0    1    2    3    4     5      6       7
    tok:   x1   y1   x2   y2   xq  | y_q   x_next  y_next
    pred:                       ^     ^      ^

The middle token is the point of the exercise. Which symbol comes next is
genuinely stochastic, so it carries a real distribution over the 50-symbol
vocabulary — and that distribution is something recursive self-training can
contract. Every hyperparameter is identical to `config_collapse.yaml`, so base
vs. extended is a controlled comparison in which the only difference is whether
the output carries a distribution at all.

### Task definition

Tokens live in one joint vocabulary of size `num_classes + num_labels`: symbol
`c` is id `c`, label `l` is id `num_classes + l`. This is exactly the layout the
existing one-hot input encoding already uses, so the output head is the dual of
the input embedding and the architecture barely changes.

- `y_q` — the query class's within-sequence label (deterministic).
- `x_next` — uniform over the K context classes, **including the query class**.
- `y_next` — the within-sequence label of `x_next` (deterministic given
  `x_next`, and recoverable only by the induction circuit).

Including the query class in `x_next` is not a detail. With `K = 2`, excluding
it would make `x_next` deterministic — only one other class is left to pick —
the token-2 loss floor would be zero, and there would be no distribution for
collapse to contract.

Because the K context classes are a uniform sample without replacement, the true
marginal over `x_next` is uniform over all 50 classes. That gives exact analytic
reference values, which every figure draws as a dashed rule:

| Quantity | True-distribution value (K=2, 50 classes, 5 labels) |
|---|---|
| symbol entropy | `log 50` = 3.912 nats |
| symbol KL to uniform | 0 |
| symbol coverage | 50 / 50 |
| token-2 CE floor | `log K` = 0.693 nats |
| token-2 accuracy ceiling | `1/K` = 0.50 |
| token-1 / token-3 chance | 0.20 |

### Pipeline

Structurally identical to the base pipeline — fixed val/test sampled once from
the true distribution, accumulating training pool, fresh model each generation.
Two things differ:

- **The rollout is autoregressive** (three forward passes). The third token is
  "the value corresponding to the symbol the model just emitted", so it has to
  be conditioned on the symbol actually produced.
- **Synthetic data is sampled, not argmaxed.** The base pipeline's argmax is
  correct there because its target is deterministic. Here, argmax would make the
  model emit one favourite symbol for every sequence and destroy the symbol
  distribution in a single generation, leaving nothing to measure. `decode:
  argmax` is kept as a deliberate contrast run; `hybrid` argmaxes the two
  deterministic tokens and samples only the stochastic one.

Contexts always come from the true distribution — only the three continuation
tokens are model-generated, matching the base pipeline's convention.

Run it with:

    python src/train_extended_collapse.py --config config_extended_collapse.yaml

### Metrics

Accuracy alone is the wrong instrument, because the distribution can contract
long before any accuracy number moves. Alongside per-token loss/accuracy/
perplexity, each generation records:

- **symbol entropy, KL to uniform, coverage, top-1 mass** — the shape of the
  emitted symbol distribution. Collapse is entropy falling and coverage dropping
  as tail symbols stop being emitted at all.
- **type validity** — whether emitted tokens are even of the right kind for
  their slot. Logits are deliberately not type-masked, so this is real. Expect
  1.0 on a trained model; it drops sharply on an undertrained one.
- **context consistency** — the mechanistic probe. Of the sequences where the
  emitted `x_next` appears in the context, how often does the emitted `y_next`
  match that symbol's label? This asks whether the induction circuit still
  resolves symbol→value on the model's *own generated* symbol. The two failure
  modes are separable: a model whose symbol distribution has collapsed onto one
  favourite can still be perfectly self-consistent, whereas one whose circuit
  has degraded cannot.

Outputs:

    data/collapse_extended/gen_<i>_train.pt        each generation's training data
    data/collapse_extended/fixed_val.pt            fixed validation set
    data/collapse_extended/fixed_test.pt           fixed test set
    checkpoints/collapse_extended/extended_collapse_gen_<i>.pt
    results/extended_collapse_metrics.csv          step-level train/val metrics
    results/extended_collapse_metrics_test_summary.csv   per-generation summary
    results/extended_collapse_metrics_symbol_counts.csv  per-generation histograms

Reading the summary CSV: the `data_*` columns on row *i* describe Data^i, which
was generated by model *i-1* — so row 0 describes the **true** distribution and
is the reference for every later row. The `test_*` columns on row *i* describe
model *i* itself.

### Plotting extended results

    python src/plot_extended_collapse_metrics.py --csv results/extended_collapse_metrics.csv

Produces:

    results/extended_train_loss.png                train loss, all generations
    results/extended_val_loss.png                  val loss, all generations
    results/extended_val_loss_next_symbol.png      token-2 loss vs its log K floor
    results/extended_val_acc_per_token.png         accuracy, one panel per token
    results/extended_collapse_summary.png          the main result, four panels
    results/extended_symbol_rank_frequency.png     tail die-off across generations

`extended_collapse_summary.png` is the figure for the extended abstract: it puts
in-context accuracy, symbol entropy, tail coverage and self-consistency side by
side, so the reader can see whether the distribution collapses while accuracy
holds, or whether both go together.

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

Two things about the extended task are easy to misread:

**`x_next` accuracy is capped at 1/K by construction.** Under the true
distribution the next symbol is a uniform choice among the K context symbols, so
0.50 at K=2 is a perfect score, not a failing one. The informative quantity at
that position is the *distribution* the model places over symbols, not whether
its argmax happened to land on the sampled one. Reading the mean loss across all
three tokens has the same problem — it is dominated by token 2's irreducible
`log K` term, which is why per-token losses are logged separately.

**Early in training, `x_next` accuracy is exactly zero, and that is expected.**
Two of the three targets are labels drawn from only 5 possibilities, while
symbols are spread over 50, so each label id is roughly 20× more frequent than
each symbol id. Before the model learns the symbol/label alternation its argmax
at the symbol slot is always a label, giving precisely 0.000 rather than chance.
`type_validity` is the diagnostic that separates this from a real failure: it
sits near 0.5 while the model is still emitting the wrong token type, and rises
to 1.0 once the alternation is learned.
