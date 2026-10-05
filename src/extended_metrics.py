"""
Collapse diagnostics for the extended task.

Accuracy alone is the wrong instrument here. The base task's output is
deterministic, so the only thing recursive training can damage is correctness.
The extended task's middle token, x_next, is a coin flip between the context
symbols, and a model can drift away from that coin flip long before any
accuracy number moves. These functions measure the drift directly.

Where collapse can show up
--------------------------
Per sequence, the truth puts 1/K on each context symbol and nothing anywhere
else. next_symbol_stats() reads the model's own probabilities on the fixed test
contexts and measures the two ways that can go wrong: committing to a favourite
(choice entropy falls) and leaking probability to symbols that were never in
the context (off-context mass rises).

Where it cannot
---------------
The histogram of emitted symbols over the whole vocabulary (symbol_counts +
distribution_stats) stays uniform even when every choice is deterministic: each
sequence offers K symbols drawn at random, so whichever one the model picks is
itself a random draw. A model that always picks the first context symbol scores
the same 3.91 nats as the truth. Those stats are kept as a global sanity check
-- they do catch a model that avoids particular symbols everywhere -- but they
cannot see per-sequence collapse.

Everything here is a pure function of tensors -- no I/O, no model -- so the
values can be checked against the analytic reference for the true distribution
(see extended_data.py):

    next-symbol choice entropy    log K = 0.693 nats for K = 2
    preferred-symbol probability  1/K   = 0.50
    off-context mass              0
    KL(truth || model) at x_next  0
    symbol histogram entropy      log(50) = 3.912 nats
    symbol coverage               50 / 50
    type validity                 1.0
    context consistency           1.0
"""

import math

import torch


def per_token_accuracy(pred_ids, true_ids):
    """
    pred_ids, true_ids: [N, 3] joint-vocab ids.
    Returns a list of three accuracies, one per predicted token.
    """
    correct = (pred_ids == true_ids)
    return [correct[:, t].float().mean().item() for t in range(3)]


def symbol_counts(cont_ids, num_classes):
    """
    Histogram over the symbol vocabulary for the emitted x_next token.

    Ids outside the symbol range are type errors, not symbols, so they are
    excluded here; type_validity() reports them separately rather than letting
    them silently distort the distribution.
    """
    ids = cont_ids[:, 1]
    ids = ids[(ids >= 0) & (ids < num_classes)]
    return torch.bincount(ids, minlength=num_classes).float()


def label_counts(cont_ids, column, num_classes, num_labels):
    """Histogram over the label vocabulary for column 0 (y_q) or 2 (y_next)."""
    ids = cont_ids[:, column] - num_classes
    ids = ids[(ids >= 0) & (ids < num_labels)]
    return torch.bincount(ids, minlength=num_labels).float()


def distribution_stats(counts):
    """
    Summarise a histogram's shape.

    KL(p || uniform) = log V - H(p), so a perfectly uniform empirical
    distribution gives entropy log V and KL exactly 0. Collapse shows up as
    entropy falling, KL rising, coverage dropping (tails dying) and top-1 mass
    rising (modes taking over).

    effective_support = exp(H) is the number of symbols the distribution
    behaves as if it had -- easier to read than nats when writing up.
    """
    counts = counts.double()
    total = counts.sum().item()
    vocab = counts.numel()
    log_v = math.log(vocab) if vocab > 1 else 0.0

    if total == 0:
        nan = float("nan")
        return {
            "entropy": nan, "entropy_normalized": nan, "kl_to_uniform": nan,
            "coverage": 0.0, "coverage_fraction": nan, "top1_mass": nan,
            "effective_support": nan,
        }

    p = counts / total
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum().item())
    coverage = float((counts > 0).sum().item())

    return {
        "entropy": entropy,
        "entropy_normalized": entropy / log_v if log_v > 0 else float("nan"),
        "kl_to_uniform": log_v - entropy,
        "coverage": coverage,
        "coverage_fraction": coverage / vocab,
        "top1_mass": float(p.max().item()),
        "effective_support": math.exp(entropy),
    }


def next_symbol_stats(probs, context_classes):
    """
    The per-sequence view of x_next -- the one collapse actually shows up in.

    probs            [N, vocab] the model's distribution at the x_next slot,
                     read on N contexts
    context_classes  [N, K] the symbols that appeared in each context

    The truth puts 1/K on each context symbol and nothing anywhere else. Each
    quantity is a mean over sequences:

        context_entropy   entropy of the model's choice among the context
                          symbols, renormalised so that leakage cannot inflate
                          it. Truth log K. Falls as the model commits to
                          favourites.
        preferred_prob    renormalised probability of its favourite context
                          symbol. Truth 1/K.
        off_context_mass  probability given to anything that is not a context
                          symbol -- wrong symbols and wrong-type tokens alike.
                          Truth 0. Rises as errors leak into the data.
        kl_to_truth       KL(truth || model), which combines both failure
                          modes. Truth 0. Equal in expectation to the x_next
                          test loss minus log K, but computed exactly instead
                          of from one sampled target per sequence.

    This has to be computed from probabilities. A generated dataset holds one
    sampled x_next per context, so no statistic of the data alone can show how
    sure the model was when it chose.
    """
    K = context_classes.shape[1]
    ctx_p = probs.gather(1, context_classes)  # [N, K]
    in_mass = ctx_p.sum(dim=1)
    renorm = ctx_p / in_mass.clamp_min(1e-12)[:, None]
    context_entropy = -(renorm * renorm.clamp_min(1e-12).log()).sum(dim=1)
    # KL(u || p) with u uniform on the K context symbols
    kl = -math.log(K) - ctx_p.clamp_min(1e-12).log().mean(dim=1)

    return {
        "context_entropy": float(context_entropy.mean().item()),
        "preferred_prob": float(renorm.max(dim=1).values.mean().item()),
        "off_context_mass": float((1.0 - in_mass).mean().item()),
        "kl_to_truth": float(kl.mean().item()),
    }


def type_validity(cont_ids, num_classes, num_labels):
    """
    Fraction of emitted tokens that are of the correct type for their slot:
    columns 0 and 2 must be label ids (>= num_classes), column 1 must be a
    symbol id (< num_classes).

    Logits are not type-masked, so the symbol/label alternation is a real
    thing the model has to learn -- usually trivially. Expect 1.0 on a trained
    model; a drop is evidence that collapse has damaged something basic.
    """
    vocab = num_classes + num_labels
    ok_yq = (cont_ids[:, 0] >= num_classes) & (cont_ids[:, 0] < vocab)
    ok_x = (cont_ids[:, 1] >= 0) & (cont_ids[:, 1] < num_classes)
    ok_yn = (cont_ids[:, 2] >= num_classes) & (cont_ids[:, 2] < vocab)
    return float(torch.stack([ok_yq, ok_x, ok_yn]).float().mean().item())


def context_consistency(cont_ids, meta, num_classes):
    """
    A behavioural probe of symbol -> value binding.

    Of the sequences where the emitted x_next is actually one of the context
    classes, what fraction have an emitted y_next equal to that class's
    within-sequence label?

    This asks whether the model still resolves symbol -> value on its OWN
    generated symbol, independently of whether that symbol was a sensible thing
    to emit. The two failure modes are separable: a model whose next-symbol
    choice has collapsed onto one favourite can still be perfectly
    self-consistent, whereas a model whose binding has degraded cannot be.
    Tracking this alongside the next-symbol stats tells us whether collapse is
    eroding the distribution, the binding, or both. It is behavioural evidence
    only: attributing it to a particular circuit needs attention analysis or
    ablations.

    Returns (consistency, in_context_fraction). The second value is the share
    of sequences the metric could be computed on at all -- if the model emits
    symbols that never appeared in the context, the question is undefined for
    those rows and they are excluded from the numerator and denominator.
    """
    emitted_sym = cont_ids[:, 1]
    emitted_val = cont_ids[:, 2]
    ctx_classes = meta["context_classes"]
    ctx_labels = meta["context_labels"]

    match = (ctx_classes == emitted_sym[:, None])  # [N, K]
    in_context = match.any(dim=1)
    n_in_context = int(in_context.sum().item())

    if n_in_context == 0:
        return float("nan"), 0.0

    # context classes are sampled without replacement, so at most one match
    slot = match.float().argmax(dim=1)
    true_label = ctx_labels[torch.arange(ctx_labels.shape[0]), slot]
    true_value_id = true_label + num_classes

    agree = (emitted_val == true_value_id) & in_context
    consistency = float(agree.sum().item()) / n_in_context
    return consistency, float(in_context.float().mean().item())
