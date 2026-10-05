"""
Collapse diagnostics for the extended task.

Accuracy alone is the wrong instrument here. The base task's output is
deterministic, so the only thing recursive training can damage is correctness.
The extended task's middle token carries a real distribution over the symbol
vocabulary, and that distribution can contract long before any accuracy number
moves. These functions measure the contraction directly.

Everything here is a pure function of tensors -- no I/O, no model -- so the
values can be checked against the analytic reference for the true distribution
(see extended_data.py):

    symbol entropy        log(50) = 3.912 nats
    symbol KL to uniform  0
    symbol coverage       50 / 50
    type validity         1.0
    context consistency   1.0
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
    The mechanistic probe.

    Of the sequences where the emitted x_next is actually one of the context
    classes, what fraction have an emitted y_next equal to that class's
    within-sequence label?

    This asks whether the induction circuit still resolves symbol -> value on
    the model's OWN generated symbol, independently of whether that symbol was
    a sensible thing to emit. The two failure modes are separable: a model
    whose symbol distribution has collapsed onto one favourite can still be
    perfectly self-consistent, whereas a model whose circuit has degraded
    cannot be. Tracking this alongside symbol entropy tells us whether collapse
    is eroding the distribution, the mechanism, or both.

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
