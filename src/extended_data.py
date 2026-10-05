"""
Extended in-context learning task.

Base sequence (T = 2K+1), one token predicted:

    x1 y1 x2 y2 ... xK yK xq                        -> predict y_q

Extended sequence (T = 2K+4), three tokens predicted autoregressively:

    x1 y1 x2 y2 ... xK yK xq  y_q  x_next  y_next

    position 2K   -> y_q      the query class's label
    position 2K+1 -> x_next   the next symbol
    position 2K+2 -> y_next   that symbol's value

Tokens live in one joint vocabulary of size num_classes + num_labels:

    symbol c -> id c
    label  l -> id num_classes + l

This is exactly the layout SymbolICLGenerator already uses for its one-hot
inputs, so the same encoding serves both the inputs and the targets.

Ground truth
------------
y_q     the query class's within-sequence label (deterministic).

x_next  uniform over the K context classes, INCLUDING the query class.
        Excluding the query would make x_next deterministic when K = 2 (only
        one other class left to pick), the token-2 loss floor would be zero,
        and there would be no distribution for model collapse to contract --
        which would defeat the entire point of the extended task.

y_next  the within-sequence label of x_next (deterministic given x_next and
        the context, and recoverable only by the induction circuit).

Per sequence, the truth is uniform over the K context symbols. That gives exact
analytic reference values for the collapse metrics (see extended_metrics.py):

    choice entropy        log(K)             = 0.693 nats for K = 2
    preferred-symbol prob 1/K                = 0.50 for K = 2
    off-context mass      0
    token-2 CE floor      log(K)             = 0.693 nats for K = 2
    token-2 accuracy cap  1/K                = 0.50 for K = 2

Because the K context classes are a uniform sample without replacement from
num_classes, the true marginal over x_next is uniform over ALL num_classes
classes:

    symbol entropy        log(num_classes)   = 3.912 nats for 50 classes
    symbol KL to uniform  0
    symbol coverage       num_classes

The marginal values hold for ANY rule that picks one of the context symbols,
however deterministic, so they are a sanity check rather than a collapse
measure. Collapse moves the per-sequence values.
"""

import random

import torch


class ExtendedSymbolICLGenerator:
    """
    Extended counterpart of SymbolICLGenerator in data.py.

    sample() returns (x, cont_ids, meta):
        x        [B, 2K+4, input_dim]  one-hot sequence, continuation filled in
        cont_ids [B, 3]                joint-vocab ids of the three targets
        meta     dict of context bookkeeping needed by the metrics
    """

    def __init__(self, num_classes=50, num_labels=5, context_len=2, device="cpu"):
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.context_len = context_len
        self.input_dim = num_classes + num_labels
        self.vocab_size = num_classes + num_labels
        self.device = device

        K = context_len
        self.seq_len = 2 * K + 4
        # positions we read logits from; each predicts the token after it
        self.pred_positions = [2 * K, 2 * K + 1, 2 * K + 2]
        # positions the three predicted tokens themselves occupy
        self.cont_positions = [2 * K + 1, 2 * K + 2, 2 * K + 3]

    def label_id(self, lab):
        """Joint-vocab id of a label."""
        return self.num_classes + lab

    def write_continuation(self, x, cont_ids):
        """
        Write the one-hots for a [B,3] tensor of joint-vocab ids into the three
        continuation positions of x, in place.

        Generation 0 calls this with the true continuation; generation i calls
        it with model-sampled ids. Either way the teacher-forced inputs always
        agree with the targets being trained against, which is what keeps
        training and autoregressive rollout consistent with each other.
        """
        rows = torch.arange(x.shape[0], device=x.device)
        ids = cont_ids.to(x.device)
        for t, pos in enumerate(self.cont_positions):
            x[:, pos, :] = 0.0
            x[rows, pos, ids[:, t]] = 1.0
        return x

    def blank_continuation(self, x):
        """
        Return a copy of x with the three continuation positions zeroed,
        leaving only the context and the query. Rollout starts from this.

        Zeroed future slots are harmless during the rollout because the mask is
        causal -- position 2K can never attend to 2K+1 onwards anyway.
        """
        x = x.clone()
        for pos in self.cont_positions:
            x[:, pos, :] = 0.0
        return x

    def sample(self, batch_size):
        B, K = batch_size, self.context_len
        x = torch.zeros(B, self.seq_len, self.input_dim, device=self.device)
        cont_ids = torch.empty(B, 3, dtype=torch.long)

        context_classes = torch.empty(B, K, dtype=torch.long)
        context_labels = torch.empty(B, K, dtype=torch.long)
        query_idx = torch.empty(B, dtype=torch.long)
        next_idx = torch.empty(B, dtype=torch.long)

        for b in range(B):
            classes = random.sample(range(self.num_classes), K)
            # random labels, consistent within this sequence
            labels = [random.randrange(self.num_labels) for _ in range(K)]
            q = random.randrange(K)
            # next symbol: uniform over the context slots, query included
            n = random.randrange(K)

            for j, (cls, lab) in enumerate(zip(classes, labels)):
                x[b, 2 * j, cls] = 1.0
                x[b, 2 * j + 1, self.label_id(lab)] = 1.0

            x[b, 2 * K, classes[q]] = 1.0  # query symbol

            cont_ids[b, 0] = self.label_id(labels[q])  # y_q
            cont_ids[b, 1] = classes[n]                # x_next
            cont_ids[b, 2] = self.label_id(labels[n])  # y_next

            context_classes[b] = torch.tensor(classes)
            context_labels[b] = torch.tensor(labels)
            query_idx[b] = q
            next_idx[b] = n

        self.write_continuation(x, cont_ids)

        meta = {
            "context_classes": context_classes,
            "context_labels": context_labels,
            "query_idx": query_idx,
            "next_idx": next_idx,
        }
        return x, cont_ids, meta
