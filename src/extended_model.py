"""
Extended-task model.

The transformer stack is IMPORTED from model.py rather than copied, so the
extended model and the base model share the same blocks by construction. The
base-vs-extended contrast is the central result of this project and it only
holds if the architectures are identical; importing makes that a fact rather
than a claim that has to be re-checked whenever anyone tunes the base model.

The only difference is the readout:

    base      Linear(d_model, num_labels) read at the final position
    extended  Linear(d_model, vocab_size) read at three positions

vocab_size = num_classes + num_labels, which is exactly the dimensionality of
the input encoding: SymbolICLGenerator already writes symbol c to index c and
label l to index num_classes + l. The head is therefore the dual of the input
embedding, and inputs and targets share one joint vocabulary.
"""

import torch.nn as nn

from model import AttentionOnlyBlock


class ExtendedICLTransformer(nn.Module):
    """
    Attention-only causal Transformer with a joint-vocabulary readout.

    Logits are NOT type-masked. A competent model learns the symbol/label
    alternation trivially, so type validity sits at 1.0 once trained; it only
    becomes informative if collapse degrades it, which is a prediction worth
    being able to falsify.
    """

    def __init__(self, input_dim, vocab_size, d_model=64, num_heads=8, depth=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            AttentionOnlyBlock(d_model, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, positions, return_attn=False):
        """
        x         [B, T, input_dim]
        positions list of ints -- the positions to read logits from. Each one
                  predicts the token that follows it.
        returns   [B, len(positions), vocab_size]
        """
        h = self.input_proj(x)
        attn_maps = []
        for block in self.blocks:
            h, a = block(h, return_attn=return_attn)
            if return_attn:
                attn_maps.append(a)
        h = self.norm(h)
        logits = self.head(h[:, positions])
        return (logits, attn_maps) if return_attn else logits
