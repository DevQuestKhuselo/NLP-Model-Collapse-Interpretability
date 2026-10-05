"""
Extended Model Collapse: dataset generation and persistence.

Mirrors collapse_data.py for the three-token extended task.

Generation 0 (real data):
    - train/val/test are all sampled from the true ExtendedSymbolICLGenerator
      distribution with true continuations.
    - val and test are FIXED: sampled once and reused, unchanged, for every
      subsequent generation.

Generation i >= 1:
    - New contexts (classes + true random label assignment + query) are sampled
      from the same true distribution as generation 0.
    - The three CONTINUATION tokens are replaced by model_{i-1}'s own
      autoregressive rollout on that context. This is the only place model
      error enters the pipeline; the context distribution itself stays
      controlled and true throughout, matching the convention the base pipeline
      already established so that base and extended remain a like-for-like
      comparison.
    - The training pool for generation i is the ACCUMULATION of
      Data^0 u Data^1 u ... u Data^i. Validation and test are NOT accumulated;
      they stay fixed at their generation-0 values.

Why the rollout is autoregressive
---------------------------------
The third token is "the value corresponding to the symbol the model just
emitted". Reading all three positions from a single forward pass would let the
model choose a value without conditioning on the symbol it actually produced,
which is precisely the dependency the extended task exists to test. So the
rollout runs three passes, writing each sampled token into the next input slot
before predicting the one after it.

Why sampling rather than argmax
-------------------------------
The base pipeline takes an argmax because its single target is deterministic,
so the argmax IS the model's belief. The extended task's middle token is a
coin flip between the context symbols: under argmax the model would always
emit its favourite of the two, so the coin flip would be gone from the data
after a single generation, leaving nothing to measure. Sampling preserves it,
so collapse can appear gradually -- as the model's choice sharpening, and as
probability leaking to symbols that were never in the context. The histogram
of emitted symbols would NOT show even the argmax collapse: whichever context
symbol gets picked is itself a uniform draw from the vocabulary (see
extended_metrics.py). `decode: argmax` remains available as a deliberate
contrast run.
"""

import torch

from extended_data import ExtendedSymbolICLGenerator


def make_true_extended_generator(cfg, device):
    """Generator matching the extended task's true distribution."""
    return ExtendedSymbolICLGenerator(
        num_classes=cfg["num_classes"],
        num_labels=cfg["num_labels"],
        context_len=cfg["context_len"],
        device=device,
    )


def _select(logits, decode, temperature):
    """One decoding step over the joint vocabulary."""
    if decode == "argmax":
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def rollout_continuation(model, x, generator, decode="sample", temperature=1.0,
                         device="cpu", chunk=512):
    """
    Autoregressively decode the three continuation tokens from `model`.

    x is a [B, T, D] batch whose context and query positions are already
    filled; whatever sits in the three continuation positions is discarded and
    overwritten with the model's own output.

    decode:
        "sample"  multinomial over the full joint vocabulary (default)
        "argmax"  greedy
        "hybrid"  greedy for the two deterministic tokens (y_q, y_next),
                  multinomial for the genuinely stochastic x_next

    Returns (x_with_generated_continuation, cont_ids [B,3]), both on CPU.
    """
    model.eval()
    out_x, out_ids = [], []

    for start in range(0, x.shape[0], chunk):
        xb = generator.blank_continuation(x[start:start + chunk].to(device))
        rows = torch.arange(xb.shape[0], device=device)
        ids_b = torch.zeros(xb.shape[0], 3, dtype=torch.long, device=device)

        for t, read_pos in enumerate(generator.pred_positions):
            logits = model(xb, positions=[read_pos])[:, 0]
            step = ("sample" if t == 1 else "argmax") if decode == "hybrid" else decode
            ids = _select(logits, step, temperature)
            ids_b[:, t] = ids

            # feed what we just produced into the next input slot
            write_pos = generator.cont_positions[t]
            xb[:, write_pos, :] = 0.0
            xb[rows, write_pos, ids] = 1.0

        out_x.append(xb.cpu())
        out_ids.append(ids_b.cpu())

    model.train()
    return torch.cat(out_x, dim=0), torch.cat(out_ids, dim=0)


def generate_fixed_extended_eval_sets(cfg, device):
    """
    Sample the FIXED validation and test sets once, from the true distribution.
    Reused unchanged across all generations so results stay comparable.
    """
    val_gen = make_true_extended_generator(cfg, device)
    test_gen = make_true_extended_generator(cfg, device)
    return val_gen.sample(cfg["val_size"]), test_gen.sample(cfg["test_size"])


def generate_extended_generation_zero_train(cfg, device):
    """Data^0: true contexts, true continuations. Size = train_pool_size."""
    gen = make_true_extended_generator(cfg, device)
    return gen.sample(cfg["train_pool_size"])


def generate_next_extended_generation_train(cfg, device, model):
    """
    Data^i (i >= 1): true contexts, continuations rolled out by model_{i-1}.

    Returns (x, cont_ids, true_cont_ids, meta) where true_cont_ids is the
    continuation the true distribution would have produced for the same
    context -- discarded from training, kept only to measure how far the
    model-generated data has drifted from the truth.
    """
    gen = make_true_extended_generator(cfg, device)
    x, true_cont_ids, meta = gen.sample(cfg["train_pool_size"])
    x, cont_ids = rollout_continuation(
        model, x, gen,
        decode=cfg.get("decode", "sample"),
        temperature=cfg.get("temperature", 1.0),
        device=device,
    )
    return x, cont_ids, true_cont_ids, meta
