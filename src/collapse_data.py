"""
Model Collapse dataset generation and persistence.

Generation 0 (real data):
    - train/val/test are all sampled from the true SymbolICLGenerator
      distribution using true within-sequence labels.
    - val and test are FIXED: sampled once and reused, unchanged, for
      every subsequent generation.

Generation i >= 1:
    - New contexts (classes + true random label assignment) are sampled
      from the same true distribution as generation 0.
    - The TARGET is replaced by model_{i-1}'s prediction on that context.
      This is the only place model error enters the pipeline; the context
      distribution itself stays controlled/true throughout.
    - This generation's (context, model-predicted-target) pairs are saved
      to disk as Data^i.
    - The training pool for generation i is the ACCUMULATION of
      Data^0 ∪ Data^1 ∪ ... ∪ Data^i. Validation and test sets are NOT
      accumulated; they stay fixed at their generation-0 values.
"""

import os
from pathlib import Path

import torch

from data import SymbolICLGenerator


def make_true_generator(cfg, device):
    """Generator matching the baseline task distribution (true labels)."""
    return SymbolICLGenerator(
        num_classes=cfg["num_classes"],
        num_labels=cfg["num_labels"],
        context_len=cfg["context_len"],
        device=device,
    )


def sample_true_batch(generator, n):
    """Sample n true-distribution (x, y) pairs, batched in one call."""
    x, y = generator.sample(n)
    return x, y


@torch.no_grad()
def label_with_model(model, x, device):
    """
    Given contexts x (true classes/labels already embedded, query symbol
    in last position), replace the target with the model's own prediction.
    This is how model error enters generation i's data.
    """
    model.eval()
    x = x.to(device)
    logits = model(x)
    preds = logits.argmax(-1)
    model.train()
    return preds.cpu()


def save_generation(data_dir, gen_idx, x, y, split="train"):
    """
    Save one generation's dataset split to disk as a .pt file.
    File naming: gen_{gen_idx}_{split}.pt
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(data_dir, f"gen_{gen_idx}_{split}.pt")
    torch.save({"x": x.cpu(), "y": y.cpu()}, path)
    return path


def load_generation(data_dir, gen_idx, split="train"):
    path = os.path.join(data_dir, f"gen_{gen_idx}_{split}.pt")
    obj = torch.load(path)
    return obj["x"], obj["y"]


def build_accumulated_train_set(data_dir, up_to_gen):
    """
    Load and concatenate Data^0 ... Data^{up_to_gen} train splits.
    This is the accumulated training pool used to train model_{up_to_gen}.
    """
    xs, ys = [], []
    for g in range(up_to_gen + 1):
        x, y = load_generation(data_dir, g, split="train")
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def generate_fixed_eval_sets(cfg, device):
    """
    Sample the FIXED validation and test sets once, from the true
    distribution. These are reused unchanged across all generations.
    """
    val_gen = make_true_generator(cfg, device)
    test_gen = make_true_generator(cfg, device)

    val_x, val_y = sample_true_batch(val_gen, cfg["val_size"])
    test_x, test_y = sample_true_batch(test_gen, cfg["test_size"])

    return (val_x, val_y), (test_x, test_y)


def generate_generation_zero_train(cfg, device):
    """Data^0: true contexts, true labels. Size = train_pool_size."""
    gen = make_true_generator(cfg, device)
    x, y = sample_true_batch(gen, cfg["train_pool_size"])
    return x, y


def generate_next_generation_train(cfg, device, model, gen_idx):
    """
    Data^i (i >= 1): true contexts, model-predicted targets from
    model_{i-1} (the model trained on generations 0..i-1).
    """
    gen = make_true_generator(cfg, device)
    x, y_true_discarded = sample_true_batch(gen, cfg["train_pool_size"])
    y_pred = label_with_model(model, x, device)
    return x, y_pred, y_true_discarded