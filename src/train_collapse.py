"""
Person 2: Base Model Collapse.

Recursive pipeline:
    Data^0 (real)              -> train Model_0 -> eval on fixed val/test
    Data^1 (Model_0-labeled)   -> pool = D0+D1   -> train Model_1 (fresh init)
    Data^2 (Model_1-labeled)   -> pool = D0+D1+D2 -> train Model_2 (fresh init)
    ...

    - Validation and test sets are sampled ONCE from the true distribution
      and reused, unchanged, for every generation.
    - Training pool accumulates across generations (D0 u D1 u ... u Di).
    - Each generation's model is freshly initialized and trained from
      scratch on that generation's accumulated pool (Data^i is "Fit" in
      Figure 2's pipeline, not a warm start from Model_{i-1}).
    - Architecture and hyperparameters are identical to the baseline
      (config_symbolic.yaml values), per the project's controlled-
      comparison requirement.
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model import ICLTransformer
from collapse_data import (
    generate_fixed_eval_sets,
    generate_generation_zero_train,
    generate_next_generation_train,
    save_generation,
    build_accumulated_train_set,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def sample_batch(x_pool, y_pool, batch_size, device):
    """Random minibatch (with replacement) from a fixed tensor pool."""
    n = x_pool.shape[0]
    idx = torch.randint(0, n, (batch_size,))
    return x_pool[idx].to(device), y_pool[idx].to(device)


@torch.no_grad()
def evaluate_on_set(model, x, y, device, batch_size=256):
    """Full-pass evaluation over a fixed (x, y) tensor set."""
    model.eval()
    n = x.shape[0]
    losses, correct = [], 0

    for start in range(0, n, batch_size):
        xb = x[start:start + batch_size].to(device)
        yb = y[start:start + batch_size].to(device)
        logits = model(xb)
        losses.append(F.cross_entropy(logits, yb).item())
        correct += (logits.argmax(-1) == yb).sum().item()

    model.train()
    avg_loss = float(np.mean(losses))
    acc = correct / n
    perplexity = float(np.exp(avg_loss))
    return avg_loss, acc, perplexity


def build_model(cfg, input_dim, device):
    model = ICLTransformer(
        input_dim=input_dim,
        num_labels=cfg["num_labels"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        depth=cfg["depth"],
    ).to(device)
    return model


def train_one_generation(cfg, model, train_x, train_y, val_x, val_y, device,
                          gen_idx, csv_writer, csv_file):
    """
    Train `model` from scratch on (train_x, train_y) for cfg['train_steps']
    steps, periodically evaluating on the fixed val set. Logs every
    eval_every steps to the metrics CSV with a `generation` column.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    for step in range(1, cfg["train_steps"] + 1):
        xb, yb = sample_batch(train_x, train_y, cfg["batch_size"], device)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg["eval_every"] == 0 or step == cfg["train_steps"]:
            train_loss = loss.item()
            train_acc = (logits.argmax(-1) == yb).float().mean().item()

            val_loss, val_acc, val_ppl = evaluate_on_set(
                model, val_x, val_y, device, batch_size=cfg["batch_size"]
            )

            print(
                f"[gen {gen_idx}] step {step}/{cfg['train_steps']} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            csv_writer.writerow({
                "generation": gen_idx,
                "step": step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_perplexity": val_ppl,
                "train_pool_size": train_x.shape[0],
            })
            csv_file.flush()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    data_dir = cfg["data_dir"]
    checkpoint_dir = cfg["checkpoint_dir"]
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(cfg["results_csv"])).mkdir(parents=True, exist_ok=True)

    # --- Fixed val/test sets, sampled once from the true distribution ---
    (val_x, val_y), (test_x, test_y) = generate_fixed_eval_sets(cfg, device)
    torch.save({"x": val_x, "y": val_y}, os.path.join(data_dir, "fixed_val.pt"))
    torch.save({"x": test_x, "y": test_y}, os.path.join(data_dir, "fixed_test.pt"))

    input_dim = cfg["num_classes"] + cfg["num_labels"]

    csv_file = open(cfg["results_csv"], "w", newline="")
    fieldnames = [
        "generation", "step", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_perplexity", "train_pool_size",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    test_results = []  # (generation, test_loss, test_acc, test_ppl, agreement)
    model = None

    for gen_idx in range(cfg["num_generations"]):
        print(f"\n=== Generation {gen_idx} ===")

        # --- Build this generation's Data^i and save it ---
        if gen_idx == 0:
            gen_x, gen_y = generate_generation_zero_train(cfg, device)
            true_label_agreement = 1.0  # gen 0 labels ARE the true labels
        else:
            gen_x, gen_y, y_true_discarded = generate_next_generation_train(
                cfg, device, model, gen_idx
            )
            true_label_agreement = (gen_y == y_true_discarded).float().mean().item()
            print(f"Gen {gen_idx} label agreement with true labels: "
                  f"{true_label_agreement:.4f}")

        save_generation(data_dir, gen_idx, gen_x, gen_y, split="train")

        # --- Accumulated training pool: D0 u D1 u ... u D{gen_idx} ---
        train_x, train_y = build_accumulated_train_set(data_dir, gen_idx)
        print(f"Accumulated training pool size: {train_x.shape[0]}")

        # --- Fresh model each generation, trained from scratch ---
        model = build_model(cfg, input_dim, device)

        model = train_one_generation(
            cfg, model, train_x, train_y, val_x, val_y, device,
            gen_idx, csv_writer, csv_file,
        )

        # --- Checkpoint ---
        ckpt_path = os.path.join(checkpoint_dir, f"collapse_gen_{gen_idx}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # --- Test on the FIXED test set ---
        test_loss, test_acc, test_ppl = evaluate_on_set(
            model, test_x, test_y, device, batch_size=cfg["batch_size"]
        )
        print(f"[gen {gen_idx}] TEST loss={test_loss:.4f} acc={test_acc:.4f} "
              f"ppl={test_ppl:.4f}")
        test_results.append((gen_idx, test_loss, test_acc, test_ppl,
                              true_label_agreement))

    csv_file.close()

    # --- Separate summary CSV: one row per generation, test metrics ---
    summary_path = cfg["results_csv"].replace(".csv", "_test_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "generation", "test_loss", "test_acc", "test_perplexity",
            "train_label_agreement_with_truth",
        ])
        writer.writeheader()
        for gen_idx, test_loss, test_acc, test_ppl, agreement in test_results:
            writer.writerow({
                "generation": gen_idx,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_perplexity": test_ppl,
                "train_label_agreement_with_truth": agreement,
            })

    print(f"\nDone. Step-level metrics: {cfg['results_csv']}")
    print(f"Per-generation test summary: {summary_path}")


if __name__ == "__main__":
    main()
