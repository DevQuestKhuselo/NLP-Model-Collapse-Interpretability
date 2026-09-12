import argparse, csv, json, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data import SymbolICLGenerator, OmniglotICLGenerator
from model import ICLTransformer

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

@torch.no_grad()
def evaluate(model, generator, batches, batch_size):
    model.eval()
    losses, correct, total = [], 0, 0
    for _ in range(batches):
        x,y = generator.sample(batch_size)
        logits = model(x)
        losses.append(F.cross_entropy(logits,y).item())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    model.train()
    return float(np.mean(losses)), correct/total

def make_generators(cfg, device):
    if cfg["mode"] == "symbolic":
        train = SymbolICLGenerator(cfg["num_classes"], cfg["num_labels"], cfg["context_len"], device)
        val   = SymbolICLGenerator(cfg["num_classes"], cfg["num_labels"], cfg["context_len"], device)
        test  = SymbolICLGenerator(cfg["num_classes"], cfg["num_labels"], cfg["context_len"], device)
        return train, val, test, train.input_dim

    if cfg["mode"] == "omniglot":
        path = cfg["data_file"]
        train_n, val_n, test_n = cfg["num_train_classes"], cfg["num_val_classes"], cfg["num_test_classes"]
        train = OmniglotICLGenerator(path, 0, train_n, range(cfg["train_exemplars"]),
                                     cfg["num_labels"], cfg["context_len"], device)
        val = OmniglotICLGenerator(path, train_n, val_n, range(cfg["train_exemplars"], cfg["train_exemplars"]+cfg["val_exemplars"]),
                                   cfg["num_labels"], cfg["context_len"], device)
        test = OmniglotICLGenerator(path, train_n+val_n, test_n, range(cfg["train_exemplars"]),
                                    cfg["num_labels"], cfg["context_len"], device)
        return train, val, test, train.input_dim

    raise ValueError("mode must be symbolic or omniglot")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_symbolic.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device","auto"))
    print("device:", device)

    train_gen, val_gen, test_gen, input_dim = make_generators(cfg, device)
    model = ICLTransformer(input_dim=input_dim,
                           num_labels=cfg["num_labels"],
                           d_model=cfg["d_model"],
                           num_heads=cfg["num_heads"],
                           depth=cfg["depth"]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    outdir = Path("results"); outdir.mkdir(exist_ok=True)
    ckptdir = Path("checkpoints"); ckptdir.mkdir(exist_ok=True)
    metrics_path = outdir / f'{cfg["mode"]}_metrics.csv'

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step","train_loss","train_acc","val_loss","val_acc"])

        for step in range(1, cfg["train_steps"]+1):
            x,y = train_gen.sample(cfg["batch_size"])
            logits = model(x)
            loss = F.cross_entropy(logits,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step == 1 or step % cfg["eval_every"] == 0:
                train_acc = (logits.argmax(-1)==y).float().mean().item()
                val_loss, val_acc = evaluate(model,val_gen,cfg["eval_batches"],cfg["batch_size"])
                print(f"{step:7d} train_loss={loss.item():.4f} train_acc={train_acc:.3f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
                writer.writerow([step,loss.item(),train_acc,val_loss,val_acc])
                f.flush()

    test_loss, test_acc = evaluate(model,test_gen,cfg["eval_batches"],cfg["batch_size"])
    print(f"FINAL test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    ckpt = {
        "model_state": model.state_dict(),
        "config": cfg,
        "input_dim": input_dim,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    torch.save(ckpt, ckptdir / f'{cfg["mode"]}_baseline.pt')

if __name__ == "__main__":
    main()
