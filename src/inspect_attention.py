import argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from train import make_generators, resolve_device
from model import ICLTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/symbolic_baseline.pt")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    device = resolve_device(cfg.get("device","auto"))

    _, _, gen, input_dim = make_generators(cfg, device)
    model = ICLTransformer(input_dim, cfg["num_labels"], cfg["d_model"],
                           cfg["num_heads"], cfg["depth"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x,y = gen.sample(1)
    with torch.no_grad():
        logits, maps = model(x, return_attn=True)

    print("target:", int(y.item()))
    print("prediction:", int(logits.argmax(-1).item()))

    for layer, A in enumerate(maps):
        A = A[0].cpu().numpy()  # [heads,T,T]
        for head in range(A.shape[0]):
            plt.figure(figsize=(5,4))
            plt.imshow(A[head], aspect="auto")
            plt.colorbar()
            plt.xlabel("Key position")
            plt.ylabel("Query position")
            plt.title(f"Layer {layer} Head {head}")
            plt.tight_layout()
            plt.savefig(f"results/attention_L{layer}_H{head}.png", dpi=160)
            plt.close()

if __name__ == "__main__":
    main()
