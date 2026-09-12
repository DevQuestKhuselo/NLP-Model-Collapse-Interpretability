import argparse
import pandas as pd
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/symbolic_metrics.csv")
args = ap.parse_args()
df = pd.read_csv(args.csv)

plt.figure(figsize=(7,4))
plt.plot(df["step"], df["train_loss"], label="train")
plt.plot(df["step"], df["val_loss"], label="validation")
plt.xlabel("Training step"); plt.ylabel("Cross-entropy loss")
plt.legend(); plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=180)
plt.close()

plt.figure(figsize=(7,4))
plt.plot(df["step"], df["train_acc"], label="train")
plt.plot(df["step"], df["val_acc"], label="validation")
plt.axhline(0.2, linestyle="--", label="chance (5 labels)")
plt.xlabel("Training step"); plt.ylabel("Accuracy")
plt.legend(); plt.tight_layout()
plt.savefig("results/accuracy_curve.png", dpi=180)
plt.close()
