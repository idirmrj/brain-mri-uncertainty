"""
is the model's confidence trustworthy?

A model can be 95% accurate but badly calibrated: it says "90% sure" and is
right only 70% of the time. This measures and fixes that.

  - ECE (Expected Calibration Error): gap between confidence and accuracy
  - reliability diagram: confidence (x) vs actual accuracy (y); the diagonal
    is perfect calibration
  - temperature scaling: a 1-parameter post-hoc fix that rescales the logits

Run:
  python src/mriunc/calibration.py --data "path/to/data" --ckpt results/best_model.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import build_dataloaders
from uncertainty import load_model

ACCENT = "#7B2CBF"


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for x, y in loader:
        logits_all.append(model(x.to(device)).cpu())
        labels_all.append(y)
    return torch.cat(logits_all), torch.cat(labels_all)


def compute_ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    correct = pred.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    xs, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            acc_bin = correct[m].float().mean().item()
            conf_bin = conf[m].mean().item()
            ece += (m.float().mean().item()) * abs(acc_bin - conf_bin)
            xs.append(((bins[i] + bins[i + 1]) / 2).item())
            accs.append(acc_bin); confs.append(conf_bin); counts.append(int(m.sum()))
    return ece, np.array(xs), np.array(accs), np.array(confs)


def fit_temperature(logits, labels, max_iter=100):
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
    opt.step(closure)
    return T.detach().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="results/best_model.pt")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out", default="results/calibration.png")
    args = ap.parse_args()

    model, classes, device = load_model(args.ckpt)
    _, val_loader, test_loader, _ = build_dataloaders(args.data, batch_size=32,
                                                      num_workers=args.num_workers)

    val_logits, val_labels = collect_logits(model, val_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)

    T = fit_temperature(val_logits, val_labels)
    print(f"Fitted temperature T = {T:.3f}  (T>1 means the model was overconfident)")

    probs_before = F.softmax(test_logits, dim=1)
    probs_after = F.softmax(test_logits / T, dim=1)

    ece_b, xb, accb, confb = compute_ece(probs_before, test_labels)
    ece_a, xa, acca, confa = compute_ece(probs_after, test_labels)
    print(f"ECE before: {ece_b:.4f}   ECE after temperature scaling: {ece_a:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, (x, acc, conf, ece, title) in zip(ax, [
        (xb, accb, confb, ece_b, f"Before (ECE={ece_b:.3f})"),
        (xa, acca, confa, ece_a, f"After T-scaling (ECE={ece_a:.3f})"),
    ]):
        a.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="perfect")
        a.plot(conf, acc, "o-", color=ACCENT, lw=2, label="model")
        a.set_xlabel("confidence"); a.set_ylabel("accuracy")
        a.set_title(title); a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.legend(); a.grid(alpha=0.3)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
