"""
run_experiments.py - launches a sweep of training configs, all logged to MLflow.

Covers the comparisons you want to defend in interviews:
  - backbone:      resnet18 vs resnet34
  - dropout:       0.3 / 0.5 / 0.7   (directly affects MC Dropout uncertainty)
  - encoder:       frozen vs fine-tuned
  - deep ensemble: N models with different seeds (built from the sweep runs)

Each run logs params + val/test accuracy + uncertainty quality (entropy ratio)
to MLflow, so you get one comparison table.

Run:
  python src/mriunc/run_experiments.py --data "path/to/data" --epochs 12
Then compare in the MLflow UI.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import mlflow

from data import build_dataloaders
from model import MRIClassifier
from uncertainty import evaluate_with_uncertainty


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train_one(cfg, data, epochs, device):
    """Train a single config, return (model, classes, best_val_acc)."""
    set_seed(cfg["seed"])
    train_loader, val_loader, test_loader, classes = build_dataloaders(
        data, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        seed=cfg["seed"])
    model = MRIClassifier(num_classes=len(classes), backbone=cfg["backbone"],
                          dropout=cfg["dropout"], pretrained=True,
                          freeze_encoder=cfg["freeze_encoder"]).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                           lr=cfg["lr"])
    best_val, best_state = 0.0, None
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        # val
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item(); total += x.size(0)
        va = correct / total
        if va > best_val:
            best_val = va; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, classes, best_val, test_loader, device


def uncertainty_quality(model, test_loader, device, n_passes=20):
    """Entropy ratio wrong/correct + test acc, using MC Dropout."""
    r = evaluate_with_uncertainty(model, test_loader, device, n_passes=n_passes)
    acc = r["correct"].float().mean().item()
    ec = r["entropy"][r["correct"]].mean().item()
    ew = r["entropy"][~r["correct"]].mean().item()
    return acc, ew / max(ec, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--ensemble_size", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    Path("results").mkdir(exist_ok=True)

    base = dict(batch_size=32, lr=1e-4, num_workers=args.num_workers,
                freeze_encoder=False, seed=42)

    # --- 1) config sweep -----------------------------------------------------
    sweep = [
        {**base, "backbone": "resnet18", "dropout": 0.3},
        {**base, "backbone": "resnet18", "dropout": 0.5},
        {**base, "backbone": "resnet18", "dropout": 0.7},
        {**base, "backbone": "resnet34", "dropout": 0.5},
        {**base, "backbone": "resnet18", "dropout": 0.5, "freeze_encoder": True},
    ]

    mlflow.set_experiment("brain-mri-uncertainty")
    for i, cfg in enumerate(sweep, 1):
        name = f"{cfg['backbone']}_d{cfg['dropout']}_{'frozen' if cfg['freeze_encoder'] else 'ft'}"
        print(f"\n[{i}/{len(sweep)}] {name}")
        with mlflow.start_run(run_name=name):
            model, classes, best_val, test_loader, dev = train_one(cfg, args.data, args.epochs, device)
            acc, ratio = uncertainty_quality(model, test_loader, dev)
            mlflow.log_params(cfg)
            mlflow.log_metrics({"best_val_acc": best_val, "test_acc": acc,
                                "entropy_ratio_wrong_correct": ratio})
            torch.save({"state_dict": model.state_dict(), "classes": classes, "args": cfg},
                       f"results/model_{name}.pt")
            print(f"   val {best_val:.3f} | test {acc:.3f} | entropy ratio {ratio:.2f}x")

    # --- 2) deep ensemble ----------------------------------------------------
    # train N resnet18 with different seeds, average their softmax on test
    print(f"\n=== Deep ensemble ({args.ensemble_size} models) ===")
    import torch.nn.functional as F
    members, classes_ref, test_loader_ref = [], None, None
    with mlflow.start_run(run_name=f"deep_ensemble_{args.ensemble_size}"):
        for s in range(args.ensemble_size):
            cfg = {**base, "backbone": "resnet18", "dropout": 0.5, "seed": 100 + s}
            print(f"  member {s+1}/{args.ensemble_size} (seed {cfg['seed']})")
            model, classes, _, test_loader, dev = train_one(cfg, args.data, args.epochs, device)
            members.append(model); classes_ref = classes; test_loader_ref = test_loader

        # ensemble inference: average softmax across members
        preds, labels, ents, correct = [], [], [], []
        for x, y in test_loader_ref:
            x = x.to(device)
            with torch.no_grad():
                p = torch.stack([F.softmax(m.eval()(x), 1) for m in members], 0).mean(0)
            pred = p.argmax(1).cpu()
            ent = -(p * (p + 1e-12).log()).sum(1).cpu()
            preds.append(pred); labels.append(y); ents.append(ent)
            correct.append(pred == y)
        correct = torch.cat(correct); ent = torch.cat(ents)
        acc = correct.float().mean().item()
        ratio = ent[~correct].mean().item() / max(ent[correct].mean().item(), 1e-9)
        mlflow.log_params({"backbone": "resnet18", "method": "deep_ensemble",
                           "ensemble_size": args.ensemble_size})
        mlflow.log_metrics({"test_acc": acc, "entropy_ratio_wrong_correct": ratio})
        print(f"  ensemble -> test {acc:.3f} | entropy ratio {ratio:.2f}x")

    print("\nDone. Compare everything in the MLflow UI (sort by test_acc or entropy_ratio).")


if __name__ == "__main__":
    main()
