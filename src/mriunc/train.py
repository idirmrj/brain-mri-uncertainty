"""
train.py - training loop for the MRI classifier.

  - runs on GPU automatically if available
  - logs params / metrics / the best model to MLflow
  - selects the checkpoint with the best VALIDATION accuracy (not the last epoch)
  - fixed seed for reproducibility

Run:  python src/mriunc/train.py --data "path/to/data" --epochs 15
Then: mlflow ui   (to browse your runs at http://localhost:5000)
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


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/best_model.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader, _, classes = build_dataloaders(
        args.data, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed)
    print("Classes:", classes)

    model = MRIClassifier(num_classes=len(classes), backbone=args.backbone,
                          dropout=args.dropout, pretrained=True,
                          freeze_encoder=args.freeze_encoder).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("brain-mri-uncertainty")
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_param("device", device)
        mlflow.log_param("classes", classes)

        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss, "val_acc": va_acc}, step=epoch)
            print(f"epoch {epoch:2d} | "
                  f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.3f}")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save({"state_dict": model.state_dict(),
                            "classes": classes, "args": vars(args)}, args.out)
                mlflow.log_metric("best_val_acc", best_val_acc, step=epoch)

        mlflow.log_artifact(args.out)
        print(f"\nBest val acc: {best_val_acc:.3f}  ->  saved to {args.out}")


if __name__ == "__main__":
    main()
