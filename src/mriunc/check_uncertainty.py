"""
check_uncertainty.py - sanity check on the core hypothesis of the project:
"the model is more uncertain when it is wrong."

Loads the trained model, runs MC Dropout over the test set, and compares the
uncertainty (entropy) of CORRECT vs WRONG predictions. If wrong predictions are
clearly more uncertain, the uncertainty signal is useful -> rejection will work.

Run:
  python src/mriunc/check_uncertainty.py --data "path/to/data" --ckpt results/best_model.pt
"""

import argparse
import torch

from data import build_dataloaders
from uncertainty import load_model, evaluate_with_uncertainty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="results/best_model.pt")
    ap.add_argument("--n_passes", type=int, default=30)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    model, classes, device = load_model(args.ckpt)
    print("Device:", device, "| classes:", classes)

    # we evaluate on the untouched Testing set
    _, _, test_loader, _ = build_dataloaders(
        args.data, batch_size=32, num_workers=args.num_workers)

    print(f"Running MC Dropout ({args.n_passes} passes) over the test set...")
    r = evaluate_with_uncertainty(model, test_loader, device, n_passes=args.n_passes)

    acc = r["correct"].float().mean().item()
    ent_correct = r["entropy"][r["correct"]].mean().item()
    ent_wrong = r["entropy"][~r["correct"]].mean().item()
    conf_correct = r["confidence"][r["correct"]].mean().item()
    conf_wrong = r["confidence"][~r["correct"]].mean().item()

    print(f"\nTest accuracy: {acc:.3f}  ({(~r['correct']).sum().item()} errors "
          f"out of {len(r['correct'])})")
    print("\n--- The key check: uncertainty of correct vs wrong ---")
    print(f"mean entropy  | correct: {ent_correct:.3f}   wrong: {ent_wrong:.3f}"
          f"   ratio: {ent_wrong/max(ent_correct,1e-9):.2f}x")
    print(f"mean confidence| correct: {conf_correct:.3f}   wrong: {conf_wrong:.3f}")

    if ent_wrong > ent_correct:
        print("\n[OK] Errors are MORE uncertain than correct predictions.")
        print("     -> the uncertainty signal is useful; rejection will work.")
    else:
        print("\n[!] Errors are NOT more uncertain. Investigate before rejection "
              "(model may be badly calibrated / overconfident).")


if __name__ == "__main__":
    main()
