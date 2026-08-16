"""
rejection.py - selective prediction / risk-coverage analysis.

Once we know uncertainty is a useful signal (check_uncertainty), we USE it:
sort predictions by uncertainty, hand the most uncertain ones to a human, and
measure the error on the cases the model keeps.

Produces the risk-coverage curve (the portfolio's key figure):
  x = coverage  (fraction of cases the model handles itself)
  y = risk      (error rate on those cases)
As we reject more uncertain cases, coverage drops and risk should drop too.

Run:
  python src/mriunc/rejection.py --data "path/to/data" --ckpt results/best_model.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from data import build_dataloaders
from uncertainty import load_model, evaluate_with_uncertainty

ACCENT = "#7B2CBF"


def risk_coverage(correct, uncertainty):
    """Sort by ASCENDING uncertainty (most confident first). At each coverage
    level c, keep the c fraction most-confident cases and compute their error."""
    order = np.argsort(uncertainty)          # most confident first
    correct_sorted = correct[order]
    n = len(correct)
    coverages = np.arange(1, n + 1) / n
    # cumulative error over the kept (most-confident) prefix
    cum_errors = np.cumsum(~correct_sorted)
    risks = cum_errors / np.arange(1, n + 1)
    return coverages, risks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="results/best_model.pt")
    ap.add_argument("--n_passes", type=int, default=30)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out", default="results/risk_coverage.png")
    args = ap.parse_args()

    model, classes, device = load_model(args.ckpt)
    _, _, test_loader, _ = build_dataloaders(args.data, batch_size=32,
                                             num_workers=args.num_workers)

    print(f"Running MC Dropout ({args.n_passes} passes)...")
    r = evaluate_with_uncertainty(model, test_loader, device, n_passes=args.n_passes)
    correct = r["correct"].numpy()
    uncertainty = r["entropy"].numpy()        # use entropy as the reject score

    cov, risk = risk_coverage(correct, uncertainty)
    base_error = (~correct).mean()

    # headline numbers at a few rejection levels
    print(f"\nBase test error (no rejection): {base_error:.3f}")
    for rej in (0.05, 0.10, 0.15, 0.20):
        keep = 1 - rej
        idx = int(keep * len(cov)) - 1
        print(f"  reject {rej:.0%} most uncertain -> coverage {keep:.0%}, "
              f"error {risk[idx]:.3f}")

    # plot
    plt.figure(figsize=(6, 4.5))
    plt.plot(cov * 100, risk * 100, color=ACCENT, lw=2.2)
    plt.axhline(base_error * 100, ls="--", color="gray", lw=1,
                label=f"no rejection ({base_error*100:.1f}%)")
    plt.xlabel("Coverage (% of cases handled automatically)")
    plt.ylabel("Error rate (%) on handled cases")
    plt.title("Risk-Coverage: rejecting uncertain cases lowers error")
    plt.gca().invert_xaxis()      # high coverage (easy) on the left -> low on right
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
