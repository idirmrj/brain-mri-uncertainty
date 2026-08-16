"""
gradcam.py - Grad-CAM heatmaps: where did the model look to decide?

Overlays a heatmap on the MRI showing which regions drove the prediction.
Two uses:
  - the hero visual for the portfolio (tumour lights up)
  - a sanity check: if it lights up on a border/artefact instead of the tumour,
    the model learned a shortcut, not the pathology.

Needs:  pip install grad-cam

Run:
  python src/mriunc/gradcam.py --data "path/to/data" --ckpt results/best_model.pt --n 6
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from data import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from uncertainty import load_model, mc_dropout_predict


def denormalize(tensor):
    """Undo ImageNet normalization -> displayable [0,1] image (H, W, 3)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="results/best_model.pt")
    ap.add_argument("--n", type=int, default=6, help="number of test images to show")
    ap.add_argument("--out", default="results/gradcam.png")
    args = ap.parse_args()

    model, classes, device = load_model(args.ckpt)
    model.eval()

    # target layer for a ResNet encoder = last conv block
    target_layers = [model.encoder.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # grab a few test images
    from torchvision import datasets
    _, eval_tf = build_transforms()
    ds = datasets.ImageFolder(Path(args.data) / "Testing", transform=eval_tf,
                              loader=lambda p: __import__("PIL.Image", fromlist=["Image"]).open(p).convert("RGB"))
    idxs = np.random.default_rng(0).choice(len(ds), size=args.n, replace=False)

    fig, axes = plt.subplots(2, args.n, figsize=(3 * args.n, 6))
    for col, i in enumerate(idxs):
        x, y = ds[i]
        xb = x.unsqueeze(0).to(device)

        # prediction + uncertainty for the caption
        out = mc_dropout_predict(model, xb, n_passes=20)
        pred = out["pred"].item(); conf = out["confidence"].item(); ent = out["entropy"].item()

        # Grad-CAM needs grads -> run outside no_grad, model in eval
        grayscale_cam = cam(input_tensor=xb)[0]           # (H, W) in [0,1]
        rgb = denormalize(x)
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        axes[0, col].imshow(rgb); axes[0, col].axis("off")
        axes[0, col].set_title(f"true: {classes[y]}", fontsize=9)
        axes[1, col].imshow(overlay); axes[1, col].axis("off")
        flag = "  [uncertain]" if ent > 0.6 else ""
        axes[1, col].set_title(f"pred: {classes[pred]} ({conf:.0%}){flag}", fontsize=9)

    axes[0, 0].set_ylabel("MRI", fontsize=11)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=11)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
