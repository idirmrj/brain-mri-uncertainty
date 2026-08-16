"""
MC Dropout uncertainty estimation.

Idea (recap): keep dropout ACTIVE at inference, run the same image N times.
The N slightly-different softmax outputs are averaged; their spread is the
uncertainty.

Per image we return:
  - pred        : predicted class (argmax of the mean softmax)
  - confidence  : the mean softmax probability of that class
  - entropy     : predictive entropy of the mean softmax (total uncertainty)
  - variance    : mean variance of the softmax across the N passes
                  (how much the model disagrees with itself -> model uncertainty)

Entropy is the main uncertainty score used downstream (calibration, rejection).
"""

import torch
import torch.nn.functional as F

from model import MRIClassifier, enable_mc_dropout


@torch.no_grad()
def mc_dropout_predict(model, x, n_passes=30):

    enable_mc_dropout(model)
    probs = torch.stack([F.softmax(model(x), dim=1)
                         for _ in range(n_passes)], dim=0)

    mean_probs = probs.mean(0) 
    var_probs = probs.var(0)

    pred = mean_probs.argmax(1)
    confidence = mean_probs.max(1).values
    entropy = -(mean_probs * (mean_probs + 1e-12).log()).sum(1)
    variance = var_probs.mean(1)

    return {"mean_probs": mean_probs, "pred": pred, "confidence": confidence,
            "entropy": entropy, "variance": variance}


def load_model(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    a = ckpt.get("args", {})
    model = MRIClassifier(
        num_classes=len(ckpt["classes"]),
        backbone=a.get("backbone", "resnet18"),
        dropout=a.get("dropout", 0.5),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["classes"], device


@torch.no_grad()
def evaluate_with_uncertainty(model, loader, device, n_passes=30):
    preds, labels, conf, ent, var = [], [], [], [], []
    for x, y in loader:
        x = x.to(device)
        out = mc_dropout_predict(model, x, n_passes=n_passes)
        preds.append(out["pred"].cpu()); labels.append(y)
        conf.append(out["confidence"].cpu())
        ent.append(out["entropy"].cpu()); var.append(out["variance"].cpu())
    preds = torch.cat(preds); labels = torch.cat(labels)
    return {
        "preds": preds, "labels": labels,
        "confidence": torch.cat(conf), "entropy": torch.cat(ent),
        "variance": torch.cat(var), "correct": (preds == labels),
    }


if __name__ == "__main__":
    model = MRIClassifier(num_classes=4, pretrained=False)
    x = torch.randn(5, 3, 224, 224)
    out = mc_dropout_predict(model, x, n_passes=20)
    for k, v in out.items():
        print(f"{k:12s} {tuple(v.shape)}")
    print("\nentropy (per image):", [round(e, 3) for e in out['entropy'].tolist()])
    print("variance > 0 (MC working):", bool((out['variance'] > 0).all()))
