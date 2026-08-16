"""
model.py - the classifier.

A pretrained ResNet encoder + a small classification head with dropout.
The dropout is the whole point of the uncertainty story: at inference we keep
it ACTIVE and run the same image N times (MC Dropout). The spread of those N
predictions is the model's uncertainty.

  mean of N passes  -> prediction
  variance of N passes -> uncertainty
"""

import torch
import torch.nn as nn
from torchvision import models


class MRIClassifier(nn.Module):
    def __init__(self, num_classes=4, backbone="resnet18",
                 dropout=0.5, pretrained=True, freeze_encoder=False):
        super().__init__()

        # -- encoder (pretrained on ImageNet) --------------------------------
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        in_features = net.fc.in_features
        net.fc = nn.Identity()          # strip the ImageNet classifier
        self.encoder = net

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # -- classification head with dropout --------------------------------
        # This dropout is what MC Dropout samples from at inference time.
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))

    def set_encoder_trainable(self, trainable: bool):
        """Unfreeze/freeze the encoder later for progressive fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = trainable


def enable_mc_dropout(model):
    """Put the model in eval mode BUT keep dropout layers active.

    Normal .eval() turns dropout off. For MC Dropout we want everything in eval
    (so BatchNorm uses running stats) EXCEPT dropout, which must stay stochastic.
    Call this before sampling uncertainty.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model


if __name__ == "__main__":
    # smoke test: forward a fake batch, and check MC Dropout gives varying outputs
    model = MRIClassifier(num_classes=4, backbone="resnet18", pretrained=False)
    x = torch.randn(4, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        a, b = model(x), model(x)
    same = torch.allclose(a, b)
    print("eval() -> deterministic:", same, "(should be True)")

    enable_mc_dropout(model)
    with torch.no_grad():
        a, b = model(x), model(x)
    same = torch.allclose(a, b)
    print("mc_dropout -> deterministic:", same, "(should be False -> uncertainty works)")
    print("output shape:", a.shape)
