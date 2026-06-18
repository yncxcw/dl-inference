from __future__ import annotations

import torch
from torch import nn


class TinyAlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def create_model() -> nn.Module:
    torch.manual_seed(7)
    model = TinyAlexNet().eval()
    for name, param in model.named_parameters():
        values = torch.arange(param.numel(), dtype=torch.float32).reshape_as(param)
        param.data.copy_((values % 17 - 8) / 17.0)
    return model
