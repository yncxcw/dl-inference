"""A tiny AlexNet-style CNN used as an end-to-end DLI example."""

from __future__ import annotations

import torch
from torch import nn


class TinyAlexNet(nn.Module):
    """Two conv/ReLU/max-pool stages followed by a two-layer classifier.

    Expects float32 input of shape ``[N, 3, 32, 32]`` (two 2x2 pools leave 8x8
    feature maps, matching the first ``Linear`` layer) and returns ``[N, 4]``.
    """

    def __init__(self) -> None:
        """Builds the feature extractor and classifier."""
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
        """Runs the network.

        Args:
            x: Input of shape ``[N, 3, 32, 32]``.

        Returns:
            Logits of shape ``[N, 4]``.
        """
        return self.classifier(self.features(x))


def create_model() -> nn.Module:
    """Creates an eval-mode model with deterministic parameters.

    Returns:
        A ``TinyAlexNet`` whose weights are a fixed arithmetic pattern, so exports
        and outputs are reproducible without a checkpoint.
    """
    torch.manual_seed(7)
    model = TinyAlexNet().eval()
    for name, param in model.named_parameters():
        values = torch.arange(param.numel(), dtype=torch.float32).reshape_as(param)
        param.data.copy_((values % 17 - 8) / 17.0)
    return model
