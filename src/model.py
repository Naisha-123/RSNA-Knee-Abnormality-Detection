"""Backbone wrappers for twelve-label knee abnormality classification."""

from __future__ import annotations

from typing import Any


class KneeClassifier:
    """Factory-like wrapper that builds a torch classifier on first use."""

    def __new__(
        cls,
        backbone_name: str = "efficientnet_b0",
        num_labels: int = 12,
        pretrained: bool = False,
    ) -> Any:
        try:
            import torch.nn as neural_network
        except ImportError as error:
            raise ImportError("Building the model requires torch.") from error

        backbone = backbone_name.lower()
        if backbone == "efficientnet_b0":
            try:
                from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            except ImportError as error:
                raise ImportError("EfficientNet requires torchvision.") from error
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            network = efficientnet_b0(weights=weights)
            feature_count = network.classifier[1].in_features
            network.classifier[1] = neural_network.Linear(feature_count, num_labels)
            return network

        if backbone in {"dinov2_vits14", "dinov2_vitb14"}:
            try:
                import torch
            except ImportError as error:
                raise ImportError("DINOv2 requires torch.") from error
            if not pretrained:
                raise ValueError("DINOv2 loading requires pretrained=True.")
            network = torch.hub.load("facebookresearch/dinov2", backbone)
            feature_count = network.embed_dim
            return neural_network.Sequential(
                network,
                neural_network.LayerNorm(feature_count),
                neural_network.Linear(feature_count, num_labels),
            )

        raise ValueError(
            f"Unknown backbone {backbone_name!r}. Use 'efficientnet_b0', 'dinov2_vits14', or 'dinov2_vitb14'."
        )


def build_model(
    backbone_name: str = "efficientnet_b0",
    num_labels: int = 12,
    pretrained: bool = False,
) -> Any:
    """Build a twelve-output classifier with a local or Kaggle-compatible API."""
    return KneeClassifier(backbone_name, num_labels, pretrained)
