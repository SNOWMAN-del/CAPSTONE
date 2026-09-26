from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_V2_S_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
    efficientnet_v2_s,
    mobilenet_v2,
    resnet18,
)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.output_dim = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FamilyAImageClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels)
        self.head = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class FamilyAStressFusionClassifier(nn.Module):
    """Separate RGB and thermal CNN encoders with feature-level fusion."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.rgb_encoder = ConvEncoder(in_channels=3)
        self.thermal_encoder = ConvEncoder(in_channels=3)
        self.head = nn.Sequential(
            nn.Linear(self.rgb_encoder.output_dim + self.thermal_encoder.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.rgb_encoder(rgb), self.thermal_encoder(thermal)], dim=1)
        return self.head(features)


class FamilyAPretrainedCNNClassifier(nn.Module):
    """ImageNet-initialized CNN options for the Family A disease ensemble."""

    def __init__(self, num_classes: int, backbone: str, pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        if backbone == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.network = efficientnet_v2_s(weights=weights)
            in_features = self.network.classifier[-1].in_features
            self.network.classifier[-1] = nn.Linear(in_features, num_classes)
        elif backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.network = resnet18(weights=weights)
            self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
        elif backbone == "mobilenet_v2":
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.network = mobilenet_v2(weights=weights)
            in_features = self.network.classifier[-1].in_features
            self.network.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported pretrained CNN backbone: {backbone}")

    def set_backbone_trainable(self, trainable: bool) -> None:
        if self.backbone == "resnet18":
            parameters = (parameter for name, parameter in self.network.named_parameters() if not name.startswith("fc."))
        else:
            parameters = self.network.features.parameters()
        for parameter in parameters:
            parameter.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FamilyAWaterRegressor(nn.Module):
    def __init__(self, num_features: int, num_targets: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PatchAttentionClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, image_size: int, dim: int = 192):
        super().__init__()
        patch_size = 16
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=6,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position[:, : tokens.size(1) + 1]
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))


class FamilyBWaterRegressor(nn.Module):
    def __init__(self, num_features: int, num_targets: int = 2, dim: int = 128):
        super().__init__()
        self.value_embed = nn.Linear(1, dim)
        self.position = nn.Parameter(torch.zeros(1, num_features, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_embed(x.unsqueeze(-1)) + self.position
        encoded = self.encoder(tokens)
        return self.head(encoded.mean(dim=1))


def build_model(
    family: str,
    task: str,
    metadata: dict,
    image_size: int,
    backbone: str = "baseline",
    pretrained: bool = True,
) -> nn.Module:
    family = family.lower()
    if task in {"disease", "stress"}:
        in_channels = metadata["input_channels"]
        num_classes = metadata["num_classes"]
        if family == "a":
            if task == "disease" and backbone in {"efficientnet_v2_s", "resnet18", "mobilenet_v2"}:
                return FamilyAPretrainedCNNClassifier(
                    num_classes=num_classes, backbone=backbone, pretrained=pretrained
                )
            if task == "stress" and backbone == "fusion":
                return FamilyAStressFusionClassifier(num_classes=num_classes)
            return FamilyAImageClassifier(in_channels=in_channels, num_classes=num_classes)
        if family == "b":
            return PatchAttentionClassifier(in_channels=in_channels, num_classes=num_classes, image_size=image_size)
    if task == "water":
        if family == "a":
            return FamilyAWaterRegressor(num_features=metadata["num_features"], num_targets=metadata["num_targets"])
        if family == "b":
            return FamilyBWaterRegressor(num_features=metadata["num_features"], num_targets=metadata["num_targets"])
    raise ValueError(f"Unsupported family/task combination: family={family}, task={task}")
