from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import torch

from .data import build_disease_loaders, seed_everything
from .models import build_model


BACKBONES = ("efficientnet_v2_s", "resnet18", "mobilenet_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a three-CNN disease ensemble using validation data.")
    parser.add_argument("--efficientnet-checkpoint", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--mobilenet-checkpoint", type=Path, required=True)
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ensemble_disease"))
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_member(checkpoint_path: Path, backbone: str, metadata: dict, image_size: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model("a", "disease", metadata, image_size, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.inference_mode()
def collect_probabilities(models, loader, device: torch.device):
    probabilities, labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        probabilities.append(torch.stack([torch.softmax(model(images), dim=1) for model in models]).cpu())
        labels.append(batch["label"])
    return torch.cat(probabilities, dim=1), torch.cat(labels)


def accuracy(probabilities: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> float:
    predictions = (probabilities * weights[:, None, None]).sum(dim=0).argmax(dim=1)
    return (predictions == labels).float().mean().item()


def candidate_weights():
    for efficientnet, resnet in product(range(0, 11), repeat=2):
        mobilenet = 10 - efficientnet - resnet
        if efficientnet >= 1 and resnet >= 1 and mobilenet >= 1:
            yield torch.tensor([efficientnet, resnet, mobilenet], dtype=torch.float32) / 10


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_disease_loaders(
        args.project_data, args.image_size, args.batch_size, args.num_workers, imagenet_normalize=True
    )
    checkpoint_paths = (args.efficientnet_checkpoint, args.resnet_checkpoint, args.mobilenet_checkpoint)
    models = [
        load_member(path, backbone, loaders.metadata, args.image_size, device)
        for path, backbone in zip(checkpoint_paths, BACKBONES)
    ]
    validation_probabilities, validation_labels = collect_probabilities(models, loaders.validation, device)
    weights = max(candidate_weights(), key=lambda values: accuracy(validation_probabilities, validation_labels, values))
    result = {
        "selection_split": "validation",
        "backbones": BACKBONES,
        "weights": dict(zip(BACKBONES, weights.tolist())),
        "validation_accuracy": accuracy(validation_probabilities, validation_labels, weights),
        "checkpoints": [str(path) for path in checkpoint_paths],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_ensemble_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.evaluate_test:
        test_probabilities, test_labels = collect_probabilities(models, loaders.test, device)
        result["test_accuracy"] = accuracy(test_probabilities, test_labels, weights)
        (args.output_dir / "test_ensemble_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
