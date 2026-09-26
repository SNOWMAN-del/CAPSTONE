from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .models import build_model


BACKBONES = ("efficientnet_v2_s", "resnet18", "mobilenet_v2")
DEFAULT_WEIGHTS = (0.5, 0.1, 0.4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one medicinal leaf disease with the three-CNN ensemble.")
    parser.add_argument("image", type=Path, help="Path to one RGB leaf image (.jpg, .jpeg, or .png).")
    parser.add_argument("--efficientnet-checkpoint", type=Path, default=Path("runs_efficientnet/family_a_disease/best.pt"))
    parser.add_argument("--resnet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_disease_resnet18/best.pt"))
    parser.add_argument("--mobilenet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_disease_mobilenet_v2/best.pt"))
    parser.add_argument("--project-data", type=Path, default=Path("Dataset/PROJECT_DATA"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--weights", nargs=3, type=float, default=DEFAULT_WEIGHTS, metavar=("EFFICIENTNET", "RESNET", "MOBILENET"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def class_names(project_data: Path) -> dict[int, str]:
    mapping_file = project_data / "MASTER" / "class_mapping.csv"
    with mapping_file.open("r", encoding="utf-8") as handle:
        return {int(row["class_id"]): row["class_name"] for row in csv.DictReader(handle)}


def load_model(checkpoint_path: Path, backbone: str, image_size: int, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        "a", "disease", {"num_classes": 21, "input_channels": 3}, image_size, backbone=backbone, pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.top_k < 1 or args.top_k > 21:
        raise ValueError("--top-k must be between 1 and 21.")

    weights = torch.tensor(args.weights, dtype=torch.float32)
    if torch.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("All ensemble weights must be non-negative and sum to more than zero.")
    weights /= weights.sum()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_paths = (args.efficientnet_checkpoint, args.resnet_checkpoint, args.mobilenet_checkpoint)
    models = [
        load_model(path, backbone, args.image_size, device) for path, backbone in zip(checkpoint_paths, BACKBONES)
    ]
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        member_probabilities = torch.stack([torch.softmax(model(image), dim=1) for model in models])
        probabilities = (member_probabilities * weights.to(device)[:, None, None]).sum(dim=0).squeeze(0)
    names = class_names(args.project_data)
    values, indices = probabilities.topk(args.top_k)
    predictions = [
        {"class_id": int(index), "class_name": names[int(index)], "confidence": round(float(value), 6)}
        for value, index in zip(values.cpu(), indices.cpu())
    ]
    print(
        json.dumps(
            {
                "image": str(args.image),
                "device": str(device),
                "ensemble_weights": dict(zip(BACKBONES, weights.tolist())),
                "prediction": predictions[0],
                "top_predictions": predictions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
