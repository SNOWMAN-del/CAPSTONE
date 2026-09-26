from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .models import build_model


DISEASE_BACKBONES = ("efficientnet_v2_s", "resnet18", "mobilenet_v2")
DISEASE_WEIGHTS = (0.5, 0.1, 0.4)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run one of the three Family A input modes.")
    modes = root.add_subparsers(dest="mode", required=True)

    disease = modes.add_parser("disease", help="Predict disease from one RGB leaf image.")
    disease.add_argument("image", type=Path)
    disease.add_argument("--efficientnet-checkpoint", type=Path, default=Path("runs_efficientnet/family_a_disease/best.pt"))
    disease.add_argument("--resnet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_disease_resnet18/best.pt"))
    disease.add_argument("--mobilenet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_disease_mobilenet_v2/best.pt"))
    disease.add_argument("--project-data", type=Path, default=Path("Dataset/PROJECT_DATA"))
    disease.add_argument("--top-k", type=int, default=3)

    stress = modes.add_parser("stress", help="Predict healthy/stressed from a synchronized RGB and thermal pair.")
    stress.add_argument("--rgb", type=Path, required=True)
    stress.add_argument("--thermal", type=Path, required=True)
    stress.add_argument("--checkpoint", type=Path, default=Path("runs_hybrid/family_a_stress_fusion/best.pt"))
    stress.add_argument("--project-data", type=Path, default=Path("Dataset/PROJECT_DATA"))

    water = modes.add_parser("water", help="Predict next-day ETa and Ks from one 41-column CSV row.")
    water.add_argument("--input-csv", type=Path, required=True)
    water.add_argument("--checkpoint", type=Path, default=Path("runs_hybrid/family_a_water_baseline/best.pt"))
    return root


def load_model(checkpoint_path: Path, task: str, metadata: dict, backbone: str, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model("a", task, metadata, 224, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def disease_names(project_data: Path) -> dict[int, str]:
    with (project_data / "MASTER" / "class_mapping.csv").open("r", encoding="utf-8") as handle:
        return {int(row["class_id"]): row["class_name"] for row in csv.DictReader(handle)}


def predict_disease(args, device: torch.device):
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    paths = (args.efficientnet_checkpoint, args.resnet_checkpoint, args.mobilenet_checkpoint)
    models = [
        load_model(path, "disease", {"num_classes": 21, "input_channels": 3}, backbone, device)[0]
        for path, backbone in zip(paths, DISEASE_BACKBONES)
    ]
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        member_probabilities = torch.stack([torch.softmax(model(image), dim=1) for model in models])
        probabilities = (member_probabilities * torch.tensor(DISEASE_WEIGHTS, device=device)[:, None, None]).sum(0)[0]
    values, indices = probabilities.topk(args.top_k)
    names = disease_names(args.project_data)
    return {
        "mode": "disease",
        "prediction": {"class_name": names[int(indices[0])], "confidence": round(float(values[0]), 6)},
        "top_predictions": [
            {"class_name": names[int(index)], "confidence": round(float(value), 6)}
            for value, index in zip(values.cpu(), indices.cpu())
        ],
    }


def predict_stress(args, device: torch.device):
    for image_path in (args.rgb, args.thermal):
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
    manifest = pd.read_csv(args.project_data / "02_multimodal_stress" / "METADATA" / "split_manifest.csv")
    class_names = sorted(manifest["class"].unique())
    model, _ = load_model(args.checkpoint, "stress", {"num_classes": len(class_names), "input_channels": 6}, "fusion", device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    rgb = transform(Image.open(args.rgb).convert("RGB")).unsqueeze(0).to(device)
    thermal = transform(Image.open(args.thermal).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(rgb, thermal), dim=1)[0]
    confidence, index = probabilities.max(dim=0)
    return {"mode": "stress", "prediction": {"class_name": class_names[int(index)], "confidence": round(float(confidence), 6)}}


def predict_water(args, device: torch.device):
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    stats = checkpoint.get("dataset_stats")
    if stats is None:
        raise ValueError("This water checkpoint lacks normalization statistics. Retrain it with the current trainer.")
    row = pd.read_csv(args.input_csv)
    if len(row) != 1:
        raise ValueError("The water input CSV must contain exactly one row.")
    feature_columns = metadata["feature_columns"]
    missing = [column for column in feature_columns if column not in row.columns]
    if missing:
        raise ValueError(f"Water input CSV is missing required columns: {missing}")
    features = row[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype="float32")
    mean = torch.as_tensor(stats["feature_mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["feature_std"], dtype=torch.float32)
    values = torch.as_tensor(features, dtype=torch.float32)
    values = torch.where(torch.isnan(values), mean, values)
    values = ((values - mean) / std).to(device)
    model = build_model("a", "water", metadata, 224, backbone="baseline", pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        standardized = model(values)[0].cpu()
    target_mean = torch.as_tensor(stats["target_mean"], dtype=torch.float32)
    target_std = torch.as_tensor(stats["target_std"], dtype=torch.float32)
    prediction = standardized * target_std + target_mean
    return {"mode": "water", "prediction": {"next_day_ETa": round(float(prediction[0]), 6), "next_day_Ks": round(float(prediction[1]), 6)}}


def main() -> None:
    args = parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "disease":
        result = predict_disease(args, device)
    elif args.mode == "stress":
        result = predict_stress(args, device)
    else:
        result = predict_water(args, device)
    result["device"] = str(device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
