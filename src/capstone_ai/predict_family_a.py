from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .irrigation import IrrigationPolicy, irrigation_advisory
from .models import build_model


DISEASE_BACKBONES = ("efficientnet_v2_s", "resnet18", "mobilenet_v2")
DISEASE_WEIGHTS = (0.5, 0.1, 0.4)
STRESS_WEIGHTS = (0.6, 0.3, 0.1)


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
    stress.add_argument("--efficientnet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_stress_efficientnet_v2_s/best.pt"))
    stress.add_argument("--resnet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_stress_resnet18/best.pt"))
    stress.add_argument("--mobilenet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_stress_mobilenet_v2/best.pt"))
    stress.add_argument("--project-data", type=Path, default=Path("Dataset/PROJECT_DATA"))

    water = modes.add_parser("water", help="Predict next-day ETa and Ks from one 41-column CSV row.")
    water.add_argument("--input-csv", type=Path, required=True)
    water.add_argument("--model", choices=("random_forest", "mlp"), default="random_forest")
    water.add_argument("--checkpoint", type=Path, default=Path("runs_hybrid/family_a_water_baseline/best.pt"))
    water.add_argument("--random-forest-model", type=Path, default=Path("runs_hybrid/family_a_water_random_forest/model.pkl"))
    water.add_argument("--mc-samples", type=int, default=30, help="Stochastic passes for MLP predictive uncertainty.")
    water.add_argument("--rainfall-mm", type=float, default=0.0)
    water.add_argument("--irrigation-config", type=Path, help="Optional JSON object matching IrrigationPolicy fields.")

    irrigation = modes.add_parser("irrigation", help="Predict irrigation need from Kaggle irrigation feature rows.")
    irrigation.add_argument("--input-csv", type=Path, required=True)
    irrigation.add_argument(
        "--model",
        type=Path,
        default=Path("model_a/kaggle_irrigation_runs/family_a_random_forest/model.pkl"),
    )
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
    checkpoint_paths = (args.efficientnet_checkpoint, args.resnet_checkpoint, args.mobilenet_checkpoint)
    models = [
        load_model(path, "stress", {"num_classes": len(class_names), "input_channels": 6}, backbone, device)[0]
        for path, backbone in zip(checkpoint_paths, DISEASE_BACKBONES)
    ]
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    rgb = transform(Image.open(args.rgb).convert("RGB")).unsqueeze(0).to(device)
    thermal = transform(Image.open(args.thermal).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        member_probabilities = torch.stack([torch.softmax(model(rgb, thermal), dim=1) for model in models])
        probabilities = (member_probabilities * torch.tensor(STRESS_WEIGHTS, device=device)[:, None, None]).sum(0)[0]
    confidence, index = probabilities.max(dim=0)
    return {
        "mode": "stress",
        "ensemble_weights": dict(zip(DISEASE_BACKBONES, STRESS_WEIGHTS)),
        "prediction": {"class_name": class_names[int(index)], "confidence": round(float(confidence), 6)},
    }


def predict_water(args, device: torch.device):
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")
    if args.model == "random_forest":
        if not args.random_forest_model.is_file():
            raise FileNotFoundError(f"Random Forest model not found: {args.random_forest_model}")
        with args.random_forest_model.open("rb") as handle:
            saved = pickle.load(handle)
        metadata = saved["metadata"]
        stats = saved["dataset_stats"]
    else:
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
    target_mean = torch.as_tensor(stats["target_mean"], dtype=torch.float32)
    target_std = torch.as_tensor(stats["target_std"], dtype=torch.float32)
    normalized = (values - mean) / std
    if args.model == "random_forest":
        standardized_prediction = torch.as_tensor(saved["model"].predict(normalized.numpy())[0], dtype=torch.float32)
        prediction = standardized_prediction * target_std + target_mean
        uncertainty = None
    else:
        values = normalized.to(device)
        model = build_model("a", "water", metadata, 224, backbone="baseline", pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        if args.mc_samples < 2:
            raise ValueError("--mc-samples must be at least 2.")
        # Keep batch normalization fixed but enable dropout for Monte-Carlo uncertainty.
        model.eval()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
        with torch.inference_mode():
            standardized_samples = torch.stack([model(values)[0].cpu() for _ in range(args.mc_samples)])
        predictions = standardized_samples * target_std + target_mean
        prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0, unbiased=True)
    policy = IrrigationPolicy()
    if args.irrigation_config:
        policy = IrrigationPolicy(**json.loads(args.irrigation_config.read_text(encoding="utf-8")))
    result = {
        "mode": "water",
        "model": args.model,
        "prediction": {"next_day_ETa": round(float(prediction[0]), 6), "next_day_Ks": round(float(prediction[1]), 6)},
        "irrigation_advisory": irrigation_advisory(float(prediction[0]), float(prediction[1]), args.rainfall_mm, policy),
    }
    if uncertainty is not None:
        result["predictive_uncertainty_std"] = {"next_day_ETa": round(float(uncertainty[0]), 6), "next_day_Ks": round(float(uncertainty[1]), 6)}
        result["uncertainty_method"] = {"name": "MC dropout", "samples": args.mc_samples, "scope": "epistemic approximation"}
    return result


def predict_irrigation(args):
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    with args.model.open("rb") as handle:
        model = pickle.load(handle)
    frame = pd.read_csv(args.input_csv)
    if frame.empty:
        raise ValueError("The irrigation input CSV must contain at least one row.")
    if "Irrigation_Need" in frame.columns:
        frame = frame.drop(columns=["Irrigation_Need"])
    predictions = model.predict(frame)
    result = {"mode": "irrigation", "model": "Kaggle Random Forest", "predictions": [str(value) for value in predictions]}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)
        classes = model.classes_
        result["confidence"] = [
            {str(classes[int(index)]): round(float(values[int(index)]), 6)}
            for values, index in zip(probabilities, probabilities.argmax(axis=1))
        ]
    return result


def main() -> None:
    args = parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "disease":
        result = predict_disease(args, device)
    elif args.mode == "stress":
        result = predict_stress(args, device)
    elif args.mode == "irrigation":
        result = predict_irrigation(args)
    else:
        result = predict_water(args, device)
    result["device"] = str(device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
