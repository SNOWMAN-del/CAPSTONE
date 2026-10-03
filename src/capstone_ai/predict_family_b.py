from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


IMAGE_SIZE = 224
IMAGENET_NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained Family B predictors.")
    modes = parser.add_subparsers(dest="mode", required=True)

    disease = modes.add_parser("disease", help="Predict a disease class from one RGB leaf image.")
    disease.add_argument("image", type=Path)
    disease.add_argument("--model", type=Path, default=Path("runs_classical_b/family_b_disease_classical/model.pkl"))
    disease.add_argument("--project-data", type=Path, default=Path("Dataset/PROJECT_DATA"))

    stress = modes.add_parser("stress", help="Predict plant stress from synchronized RGB and thermal images.")
    stress.add_argument("--rgb", type=Path, required=True)
    stress.add_argument("--thermal", type=Path, required=True)
    stress.add_argument("--model", type=Path, default=Path("runs_classical_b/family_b_stress_classical/model.pkl"))
    stress.add_argument("--project-data", type=Path, default=Path("Dataset/PROJECT_DATA"))

    water = modes.add_parser("water", help="Predict next-day ETa and Ks from a one-row 41-feature CSV.")
    water.add_argument("--input-csv", type=Path, required=True)
    water.add_argument("--model", type=Path, default=Path("runs_classical_b/water_model_selection/selected_model.pkl"))

    irrigation = modes.add_parser("irrigation", help="Predict irrigation need for rows from the Kaggle irrigation feature schema.")
    irrigation.add_argument("--input-csv", type=Path, required=True)
    irrigation.add_argument("--model", type=Path, default=Path("model_b/kaggle_irrigation_runs/family_b_stacking/model.pkl"))
    return parser.parse_args()


def load_pickle(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def encoder(device: torch.device) -> torch.nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    return model.to(device).eval()


def image_tensor(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), IMAGENET_NORMALIZE])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)


def disease_names(project_data: Path) -> dict[int, str]:
    with (project_data / "MASTER" / "class_mapping.csv").open("r", encoding="utf-8") as handle:
        return {int(row["class_id"]): row["class_name"] for row in csv.DictReader(handle)}


@torch.inference_mode()
def predict_disease(args: argparse.Namespace, device: torch.device) -> dict:
    classifier = load_pickle(args.model)
    vector = encoder(device)(image_tensor(args.image).to(device)).cpu().numpy()
    class_id = int(classifier.predict(vector)[0])
    return {"mode": "disease", "model": "MobileNetV2 embeddings + Linear SVM", "prediction": disease_names(args.project_data)[class_id]}


@torch.inference_mode()
def predict_stress(args: argparse.Namespace, device: torch.device) -> dict:
    classifier = load_pickle(args.model)
    feature_encoder = encoder(device)
    rgb = feature_encoder(image_tensor(args.rgb).to(device))
    thermal = feature_encoder(image_tensor(args.thermal).to(device))
    vector = torch.cat((rgb, thermal), dim=1).cpu().numpy()
    class_id = int(classifier.predict(vector)[0])
    manifest = pd.read_csv(args.project_data / "02_multimodal_stress" / "METADATA" / "split_manifest.csv")
    names = sorted(manifest["class"].unique())
    return {"mode": "stress", "model": "paired MobileNetV2 embeddings + Linear SVM", "prediction": names[class_id]}


def predict_water(args: argparse.Namespace) -> dict:
    saved = load_pickle(args.model)
    row = pd.read_csv(args.input_csv)
    if len(row) != 1:
        raise ValueError("The water input CSV must contain exactly one row.")
    metadata, stats = saved["metadata"], saved["dataset_stats"]
    features = metadata["feature_columns"]
    missing = [feature for feature in features if feature not in row.columns]
    if missing:
        raise ValueError(f"Water input CSV is missing required columns: {missing}")
    values = row[features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    mean = np.asarray(stats["feature_mean"], dtype=np.float32)
    std = np.asarray(stats["feature_std"], dtype=np.float32)
    normalized = (np.where(np.isnan(values), mean, values) - mean) / std
    prediction = saved["model"].predict(normalized)[0]
    original_units = prediction * np.asarray(stats["target_std"]) + np.asarray(stats["target_mean"])
    return {
        "mode": "water",
        "model": saved.get("model_name", type(saved["model"]).__name__),
        "prediction": {"next_day_ETa": round(float(original_units[0]), 6), "next_day_Ks": round(float(original_units[1]), 6)},
    }


def engineer_irrigation_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns
    if len(numeric):
        result["engineered_numeric_mean"] = result[numeric].mean(axis=1)
        result["engineered_numeric_std"] = result[numeric].std(axis=1).fillna(0)
    return result


def predict_irrigation(args: argparse.Namespace) -> dict:
    model = load_pickle(args.model)
    frame = pd.read_csv(args.input_csv)
    if frame.empty:
        raise ValueError("The irrigation input CSV must contain at least one row.")
    predictions = model.predict(engineer_irrigation_features(frame))
    return {"mode": "irrigation", "model": "feature-engineered stacking classifier", "predictions": [str(value) for value in predictions]}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "disease":
        result = predict_disease(args, device)
    elif args.mode == "stress":
        result = predict_stress(args, device)
    elif args.mode == "water":
        result = predict_water(args)
    else:
        result = predict_irrigation(args)
    result["device"] = str(device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
