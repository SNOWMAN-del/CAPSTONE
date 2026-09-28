from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import build_water_loaders
from .metrics import regression_metrics
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Family A water MLP in original ETa/Ks units.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs_hybrid/family_a_water_baseline/best.pt"))
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("runs_hybrid/family_a_water_baseline/validation_metrics.json"))
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, loader, stats, device):
    predictions, targets = [], []
    for batch in loader:
        predictions.append(model(batch["features"].to(device)).cpu().numpy())
        targets.append(batch["targets"].numpy())
    mean, std = np.asarray(stats["target_mean"]), np.asarray(stats["target_std"])
    return regression_metrics(np.concatenate(predictions) * std + mean, np.concatenate(targets) * std + mean)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = checkpoint.get("dataset_stats")
    if not stats:
        raise ValueError("Checkpoint lacks training statistics; retrain with the current trainer.")
    loaders = build_water_loaders(args.project_data, args.batch_size, 0)
    model = build_model("a", "water", checkpoint["metadata"], 224, backbone="baseline", pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    result = {"model": "FamilyAWaterRegressor", "selection_split": "validation", "validation": evaluate(model, loaders.validation, stats, device)}
    if args.evaluate_test:
        result["test"] = evaluate(model, loaders.test, stats, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
