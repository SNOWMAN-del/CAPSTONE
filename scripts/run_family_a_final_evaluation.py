from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from capstone_ai.calibrate import collect_logits
from capstone_ai.data import build_disease_loaders, build_stress_loaders
from capstone_ai.evaluate_water import evaluate as evaluate_water
from capstone_ai.metrics import classification_calibration
from capstone_ai.models import build_model
from capstone_ai.data import build_water_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Run the one-time Family A held-out evaluation using a frozen configuration.")
    parser.add_argument("--config", type=Path, default=Path("configs/family_a_final.json"))
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--disease-batch-size", type=int, default=32)
    parser.add_argument("--stress-batch-size", type=int, default=16)
    parser.add_argument("--water-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs_hybrid/family_a_final_test_metrics.json"))
    return parser.parse_args()


def load_members(task, entry, metadata, image_size, device):
    models = []
    for checkpoint_path, backbone in zip(entry["checkpoints"], entry["backbones"]):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model("a", task, metadata, image_size, backbone=backbone, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    return models


def evaluate_image(task, entry, loaders, image_size, device):
    weights = torch.tensor(entry["weights"], device=device)
    models = load_members(task, entry, loaders.metadata, image_size, device)
    logits, labels, members = collect_logits(models, loaders.test, task, weights, device)
    raw = torch.softmax(logits, dim=1)
    calibrated = torch.softmax(logits / float(entry["temperature"]), dim=1)
    return {
        "accuracy": float(raw.argmax(dim=1).eq(labels).float().mean()),
        "calibration_before": classification_calibration(raw, labels),
        "calibration_after": classification_calibration(calibrated, labels),
        "mean_ensemble_disagreement": float(members.var(dim=0).mean(dim=1).mean()),
        "temperature_from_validation": entry["temperature"],
        "samples": int(labels.numel()),
    }


def evaluate_selected_water(entry, project_data, batch_size, device):
    checkpoint = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
    stats = checkpoint["dataset_stats"]
    loaders = build_water_loaders(project_data, batch_size, 0)
    model = build_model("a", "water", checkpoint["metadata"], 224, backbone="baseline", pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return evaluate_water(model, loaders.test, stats, device)


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_for_one_time_held_out_evaluation":
        raise ValueError("Configuration is not marked frozen for held-out evaluation.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disease_loaders = build_disease_loaders(args.project_data, args.image_size, args.disease_batch_size, args.num_workers, imagenet_normalize=True)
    stress_loaders = build_stress_loaders(args.project_data, args.image_size, args.stress_batch_size, args.num_workers, imagenet_normalize=True)
    result = {
        "configuration": str(args.config),
        "device": str(device),
        "protocol": "One-time held-out evaluation. No model, weights, temperature, or irrigation-policy selection is performed here.",
        "disease": evaluate_image("disease", config["disease"], disease_loaders, args.image_size, device),
        "water_stress": evaluate_image("stress", config["stress"], stress_loaders, args.image_size, device),
        "water": evaluate_selected_water(config["water"], args.project_data, args.water_batch_size, device),
        "irrigation_policy": config["irrigation_policy"],
        "ood": "Not evaluated: provide a documented external OOD dataset to capstone_ai.ood.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
