from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .models import build_model


DISEASE_BACKBONES = ("efficientnet_v2_s", "resnet18", "mobilenet_v2")
DISEASE_WEIGHTS = torch.tensor((0.5, 0.1, 0.4), dtype=torch.float32)
STRESS_WEIGHTS = torch.tensor((0.6, 0.3, 0.1), dtype=torch.float32)
IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Grad-CAM explanations for Family A image predictions.")
    modes = parser.add_subparsers(dest="mode", required=True)

    disease = modes.add_parser("disease", help="Explain the three-CNN disease ensemble.")
    disease.add_argument("image", type=Path)
    disease.add_argument("--efficientnet-checkpoint", type=Path, default=Path("runs_efficientnet/family_a_disease/best.pt"))
    disease.add_argument("--resnet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_disease_resnet18/best.pt"))
    disease.add_argument("--mobilenet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_disease_mobilenet_v2/best.pt"))
    disease.add_argument("--project-data", type=Path, default=Path("Dataset/PROJECT_DATA"))

    stress = modes.add_parser("stress", help="Explain the three-CNN RGB/thermal stress ensemble.")
    stress.add_argument("--rgb", type=Path, required=True)
    stress.add_argument("--thermal", type=Path, required=True)
    stress.add_argument("--efficientnet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_stress_efficientnet_v2_s/best.pt"))
    stress.add_argument("--resnet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_stress_resnet18/best.pt"))
    stress.add_argument("--mobilenet-checkpoint", type=Path, default=Path("runs_hybrid/family_a_stress_mobilenet_v2/best.pt"))
    stress.add_argument("--project-data", type=Path, default=Path("Dataset/PROJECT_DATA"))

    water = modes.add_parser("water", help="Explain water prediction with Integrated Gradients feature attribution.")
    water.add_argument("--input-csv", type=Path, required=True)
    water.add_argument("--checkpoint", type=Path, default=Path("runs_hybrid/family_a_water_baseline/best.pt"))
    water.add_argument("--target", choices=("eta", "ks"), default="eta")
    water.add_argument("--steps", type=int, default=64)

    parser.add_argument("--output-dir", type=Path, default=Path("runs_hybrid/xai"))
    return parser.parse_args()


def load_model(checkpoint_path: Path, task: str, metadata: dict, backbone: str, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model("a", task, metadata, IMAGE_SIZE, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def prepare_image(path: Path, normalize: bool) -> tuple[Image.Image, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    transform_steps = [transforms.ToTensor()]
    if normalize:
        transform_steps.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return image, transforms.Compose(transform_steps)(image).unsqueeze(0)


def gradcam(model, inputs, layers: dict[str, torch.nn.Module], target_class: int) -> dict[str, np.ndarray]:
    activations: dict[str, torch.Tensor] = {}
    handles = []
    for name, layer in layers.items():
        def hook(_, __, output, layer_name=name):
            activations[layer_name] = output
            output.retain_grad()
        handles.append(layer.register_forward_hook(hook))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
        logits[0, target_class].backward()
        cams = {}
        for name, activation in activations.items():
            gradients = activation.grad
            channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
            heatmap = torch.relu((channel_weights * activation).sum(dim=1, keepdim=True))
            heatmap = torch.nn.functional.interpolate(heatmap, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
            heatmap = heatmap[0, 0]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            cams[name] = heatmap.detach().cpu().numpy()
        return cams
    finally:
        for handle in handles:
            handle.remove()


def overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    base = np.asarray(image, dtype=np.float32)
    red = 255 * heatmap
    green = 255 * (1 - np.abs(2 * heatmap - 1))
    blue = 255 * (1 - heatmap)
    colored = np.stack((red, green, blue), axis=2)
    return Image.fromarray(np.clip(0.55 * base + 0.45 * colored, 0, 255).astype(np.uint8))


def disease_class_names(project_data: Path) -> dict[int, str]:
    with (project_data / "MASTER" / "class_mapping.csv").open("r", encoding="utf-8") as handle:
        return {int(row["class_id"]): row["class_name"] for row in csv.DictReader(handle)}


def explain_disease(args, device: torch.device) -> dict:
    original, image = prepare_image(args.image, normalize=True)
    image = image.to(device)
    checkpoint_paths = (args.efficientnet_checkpoint, args.resnet_checkpoint, args.mobilenet_checkpoint)
    models = [
        load_model(path, "disease", {"num_classes": 21, "input_channels": 3}, backbone, device)
        for path, backbone in zip(checkpoint_paths, DISEASE_BACKBONES)
    ]
    with torch.inference_mode():
        member_probabilities = torch.stack([torch.softmax(model(image), dim=1) for model in models])
        probabilities = (member_probabilities * DISEASE_WEIGHTS.to(device)[:, None, None]).sum(0)[0]
    confidence, target = probabilities.max(dim=0)
    target_class = int(target)
    layers = (
        {"efficientnet": models[0].network.features[-1]},
        {"resnet": models[1].network.layer4[-1].conv2},
        {"mobilenet": models[2].network.features[-1]},
    )
    heatmaps = [gradcam(model, image, layer, target_class)[name] for model, layer, name in zip(models, layers, ("efficientnet", "resnet", "mobilenet"))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, heatmap in zip(("efficientnet", "resnet", "mobilenet"), heatmaps):
        overlay(original, heatmap).save(args.output_dir / f"disease_{name}_gradcam.png")
    ensemble_heatmap = sum(weight * heatmap for weight, heatmap in zip(DISEASE_WEIGHTS.tolist(), heatmaps))
    overlay(original, ensemble_heatmap).save(args.output_dir / "disease_ensemble_gradcam.png")
    return {
        "mode": "disease",
        "prediction": disease_class_names(args.project_data)[target_class],
        "confidence": round(float(confidence), 6),
        "outputs": [str(args.output_dir / name) for name in (
            "disease_efficientnet_gradcam.png", "disease_resnet_gradcam.png", "disease_mobilenet_gradcam.png", "disease_ensemble_gradcam.png"
        )],
    }


def explain_stress(args, device: torch.device) -> dict:
    rgb_original, rgb = prepare_image(args.rgb, normalize=True)
    thermal_original, thermal = prepare_image(args.thermal, normalize=True)
    manifest = pd.read_csv(args.project_data / "02_multimodal_stress" / "METADATA" / "split_manifest.csv")
    class_names = sorted(manifest["class"].unique())
    rgb, thermal = rgb.to(device), thermal.to(device)
    checkpoint_paths = (args.efficientnet_checkpoint, args.resnet_checkpoint, args.mobilenet_checkpoint)
    models = [
        load_model(path, "stress", {"num_classes": len(class_names), "input_channels": 6}, backbone, device)
        for path, backbone in zip(checkpoint_paths, DISEASE_BACKBONES)
    ]
    with torch.inference_mode():
        member_probabilities = torch.stack([torch.softmax(model(rgb, thermal), dim=1) for model in models])
        probabilities = (member_probabilities * STRESS_WEIGHTS.to(device)[:, None, None]).sum(0)[0]
    confidence, target = probabilities.max(dim=0)
    member_heatmaps = []
    for model, backbone in zip(models, DISEASE_BACKBONES):
        if backbone == "efficientnet_v2_s":
            layers = {"rgb": model.rgb_encoder.network.features[-1], "thermal": model.thermal_encoder.network.features[-1]}
        elif backbone == "resnet18":
            layers = {"rgb": model.rgb_encoder.network.layer4[-1].conv2, "thermal": model.thermal_encoder.network.layer4[-1].conv2}
        else:
            layers = {"rgb": model.rgb_encoder.network.features[-1], "thermal": model.thermal_encoder.network.features[-1]}
        member_heatmaps.append(gradcam(model, (rgb, thermal), layers, int(target)))
    rgb_heatmap = sum(weight * maps["rgb"] for weight, maps in zip(STRESS_WEIGHTS.tolist(), member_heatmaps))
    thermal_heatmap = sum(weight * maps["thermal"] for weight, maps in zip(STRESS_WEIGHTS.tolist(), member_heatmaps))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = args.output_dir / "stress_ensemble_rgb_gradcam.png"
    thermal_path = args.output_dir / "stress_ensemble_thermal_gradcam.png"
    overlay(rgb_original, rgb_heatmap).save(rgb_path)
    overlay(thermal_original, thermal_heatmap).save(thermal_path)
    return {
        "mode": "stress",
        "prediction": class_names[int(target)],
        "confidence": round(float(confidence), 6),
        "ensemble_weights": dict(zip(DISEASE_BACKBONES, STRESS_WEIGHTS.tolist())),
        "outputs": [str(rgb_path), str(thermal_path)],
    }


def explain_water(args, device: torch.device) -> dict:
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")
    if args.steps < 2:
        raise ValueError("--steps must be at least 2.")
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
    raw_values = row[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    mean = torch.as_tensor(stats["feature_mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["feature_std"], dtype=torch.float32)
    values = torch.as_tensor(raw_values, dtype=torch.float32)[0]
    values = torch.where(torch.isnan(values), mean, values)
    normalized = ((values - mean) / std).to(device)
    model = build_model("a", "water", metadata, IMAGE_SIZE, backbone="baseline", pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    target_index = 0 if args.target == "eta" else 1
    baseline = torch.zeros_like(normalized)
    average_gradient = torch.zeros_like(normalized)
    for alpha in torch.linspace(0, 1, args.steps, device=device):
        interpolated = (baseline + alpha * (normalized - baseline)).unsqueeze(0).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(interpolated)[0, target_index].backward()
        average_gradient += interpolated.grad[0]
    attributions = (normalized - baseline) * average_gradient / args.steps
    order = torch.argsort(attributions.abs(), descending=True)
    top_features = [
        {
            "feature": feature_columns[int(index)],
            "attribution": round(float(attributions[int(index)].cpu()), 6),
        }
        for index in order[:10]
    ]
    return {
        "mode": "water",
        "target": "next_day_ETa" if args.target == "eta" else "next_day_Ks",
        "top_feature_attributions": top_features,
        "interpretation": "Positive values increase the selected prediction relative to the training-average baseline; negative values decrease it.",
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "disease":
        result = explain_disease(args, device)
    elif args.mode == "stress":
        result = explain_stress(args, device)
    else:
        result = explain_water(args, device)
    result["device"] = str(device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
