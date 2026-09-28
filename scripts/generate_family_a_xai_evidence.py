from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from capstone_ai.data import build_disease_loaders, build_stress_loaders, build_water_loaders
from capstone_ai.models import build_model
from capstone_ai import xai


PROJECT_DATA = Path("Dataset/PROJECT_DATA")
OUTPUT_DIR = Path("runs_hybrid/xai_evidence")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def disease_models(metadata):
    paths = (
        Path("runs_efficientnet/family_a_disease/best.pt"),
        Path("runs_hybrid/family_a_disease_resnet18/best.pt"),
        Path("runs_hybrid/family_a_disease_mobilenet_v2/best.pt"),
    )
    return [xai.load_model(path, "disease", metadata, backbone, DEVICE) for path, backbone in zip(paths, xai.DISEASE_BACKBONES)]


def stress_models(metadata):
    paths = (
        Path("runs_hybrid/family_a_stress_efficientnet_v2_s/best.pt"),
        Path("runs_hybrid/family_a_stress_resnet18/best.pt"),
        Path("runs_hybrid/family_a_stress_mobilenet_v2/best.pt"),
    )
    return [xai.load_model(path, "stress", metadata, backbone, DEVICE) for path, backbone in zip(paths, xai.DISEASE_BACKBONES)]


@torch.inference_mode()
def first_correct_disease(loaders):
    models = disease_models(loaders.metadata)
    weights = xai.DISEASE_WEIGHTS.to(DEVICE)
    dataset = loaders.validation.dataset
    for image_path, label in dataset.samples:
        _, image = xai.prepare_image(image_path, normalize=True)
        probabilities = torch.stack([torch.softmax(model(image.to(DEVICE)), dim=1) for model in models])
        prediction = int((probabilities * weights[:, None, None]).sum(dim=0)[0].argmax())
        if prediction == label:
            return image_path, label
    raise RuntimeError("No correct disease validation prediction was found.")


@torch.inference_mode()
def first_correct_stress(loaders):
    models = stress_models(loaders.metadata)
    weights = xai.STRESS_WEIGHTS.to(DEVICE)
    dataset = loaders.validation.dataset
    for _, row in dataset.dataframe.iterrows():
        rgb_path, thermal_path = dataset._resolve(row["rgb_path"]), dataset._resolve(row["thermal_path"])
        _, rgb = xai.prepare_image(rgb_path, normalize=True)
        _, thermal = xai.prepare_image(thermal_path, normalize=True)
        probabilities = torch.stack([torch.softmax(model(rgb.to(DEVICE), thermal.to(DEVICE)), dim=1) for model in models])
        prediction = int((probabilities * weights[:, None, None]).sum(dim=0)[0].argmax())
        label = dataset.class_to_id[row["class"]]
        if prediction == label:
            return rgb_path, thermal_path, str(row["class"])
    raise RuntimeError("No correct stress validation prediction was found.")


@torch.inference_mode()
def lowest_error_water_row(loaders):
    checkpoint_path = Path("runs_hybrid/family_a_water_baseline/best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = build_model("a", "water", checkpoint["metadata"], 224, backbone="baseline", pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = loaders.validation.dataset
    predictions = model(torch.as_tensor(dataset.features, device=DEVICE)).cpu()
    errors = (predictions - torch.as_tensor(dataset.targets)).abs().mean(dim=1)
    return int(errors.argmin())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    disease_loaders = build_disease_loaders(PROJECT_DATA, 224, 1, 0, imagenet_normalize=True)
    stress_loaders = build_stress_loaders(PROJECT_DATA, 224, 1, 0, imagenet_normalize=True)
    water_loaders = build_water_loaders(PROJECT_DATA, 256, 0)

    disease_path, disease_label = first_correct_disease(disease_loaders)
    disease_output = OUTPUT_DIR / "disease"
    disease_result = xai.explain_disease(SimpleNamespace(
        image=disease_path,
        efficientnet_checkpoint=Path("runs_efficientnet/family_a_disease/best.pt"),
        resnet_checkpoint=Path("runs_hybrid/family_a_disease_resnet18/best.pt"),
        mobilenet_checkpoint=Path("runs_hybrid/family_a_disease_mobilenet_v2/best.pt"),
        project_data=PROJECT_DATA,
        output_dir=disease_output,
    ), DEVICE)

    rgb_path, thermal_path, stress_label = first_correct_stress(stress_loaders)
    stress_output = OUTPUT_DIR / "stress"
    stress_result = xai.explain_stress(SimpleNamespace(
        rgb=rgb_path,
        thermal=thermal_path,
        efficientnet_checkpoint=Path("runs_hybrid/family_a_stress_efficientnet_v2_s/best.pt"),
        resnet_checkpoint=Path("runs_hybrid/family_a_stress_resnet18/best.pt"),
        mobilenet_checkpoint=Path("runs_hybrid/family_a_stress_mobilenet_v2/best.pt"),
        project_data=PROJECT_DATA,
        output_dir=stress_output,
    ), DEVICE)

    water_index = lowest_error_water_row(water_loaders)
    water_frame = water_loaders.validation.dataset.df.iloc[[water_index]].copy()
    water_input = OUTPUT_DIR / "water_representative_low_error.csv"
    water_frame.to_csv(water_input, index=False)
    water_result = xai.explain_water(SimpleNamespace(
        input_csv=water_input,
        checkpoint=Path("runs_hybrid/family_a_water_baseline/best.pt"),
        target="eta",
        steps=64,
    ), DEVICE)

    summary = {
        "split": "validation only; no held-out test inputs used",
        "device": str(DEVICE),
        "disease": {"expected_label_id": disease_label, "source_image": str(disease_path), **disease_result},
        "water_stress": {"expected_class": stress_label, "rgb_source": str(rgb_path), "thermal_source": str(thermal_path), **stress_result},
        "water": {"selection": "lowest mean standardized absolute error on validation", "validation_row_index": water_index, "input_csv": str(water_input), **water_result},
        "interpretation_guardrail": "Attribution highlights model sensitivity, not a causal diagnosis or agronomic mechanism.",
    }
    (OUTPUT_DIR / "xai_evidence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
