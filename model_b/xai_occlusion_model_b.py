"""Occlusion-sensitivity XAI for frozen MobileNet + calibrated SVM Model B."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from train_classical_model_b import encoder, features, image_loaders


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create validation-only occlusion XAI maps for Model B image classifiers.")
    parser.add_argument("--task", choices=["disease", "stress"], required=True)
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=7, help="Occlusion grid side length.")
    parser.add_argument("--output-dir", default="runs_classical_b")
    return parser.parse_args()


@torch.inference_mode()
def probability(model, extractor, task: str, first: torch.Tensor, second: torch.Tensor | None, device: torch.device) -> np.ndarray:
    if task == "disease":
        vectors = extractor(first.to(device)).cpu().numpy()
    else:
        vectors = torch.cat((extractor(first.to(device)), extractor(second.to(device))), dim=1).cpu().numpy()
    return model.predict_proba(vectors)


def occlusion_map(model, extractor, task: str, first: torch.Tensor, second: torch.Tensor | None, class_label: int, modality: str, grid: int, device: torch.device) -> tuple[float, np.ndarray]:
    base_probabilities = probability(model, extractor, task, first.unsqueeze(0), None if second is None else second.unsqueeze(0), device)[0]
    class_index = int(np.where(model.classes_ == class_label)[0][0])
    baseline = float(base_probabilities[class_index])
    _, height, width = first.shape
    patch_h, patch_w = height // grid, width // grid
    first_batch, second_batch = [], []
    for row in range(grid):
        for col in range(grid):
            changed_first, changed_second = first.clone(), None if second is None else second.clone()
            y0, y1 = row * patch_h, height if row == grid - 1 else (row + 1) * patch_h
            x0, x1 = col * patch_w, width if col == grid - 1 else (col + 1) * patch_w
            if modality == "rgb" or task == "disease": changed_first[:, y0:y1, x0:x1] = 0
            else: changed_second[:, y0:y1, x0:x1] = 0
            first_batch.append(changed_first)
            if changed_second is not None: second_batch.append(changed_second)
    probabilities = probability(model, extractor, task, torch.stack(first_batch), None if second is None else torch.stack(second_batch), device)
    return baseline, (baseline - probabilities[:, class_index]).reshape(grid, grid)


def overlay(image: torch.Tensor, scores: np.ndarray, path: Path) -> None:
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    pixels = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    base = Image.fromarray((pixels * 255).astype(np.uint8)).convert("RGBA")
    positive = np.maximum(scores, 0)
    scaled = positive / positive.max() if positive.max() > 0 else positive
    heat = Image.fromarray((scaled * 210).astype(np.uint8)).resize(base.size, Image.Resampling.BILINEAR)
    red = Image.new("RGBA", base.size, (255, 0, 0, 0)); red.putalpha(heat)
    Image.alpha_composite(base, red).convert("RGB").save(path)


def main() -> None:
    args = arguments(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evidence_dir = Path(args.output_dir) / f"family_b_{args.task}_classical" / "evidence"
    model_path = evidence_dir / "calibrated_model.pkl"
    if not model_path.is_file(): raise FileNotFoundError(f"Run calibration first: {model_path}")
    import pickle
    with model_path.open("rb") as file: model = pickle.load(file)
    print(f"Using {device}; selecting one correct validation prediction...", flush=True)
    bundle = image_loaders(args); extractor = encoder(device)
    x_validation, y_validation = features(bundle.validation, args.task, extractor, device)
    predicted = model.predict(x_validation); correct = np.flatnonzero(predicted == y_validation)
    if not len(correct): raise RuntimeError("No correct validation prediction available for XAI.")
    index = int(correct[0]); sample = bundle.validation.dataset[index]; label = int(sample["label"])
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result = {"task": args.task, "scope": "one correct validation sample; held-out test data were not used", "method": "occlusion sensitivity: positive score denotes the drop in calibrated class probability when a patch is replaced by the normalized-image baseline", "dataset_index": index, "class_id": label, "baseline_confidence": None, "maps": {}}
    if args.task == "disease":
        image = sample["image"]
        confidence, scores = occlusion_map(model, extractor, args.task, image, None, label, "rgb", args.grid_size, device)
        file = evidence_dir / "disease_occlusion.png"; overlay(image, scores, file)
        result["baseline_confidence"] = confidence; result["maps"]["rgb"] = {"file": str(file), "grid_scores": scores.tolist()}
    else:
        rgb, thermal = sample["rgb"], sample["thermal"]
        confidence, rgb_scores = occlusion_map(model, extractor, args.task, rgb, thermal, label, "rgb", args.grid_size, device)
        _, thermal_scores = occlusion_map(model, extractor, args.task, rgb, thermal, label, "thermal", args.grid_size, device)
        rgb_file, thermal_file = evidence_dir / "stress_occlusion_rgb.png", evidence_dir / "stress_occlusion_thermal.png"
        overlay(rgb, rgb_scores, rgb_file); overlay(thermal, thermal_scores, thermal_file)
        result["baseline_confidence"] = confidence; result["maps"] = {"rgb": {"file": str(rgb_file), "grid_scores": rgb_scores.tolist()}, "thermal": {"file": str(thermal_file), "grid_scores": thermal_scores.tolist()}}
    (evidence_dir / "occlusion_xai.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"task": args.task, "class_id": label, "baseline_confidence": result["baseline_confidence"], "files": {name: item["file"] for name, item in result["maps"].items()}}, indent=2))


if __name__ == "__main__": main()
