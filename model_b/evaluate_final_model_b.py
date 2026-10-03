"""One locked held-out test evaluation for the finalized classical Model B."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from capstone_ai.data import build_disease_loaders, build_stress_loaders, build_water_loaders
from capstone_ai.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the one locked final Model B test evaluation.")
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--runs-dir", default="runs_classical_b")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--confirm-final-evaluation", action="store_true", help="Required because this reads reserved test splits.")
    return parser.parse_args()


def feature_encoder(device: torch.device) -> torch.nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    return model.to(device).eval()


@torch.inference_mode()
def features(loader, task: str, model: torch.nn.Module, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    vectors, labels = [], []
    for batch in loader:
        if task == "disease":
            vector = model(batch["image"].to(device))
        else:
            vector = torch.cat((model(batch["rgb"].to(device)), model(batch["thermal"].to(device))), dim=1)
        vectors.append(vector.cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(vectors), np.concatenate(labels)


def image_result(task: str, args: argparse.Namespace, runs_dir: Path, encoder: torch.nn.Module, device: torch.device) -> dict:
    with (runs_dir / f"family_b_{task}_classical" / "model.pkl").open("rb") as file:
        classifier = pickle.load(file)
    if task == "disease":
        bundle = build_disease_loaders(args.project_data, args.image_size, args.batch_size, args.num_workers, True)
    else:
        bundle = build_stress_loaders(args.project_data, args.image_size, args.batch_size, args.num_workers, True)
    x, y = features(bundle.test, task, encoder, device)
    predicted = classifier.predict(x)
    return {"accuracy": float(accuracy_score(y, predicted)), "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)), "samples": int(y.size)}


def water_result(args: argparse.Namespace, runs_dir: Path) -> dict:
    with (runs_dir / "water_model_selection" / "selected_model.pkl").open("rb") as file:
        saved = pickle.load(file)
    bundle = build_water_loaders(args.project_data, args.batch_size, args.num_workers)
    train, test = bundle.train.dataset, bundle.test.dataset
    prediction = saved["model"].predict(test.features)
    original_prediction = prediction * train.stats["target_std"] + train.stats["target_mean"]
    original_target = test.targets * train.stats["target_std"] + train.stats["target_mean"]
    return {"selected_model": saved["model_name"], **regression_metrics(original_prediction, original_target)}


def main() -> None:
    args = parse_args()
    if not args.confirm_final_evaluation:
        raise SystemExit("Refusing to open reserved test splits. Re-run with --confirm-final-evaluation when ready.")
    runs_dir = Path(args.runs_dir)
    output_path = runs_dir / "final_model_b_test_metrics.json"
    if output_path.exists():
        raise SystemExit(f"Final test report already exists: {output_path}. It will not be overwritten.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = feature_encoder(device)
    result = {
        "evaluation_type": "locked_held_out_test",
        "image_feature_extraction_device": str(device),
        "disease": image_result("disease", args, runs_dir, encoder, device),
        "stress": image_result("stress", args, runs_dir, encoder, device),
        "water": water_result(args, runs_dir),
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
