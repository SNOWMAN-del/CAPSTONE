"""Fast classical Family B: MobileNet features + Linear SVM; Extra Trees for water."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

# Allow this standalone script to be run directly from F:\CAPSTONE without
# requiring the user to set PYTHONPATH manually.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from capstone_ai.data import build_disease_loaders, build_stress_loaders, build_water_loaders, seed_everything
from capstone_ai.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight classical Family B.")
    parser.add_argument("--task", choices=["disease", "stress", "water"], required=True)
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs_classical_b")
    parser.add_argument("--torch-home", default=".torch_cache")
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def image_loaders(args: argparse.Namespace):
    if args.task == "disease":
        bundle = build_disease_loaders(args.project_data, args.image_size, args.batch_size, args.num_workers, True)
        bundle.train.dataset.transform = bundle.validation.dataset.transform
    else:
        bundle = build_stress_loaders(args.project_data, args.image_size, args.batch_size, args.num_workers, True)
        bundle.train.dataset.train = False
    return bundle


def encoder(device: torch.device) -> torch.nn.Module:
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


def classification_metrics(model, x: np.ndarray, y: np.ndarray) -> dict:
    predicted = model.predict(x)
    return {"accuracy": float(accuracy_score(y, predicted)), "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)), "samples": int(y.size)}


def train_images(args: argparse.Namespace, run_dir: Path, device: torch.device) -> dict:
    bundle = image_loaders(args)
    extractor = encoder(device)
    train_x, train_y = features(bundle.train, args.task, extractor, device)
    valid_x, valid_y = features(bundle.validation, args.task, extractor, device)
    model = make_pipeline(StandardScaler(), LinearSVC(C=args.svm_c, class_weight="balanced", dual="auto", max_iter=10000))
    model.fit(train_x, train_y)
    result = {"method": "frozen_mobilenet_v2_embeddings_plus_linear_svm", "validation": classification_metrics(model, valid_x, valid_y), "feature_dimensions": int(train_x.shape[1]), "train_samples": int(train_y.size)}
    if args.evaluate_test:
        test_x, test_y = features(bundle.test, args.task, extractor, device)
        result["test"] = classification_metrics(model, test_x, test_y)
    with (run_dir / "model.pkl").open("wb") as file:
        pickle.dump(model, file)
    return result


def train_water(args: argparse.Namespace, run_dir: Path) -> dict:
    bundle = build_water_loaders(args.project_data, args.batch_size, args.num_workers)
    train, valid = bundle.train.dataset, bundle.validation.dataset
    model = ExtraTreesRegressor(n_estimators=args.trees, max_features=0.8, random_state=args.seed, n_jobs=-1)
    model.fit(train.features, train.targets)
    def score(dataset):
        prediction = model.predict(dataset.features)
        return regression_metrics(prediction * train.stats["target_std"] + train.stats["target_mean"], dataset.targets * train.stats["target_std"] + train.stats["target_mean"])
    result = {"method": "extra_trees_regressor", "validation": score(valid), "feature_dimensions": int(train.features.shape[1]), "train_samples": int(len(train))}
    if args.evaluate_test:
        result["test"] = score(bundle.test.dataset)
    with (run_dir / "model.pkl").open("wb") as file:
        pickle.dump({"model": model, "dataset_stats": train.stats, "metadata": bundle.metadata}, file)
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(Path(args.torch_home).resolve()))
    seed_everything(args.seed)
    run_dir = Path(args.output_dir) / f"family_b_{args.task}_classical"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MobileNet feature extraction device: {device}")
    if device.type == "cpu" and args.task in {"disease", "stress"}:
        print("WARNING: CUDA is unavailable to PyTorch; MobileNet features will be extracted on CPU.")
        print("The Linear SVM itself is a CPU algorithm even when CUDA is available.")
    result = train_water(args, run_dir) if args.task == "water" else train_images(args, run_dir, device)
    result.update({"task": args.task, "device": str(device), "test_evaluated": args.evaluate_test})
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
