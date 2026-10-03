"""Validation-only Platt calibration and uncertainty evidence for Model B images."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path: sys.path.insert(0, str(SRC_DIRECTORY))

from train_classical_model_b import encoder, features, image_loaders


def arguments():
    parser = argparse.ArgumentParser(description="Calibrate frozen-MobileNet + SVM Model B probabilities without using the test split.")
    parser.add_argument("--task", choices=["disease", "stress"], required=True)
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="runs_classical_b")
    return parser.parse_args()


def calibration(probabilities, labels, classes, bins=15):
    confidence = probabilities.max(axis=1); prediction = classes[probabilities.argmax(axis=1)]; correct = prediction == labels
    ece = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins; mask = (confidence >= lower) & ((confidence < upper) if upper < 1 else (confidence <= upper))
        if mask.any(): ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    indices = np.searchsorted(classes, labels); one_hot = np.zeros_like(probabilities); one_hot[np.arange(len(labels)), indices] = 1
    return {"samples": int(len(labels)), "accuracy": float(correct.mean()), "nll": float(log_loss(labels, probabilities, labels=classes)), "brier": float(np.mean(np.sum((probabilities-one_hot)**2, axis=1))), "ece": float(ece), "mean_predictive_entropy": float(np.mean(-np.sum(probabilities*np.log(np.clip(probabilities,1e-12,1)),axis=1)))}


def main():
    args = arguments(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for frozen MobileNet feature extraction.", flush=True)
    bundle = image_loaders(args); extractor = encoder(device)
    print("Extracting deterministic training features...", flush=True)
    train_x, train_y = features(bundle.train, args.task, extractor, device)
    print("Extracting validation features...", flush=True)
    validation_x, validation_y = features(bundle.validation, args.task, extractor, device)
    print("Fitting three-fold Platt calibration on training features...", flush=True)
    base = make_pipeline(StandardScaler(), LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=10000))
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(train_x, train_y)
    probability = model.predict_proba(validation_x); result = {"task": args.task, "scope": "calibrated from training features with three-fold cross-validation; evaluated on validation only", "device": str(device), "uncertainty_method": "Platt-calibrated Linear SVM probabilities", "validation": calibration(probability, validation_y, model.classes_)}
    output = Path(args.output_dir) / f"family_b_{args.task}_classical" / "evidence"; output.mkdir(parents=True, exist_ok=True)
    with (output / "calibrated_model.pkl").open("wb") as file: pickle.dump(model,file)
    (output / "calibration.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
