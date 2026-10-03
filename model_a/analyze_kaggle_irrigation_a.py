"""Validation-only uncertainty and permutation XAI for Family A irrigation benchmark."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


DATA = Path("kaggle_irrigation_data/irrigation_prediction.csv")
MODEL = Path("model_a/kaggle_irrigation_runs/family_a_random_forest/model.pkl")
OUTPUT = Path("model_a/kaggle_irrigation_runs/family_a_random_forest/evidence")


def calibration(probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray, bins: int = 15) -> dict:
    indices = np.searchsorted(classes, labels)
    confidence = probabilities.max(axis=1)
    predicted = classes[probabilities.argmax(axis=1)]
    correct = predicted == labels
    ece = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1 else (confidence <= upper))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), indices] = 1
    return {
        "samples": int(len(labels)),
        "accuracy": float(correct.mean()),
        "nll": float(log_loss(labels, probabilities, labels=classes)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": float(ece),
        "mean_predictive_entropy": float(np.mean(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1)), axis=1))),
    }


def main() -> None:
    if not DATA.is_file() or not MODEL.is_file():
        raise FileNotFoundError("Train Family A Kaggle irrigation model before generating evidence.")
    frame = pd.read_csv(DATA)
    x, y = frame.drop(columns=["Irrigation_Need"]), frame["Irrigation_Need"].astype(str)
    _, x_remaining, _, y_remaining = train_test_split(x, y, test_size=0.30, stratify=y, random_state=42)
    x_validation, _, y_validation, _ = train_test_split(x_remaining, y_remaining, test_size=0.50, stratify=y_remaining, random_state=42)
    with MODEL.open("rb") as file:
        model = pickle.load(file)
    probabilities = model.predict_proba(x_validation)
    classes = model.classes_
    baseline_accuracy = float((model.predict(x_validation) == y_validation.to_numpy()).mean())
    rng = np.random.default_rng(42)
    importances = []
    for column in x_validation.columns:
        shuffled = x_validation.copy()
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
        shuffled_accuracy = float((model.predict(shuffled) == y_validation.to_numpy()).mean())
        importances.append({"feature": column, "accuracy_drop": baseline_accuracy - shuffled_accuracy})
    importances.sort(key=lambda item: item["accuracy_drop"], reverse=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(importances).to_csv(OUTPUT / "permutation_importance_validation.csv", index=False)
    result = {
        "scope": "validation_only; locked test split was not used",
        "uncertainty_method": "Random Forest vote-derived class probabilities",
        "calibration": calibration(probabilities, y_validation.to_numpy(), classes),
        "permutation_importance": importances,
    }
    (OUTPUT / "uncertainty_and_xai.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"calibration": result["calibration"], "top_features": importances[:10]}, indent=2))


if __name__ == "__main__":
    main()
