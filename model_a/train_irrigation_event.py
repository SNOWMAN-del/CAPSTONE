"""Leakage-safe retrospective irrigation-event experiment for Family A."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


FEATURES = [
    "Year", "DOY", "Treatment", "WB_Precip", "WB_Root Zone Field Capacity",
    "W_AirTemp_Avg", "W_AirTemp_Max", "W_AirTemp_Min", "W_Vap_Press_Avg",
    "W_RH_Max", "W_RH_Min", "W_DlySolRad", "W_WindRun_Tot", "W_DlyRain",
    "W_SoilTemp_5cm", "W_SoilTemp_15cm", "W_ETr", "W_ETo",
    "SWC_0 - 15_Mean", "SWC_30_Mean", "SWC_60_Mean", "SWC_90_Mean",
    "SWC_120_Mean", "SWC_150_Mean", "SWC_200_Mean",
]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a chronological irrigation-event classifier from the master water table.")
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="model_a/irrigation_event_runs")
    return parser.parse_args()


def metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    prediction = (probabilities >= threshold).astype(int)
    result = {
        "samples": int(y.size), "positive_events": int(y.sum()), "positive_rate": float(y.mean()),
        "threshold": threshold, "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y)) == 2:
        result["average_precision"] = float(average_precision_score(y, probabilities))
        result["roc_auc"] = float(roc_auc_score(y, probabilities))
    return result


def main() -> None:
    config = args()
    source = Path(config.project_data) / "04_water_need" / "METADATA" / "master_water_dataset.csv"
    data = pd.read_csv(source, low_memory=False)
    data["Date"] = pd.to_datetime(data["Date"])
    data["WB_Irrigation"] = pd.to_numeric(data["WB_Irrigation"], errors="coerce")
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values(["Treatment", "Date"]).copy()
    # The target is the actual irrigation event at the next recorded observation
    # for the same treatment; no future value is used as an input.
    data["Target_Next_Observation_Date"] = data.groupby("Treatment")["Date"].shift(-1)
    data["Target_Next_Irrigation"] = data.groupby("Treatment")["WB_Irrigation"].shift(-1)
    data = data.dropna(subset=["Target_Next_Irrigation"]).copy()
    data["irrigation_event"] = (data["Target_Next_Irrigation"] > 0).astype(int)
    train = data[data["Year"] <= 2013]
    validation = data[data["Year"] == 2014]
    held_out = data[data["Year"] >= 2015]
    if train["irrigation_event"].nunique() < 2 or validation["irrigation_event"].nunique() < 2:
        raise ValueError("Training or validation period lacks both irrigation classes.")
    model = RandomForestClassifier(
        n_estimators=config.trees, class_weight="balanced", min_samples_leaf=2,
        random_state=config.seed, n_jobs=-1,
    ).fit(train[FEATURES], train["irrigation_event"])
    validation_probability = model.predict_proba(validation[FEATURES])[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.05)
    threshold = float(max(thresholds, key=lambda value: f1_score(validation["irrigation_event"], validation_probability >= value, zero_division=0)))
    result = {
        "task": "retrospective_next_observation_irrigation_event_prediction",
        "protocol": "train=2008-2013; validation=2014; 2015-2016 not evaluated by this command",
        "target_definition": "1 when recorded irrigation at the next observation for the same treatment is greater than zero",
        "features": FEATURES,
        "validation": metrics(validation["irrigation_event"].to_numpy(), validation_probability, threshold),
        "held_out_period": {"rows": int(len(held_out)), "positive_events": int(held_out["irrigation_event"].sum()), "evaluated": False},
    }
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "model.pkl").open("wb") as file:
        pickle.dump({"model": model, "features": FEATURES, "threshold": threshold}, file)
    (output / "validation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
