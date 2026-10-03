"""Select a fast classical water model using the validation split only."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from capstone_ai.data import build_water_loaders, seed_everything
from capstone_ai.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare simple Family B water models on validation data only.")
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs_classical_b/water_model_selection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    bundle = build_water_loaders(args.project_data, batch_size=64, num_workers=0)
    train, validation = bundle.train.dataset, bundle.validation.dataset
    candidates = {
        "extra_trees": ExtraTreesRegressor(n_estimators=args.trees, max_features=0.8, random_state=args.seed, n_jobs=-1),
        "random_forest": RandomForestRegressor(n_estimators=args.trees, max_features=0.8, random_state=args.seed, n_jobs=-1),
        "ridge": Ridge(alpha=10.0),
        "rbf_svr_c1": MultiOutputRegressor(SVR(kernel="rbf", C=1.0, epsilon=0.05)),
        "rbf_svr_c10": MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.05)),
    }
    comparisons = {}
    fitted_models = {}
    for name, candidate in candidates.items():
        model = clone(candidate).fit(train.features, train.targets)
        normalized_prediction = model.predict(validation.features)
        normalized_mse = float(np.mean((normalized_prediction - validation.targets) ** 2))
        original_prediction = normalized_prediction * train.stats["target_std"] + train.stats["target_mean"]
        original_target = validation.targets * train.stats["target_std"] + train.stats["target_mean"]
        comparisons[name] = {
            "selection_normalized_mse": normalized_mse,
            "validation": regression_metrics(original_prediction, original_target),
        }
        fitted_models[name] = model
    best_name = min(comparisons, key=lambda name: comparisons[name]["selection_normalized_mse"])
    result = {
        "selection_rule": "lowest validation MSE on targets standardized with train-split statistics",
        "selected_model": best_name,
        "comparisons": comparisons,
        "test_evaluated": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "selected_model.pkl").open("wb") as file:
        pickle.dump({"model": fitted_models[best_name], "model_name": best_name, "dataset_stats": train.stats, "metadata": bundle.metadata}, file)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
