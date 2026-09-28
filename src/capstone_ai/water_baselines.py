from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .data import build_water_loaders
from .metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a chronological-split Random Forest water baseline.")
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs_hybrid/family_a_water_random_forest"))
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def dataset_arrays(dataset):
    return dataset.features, dataset.targets, dataset.stats


def unscale(values: np.ndarray, stats: dict) -> np.ndarray:
    return values * np.asarray(stats["target_std"]) + np.asarray(stats["target_mean"])


def main() -> None:
    args = parse_args()
    loaders = build_water_loaders(args.project_data, batch_size=256, num_workers=0)
    x_train, y_train, stats = dataset_arrays(loaders.train.dataset)
    x_validation, y_validation, _ = dataset_arrays(loaders.validation.dataset)
    model = RandomForestRegressor(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    result = {
        "model": "RandomForestRegressor",
        "selection_split": "validation",
        "parameters": {"trees": args.trees, "max_depth": args.max_depth, "min_samples_leaf": args.min_samples_leaf},
        "validation": regression_metrics(unscale(model.predict(x_validation), stats), unscale(y_validation, stats)),
        "feature_importance": [
            {"feature": feature, "importance": float(value)}
            for feature, value in sorted(zip(loaders.metadata["feature_columns"], model.feature_importances_), key=lambda item: item[1], reverse=True)
        ],
    }
    if args.evaluate_test:
        x_test, y_test, _ = dataset_arrays(loaders.test.dataset)
        result["test"] = regression_metrics(unscale(model.predict(x_test), stats), unscale(y_test, stats))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    with (args.output_dir / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "metadata": loaders.metadata, "dataset_stats": stats}, handle)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
