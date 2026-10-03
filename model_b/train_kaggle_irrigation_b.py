"""Family B: engineered-feature stacking ensemble for the Kaggle irrigation benchmark."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def arguments():
    parser = argparse.ArgumentParser(description="Train Family B feature-engineered stacked ensemble on the public Kaggle irrigation benchmark.")
    parser.add_argument("--data", type=Path, default=Path("kaggle_irrigation_data/irrigation_prediction.csv"))
    parser.add_argument("--target", default="Irrigation_Need")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("model_b/kaggle_irrigation_runs/family_b_stacking"))
    return parser.parse_args()


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = result.select_dtypes(include="number").columns
    # Stable row-level agronomic summary features; no target or test information is used.
    if len(numeric):
        result["engineered_numeric_mean"] = result[numeric].mean(axis=1)
        result["engineered_numeric_std"] = result[numeric].std(axis=1).fillna(0)
    return result


def score(model, x, y):
    predicted = model.predict(x)
    return {"accuracy": float(accuracy_score(y, predicted)), "balanced_accuracy": float(balanced_accuracy_score(y, predicted)), "macro_f1": float(f1_score(y, predicted, average="macro")), "classification_report": classification_report(y, predicted, output_dict=True, zero_division=0)}


def main():
    args = arguments()
    if not args.data.is_file(): raise FileNotFoundError(f"Download the Kaggle CSV first; expected: {args.data}")
    frame = pd.read_csv(args.data)
    if args.target not in frame: raise ValueError(f"Target column {args.target!r} not found. Available columns: {frame.columns.tolist()}")
    x, y = engineer(frame.drop(columns=[args.target])), frame[args.target].astype(str)
    x_train, x_remaining, y_train, y_remaining = train_test_split(x, y, test_size=0.30, stratify=y, random_state=args.seed)
    x_validation, x_test, y_validation, y_test = train_test_split(x_remaining, y_remaining, test_size=0.50, stratify=y_remaining, random_state=args.seed)
    numeric = x.select_dtypes(include="number").columns.tolist(); categorical = [name for name in x.columns if name not in numeric]
    preprocess = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric), ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)])
    ensemble = StackingClassifier(estimators=[("extra_trees", ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=args.seed, n_jobs=-1)), ("random_forest", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=args.seed, n_jobs=-1)), ("gradient_boosting", GradientBoostingClassifier(random_state=args.seed))], final_estimator=GradientBoostingClassifier(random_state=args.seed), cv=5, n_jobs=-1)
    model = Pipeline([("preprocess", preprocess), ("classifier", ensemble)])
    model.fit(x_train, y_train)
    result = {"family": "B", "method": "agronomic_summary_features_plus_stacked_tree_ensemble", "dataset": str(args.data), "split": "stratified 70/15/15, seed=42", "rows": {"train": len(x_train), "validation": len(x_validation), "test": len(x_test)}, "classes": sorted(y.unique()), "validation": score(model, x_validation, y_validation), "test": score(model, x_test, y_test)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "model.pkl").open("wb") as file: pickle.dump(model, file)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("family", "method", "rows", "classes", "validation", "test")}, indent=2))


if __name__ == "__main__": main()
