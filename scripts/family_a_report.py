from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def parse_args():
    parser = argparse.ArgumentParser(description="Create a traceable Family A implementation-status report.")
    parser.add_argument("--disease", type=Path, default=Path("runs_hybrid/ensemble_disease/validation_ensemble_metrics.json"))
    parser.add_argument("--stress", type=Path, default=Path("runs_hybrid/ensemble_stress/validation_ensemble_metrics.json"))
    parser.add_argument("--water", type=Path, default=Path("runs_hybrid/family_a_water_random_forest/metrics.json"))
    parser.add_argument("--disease-calibration", type=Path, default=Path("runs_hybrid/disease_calibration.json"))
    parser.add_argument("--stress-calibration", type=Path, default=Path("runs_hybrid/stress_calibration.json"))
    parser.add_argument("--final-test", type=Path, default=Path("runs_hybrid/family_a_final_test_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("runs_hybrid/family_a_status_report.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    report = {
        "claim": "Uncertainty-Aware Multimodal Edge-AI Framework for Disease, Water-Stress and Irrigation Decision Support in Medicinal and Aromatic Crops",
        "evidence": {"disease": read_json(args.disease), "water_stress": read_json(args.stress), "water": read_json(args.water), "disease_calibration": read_json(args.disease_calibration), "stress_calibration": read_json(args.stress_calibration), "final_held_out_test": read_json(args.final_test)},
        "completion_rules": {
            "final_evaluation": "Run only after model design, calibration settings, and irrigation policy are frozen.",
            "deployment": "Report target-device latency, peak memory, parameter count, and model size for each selected model.",
            "irrigation": "Validate advisory policy with field constraints and agronomist review before operational use.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
