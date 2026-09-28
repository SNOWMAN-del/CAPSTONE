from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import build_disease_loaders, build_stress_loaders, build_water_loaders
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure parameter count, batch-one latency, and peak GPU memory.")
    parser.add_argument("--task", choices=("disease", "stress", "water"), required=True)
    parser.add_argument("--backbone", default="baseline")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("runs_hybrid/edge_benchmark.json"))
    return parser.parse_args()


def first_input(args):
    if args.task == "disease":
        bundle = build_disease_loaders(args.project_data, args.image_size, 1, 0, imagenet_normalize=args.backbone != "baseline")
        return bundle.metadata, (next(iter(bundle.validation))["image"],)
    if args.task == "stress":
        bundle = build_stress_loaders(args.project_data, args.image_size, 1, 0, imagenet_normalize=args.backbone != "fusion")
        batch = next(iter(bundle.validation))
        return bundle.metadata, (batch["rgb"], batch["thermal"])
    bundle = build_water_loaders(args.project_data, 1, 0)
    return bundle.metadata, (next(iter(bundle.validation))["features"],)


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be positive.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata, inputs = first_input(args)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model("a", args.task, checkpoint.get("metadata", metadata), args.image_size, backbone=args.backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    inputs = tuple(value.to(device) for value in inputs)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(args.runs):
            start = time.perf_counter()
            model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            durations.append((time.perf_counter() - start) * 1000)
    result = {
        "task": args.task,
        "backbone": args.backbone,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "input_batch_size": 1,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_size_mb": round(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1024**2, 3),
        "latency_ms": {"median": float(torch.tensor(durations).median()), "p95": float(torch.quantile(torch.tensor(durations), 0.95)), "runs": args.runs},
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3) if device.type == "cuda" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
