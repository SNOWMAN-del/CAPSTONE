from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .data import build_disease_loaders, build_stress_loaders, build_water_loaders, seed_everything
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one capstone model family on one prepared data component.")
    parser.add_argument("--task", choices=["disease", "stress", "water"], required=True)
    parser.add_argument("--family", choices=["a", "b"], required=True)
    parser.add_argument(
        "--backbone",
        choices=["baseline", "efficientnet_v2_s", "resnet18", "mobilenet_v2", "fusion"],
        default="baseline",
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-batches", type=int, default=0, help="Use a small positive value for smoke tests.")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--torch-home", default=".torch_cache", help="Writable cache directory for pretrained weights.")
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the reserved test split only after the model design is frozen.",
    )
    return parser.parse_args()


def make_loaders(args: argparse.Namespace):
    if args.task == "disease":
        return build_disease_loaders(
            args.project_data,
            args.image_size,
            args.batch_size,
            args.num_workers,
            imagenet_normalize=args.backbone in {"efficientnet_v2_s", "resnet18", "mobilenet_v2"},
        )
    if args.task == "stress":
        return build_stress_loaders(
            args.project_data,
            args.image_size,
            args.batch_size,
            args.num_workers,
            imagenet_normalize=args.backbone in {"efficientnet_v2_s", "resnet18", "mobilenet_v2"},
        )
    if args.task == "water":
        return build_water_loaders(args.project_data, args.batch_size, args.num_workers)
    raise ValueError(args.task)


def batch_to_device(task: str, batch: dict, device: torch.device):
    if task == "disease":
        return batch["image"].to(device), batch["label"].to(device)
    if task == "stress":
        return (batch["rgb"].to(device), batch["thermal"].to(device)), batch["label"].to(device)
    if task == "water":
        return batch["features"].to(device), batch["targets"].to(device)
    raise ValueError(task)


def run_epoch(model, loader, criterion, optimizer, task: str, device: torch.device, limit_batches: int, train: bool):
    model.train(train)
    total_loss = 0.0
    total_items = 0
    correct = 0
    iterator = tqdm(loader, leave=False)
    for batch_index, batch in enumerate(iterator, start=1):
        inputs, targets = batch_to_device(task, batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
            loss = criterion(outputs, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        if task != "water":
            correct += (outputs.argmax(dim=1) == targets).sum().item()
        if limit_batches and batch_index >= limit_batches:
            break
    metrics = {"loss": total_loss / max(total_items, 1)}
    if task != "water":
        metrics["accuracy"] = correct / max(total_items, 1)
    return metrics


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(Path(args.torch_home).resolve()))
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(args)
    model = build_model(
        args.family,
        args.task,
        loaders.metadata,
        args.image_size,
        backbone=args.backbone,
        pretrained=args.pretrained,
    ).to(device)
    criterion = nn.MSELoss() if args.task == "water" else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=args.scheduler_patience, min_lr=args.min_learning_rate
    )

    run_dir = Path(args.output_dir) / f"family_{args.family}_{args.task}_{args.backbone}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(loaders.metadata, indent=2, default=str), encoding="utf-8")

    best_validation = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        if hasattr(model, "set_backbone_trainable"):
            model.set_backbone_trainable(epoch > args.freeze_epochs)
        if hasattr(model, "set_backbones_trainable"):
            model.set_backbones_trainable(epoch > args.freeze_epochs)
        train_metrics = run_epoch(
            model, loaders.train, criterion, optimizer, args.task, device, args.limit_batches, train=True
        )
        validation_metrics = run_epoch(
            model, loaders.validation, criterion, optimizer, args.task, device, args.limit_batches, train=False
        )
        scheduler.step(validation_metrics["loss"])
        epoch_result = {
            "epoch": epoch,
            "device": str(device),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, indent=2))
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "metadata": loaders.metadata,
                    "dataset_stats": getattr(loaders.train.dataset, "stats", None),
                    "epoch": epoch_result,
                },
                run_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
            print(json.dumps({"early_stopping": True, "epoch": epoch}, indent=2))
            break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    result = {"checkpoint": str(run_dir / "best.pt"), "best_epoch": best_checkpoint["epoch"]}
    if args.evaluate_test:
        model.load_state_dict(best_checkpoint["model"])
        test_metrics = run_epoch(model, loaders.test, criterion, optimizer, args.task, device, args.limit_batches, train=False)
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        result["test"] = test_metrics
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
