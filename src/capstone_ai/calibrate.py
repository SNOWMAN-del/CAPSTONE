from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .data import build_disease_loaders, build_stress_loaders
from .metrics import classification_calibration
from .models import build_model


BACKBONES = ("efficientnet_v2_s", "resnet18", "mobilenet_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit temperature scaling and report calibration for a Family A image ensemble.")
    parser.add_argument("--task", choices=("disease", "stress"), required=True)
    parser.add_argument("--efficientnet-checkpoint", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--mobilenet-checkpoint", type=Path, required=True)
    parser.add_argument("--weights", nargs=3, type=float, required=True)
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs_hybrid/calibration.json"))
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def loaders_for(args):
    if args.task == "disease":
        return build_disease_loaders(args.project_data, args.image_size, args.batch_size, args.num_workers, imagenet_normalize=True)
    return build_stress_loaders(args.project_data, args.image_size, args.batch_size, args.num_workers, imagenet_normalize=True)


def members(args, metadata, device):
    models = []
    for checkpoint_path, backbone in zip((args.efficientnet_checkpoint, args.resnet_checkpoint, args.mobilenet_checkpoint), BACKBONES):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model("a", args.task, metadata, args.image_size, backbone=backbone, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def collect_logits(models, loader, task, weights, device):
    logits, labels, member_probabilities = [], [], []
    for batch in loader:
        if task == "disease":
            outputs = torch.stack([model(batch["image"].to(device)) for model in models])
        else:
            rgb, thermal = batch["rgb"].to(device), batch["thermal"].to(device)
            outputs = torch.stack([model(rgb, thermal) for model in models])
        probabilities = torch.softmax(outputs, dim=2)
        ensemble = (probabilities * weights[:, None, None]).sum(dim=0)
        logits.append(ensemble.clamp_min(1e-12).log().cpu())
        labels.append(batch["label"].cpu())
        member_probabilities.append(probabilities.cpu())
    return torch.cat(logits), torch.cat(labels), torch.cat(member_probabilities, dim=1)


def fit_temperature(logits, labels) -> float:
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    criterion = nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 20))


def main() -> None:
    args = parse_args()
    if any(weight < 0 for weight in args.weights) or sum(args.weights) <= 0:
        raise ValueError("Ensemble weights must be non-negative and sum to more than zero.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = loaders_for(args)
    weights = torch.tensor(args.weights, device=device) / sum(args.weights)
    models = members(args, loaders.metadata, device)
    validation_logits, validation_labels, member_probabilities = collect_logits(models, loaders.validation, args.task, weights, device)
    temperature = fit_temperature(validation_logits, validation_labels)
    raw = torch.softmax(validation_logits, dim=1)
    calibrated = torch.softmax(validation_logits / temperature, dim=1)
    disagreement = member_probabilities.var(dim=0).mean(dim=1)
    result = {
        "task": args.task,
        "calibration_split": "validation",
        "temperature": temperature,
        "validation_before": classification_calibration(raw, validation_labels),
        "validation_after": classification_calibration(calibrated, validation_labels),
        "validation_mean_ensemble_disagreement": float(disagreement.mean()),
        "note": "Temperature is fitted on validation data; validation-after metrics are optimistic and must not be called final performance.",
    }
    if args.evaluate_test:
        test_logits, test_labels, test_members = collect_logits(models, loaders.test, args.task, weights, device)
        result["test_after"] = classification_calibration(torch.softmax(test_logits / temperature, dim=1), test_labels)
        result["test_mean_ensemble_disagreement"] = float(test_members.var(dim=0).mean(dim=1).mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
