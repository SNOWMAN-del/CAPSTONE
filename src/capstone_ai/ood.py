from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .data import VALID_IMAGE_EXTENSIONS, build_disease_loaders
from .models import build_model


BACKBONES = ("efficientnet_v2_s", "resnet18", "mobilenet_v2")


class ImageFolderDataset(Dataset):
    def __init__(self, directory: Path, transform):
        self.paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS)
        if not self.paths:
            raise ValueError(f"No supported images found in {directory}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(Image.open(self.paths[index]).convert("RGB"))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate disease-ensemble OOD detection using an external image set.")
    parser.add_argument("--ood-dir", type=Path, required=True, help="External images not represented by the disease training distribution.")
    parser.add_argument("--efficientnet-checkpoint", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--mobilenet-checkpoint", type=Path, required=True)
    parser.add_argument("--weights", nargs=3, type=float, required=True)
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs_hybrid/disease_ood.json"))
    return parser.parse_args()


def load_models(args, metadata, device):
    models = []
    for path, backbone in zip((args.efficientnet_checkpoint, args.resnet_checkpoint, args.mobilenet_checkpoint), BACKBONES):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = build_model("a", "disease", metadata, args.image_size, backbone=backbone, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    return models


@torch.inference_mode()
def scores(models, loader, weights, device):
    values = []
    for batch in loader:
        images = batch["image"] if isinstance(batch, dict) else batch
        probabilities = torch.stack([torch.softmax(model(images.to(device)), dim=1) for model in models])
        values.extend((probabilities * weights[:, None, None]).sum(dim=0).max(dim=1).values.cpu().tolist())
    return np.asarray(values)


def main():
    args = parse_args()
    if any(weight < 0 for weight in args.weights) or sum(args.weights) <= 0:
        raise ValueError("Ensemble weights must be non-negative and sum to more than zero.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_distribution = build_disease_loaders(args.project_data, args.image_size, args.batch_size, args.num_workers, imagenet_normalize=True)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    ood_loader = DataLoader(ImageFolderDataset(args.ood_dir, transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    models = load_models(args, in_distribution.metadata, device)
    weights = torch.tensor(args.weights, device=device) / sum(args.weights)
    id_scores = scores(models, in_distribution.validation, weights, device)
    ood_scores = scores(models, ood_loader, weights, device)
    labels = np.concatenate((np.zeros(len(id_scores)), np.ones(len(ood_scores))))
    anomaly_scores = np.concatenate((-id_scores, -ood_scores))
    result = {
        "task": "disease",
        "method": "maximum_softmax_probability",
        "in_distribution_split": "validation",
        "ood_source": str(args.ood_dir),
        "auroc": float(roc_auc_score(labels, anomaly_scores)),
        "average_precision": float(average_precision_score(labels, anomaly_scores)),
        "mean_id_confidence": float(id_scores.mean()),
        "mean_ood_confidence": float(ood_scores.mean()),
        "id_samples": int(len(id_scores)),
        "ood_samples": int(len(ood_scores)),
        "note": "OOD conclusions apply only to this declared external distribution and score; do not generalize them to all field shifts.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
