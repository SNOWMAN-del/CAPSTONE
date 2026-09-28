from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .paths import disease_dir, project_data_dir, stress_dir, water_dir


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WATER_TARGET_COLUMNS = ["Target_ETa_t1", "Target_Ks_t1"]
WATER_NON_FEATURE_COLUMNS = [
    "Date",
    "Prediction_Target_Date",
    "Treatment",
    "Split",
    "Target_ETa_t1",
    "Target_Ks_t1",
]


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    metadata: dict


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_class_mapping(root: Path) -> dict[tuple[str, str, str], int]:
    mapping_file = root / "MASTER" / "class_mapping.csv"
    mapping: dict[tuple[str, str, str], int] = {}
    with mapping_file.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mapping[(row["dataset"], row["plant"], row["original_class"])] = int(row["class_id"])
    return mapping


def _identify_disease_class(path: Path, split_dir: Path, mapping: dict[tuple[str, str, str], int]) -> int:
    parts = path.relative_to(split_dir).parts
    dataset = parts[0]
    if dataset == "AI-MedLeaf":
        plant = parts[1]
        original_class = parts[2]
    elif dataset == "Medicinal_leaf_dataset":
        folder_name = parts[1]
        if folder_name.startswith("Kalanchoe_"):
            plant = "Kalanchoe"
            original_class = folder_name[len("Kalanchoe_") :]
        elif folder_name.startswith("Tulsi_"):
            plant = "Tulsi"
            original_class = folder_name[len("Tulsi_") :]
        else:
            raise ValueError(f"Unknown medicinal disease folder: {folder_name}")
    else:
        raise ValueError(f"Unknown disease dataset folder: {dataset}")
    return mapping[(dataset, plant, original_class)]


class DiseaseDataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.split_dir = disease_dir(root) / "SPLITS" / split
        self.transform = transform
        mapping = _load_class_mapping(root)
        self.samples = []
        for image_path in sorted(self.split_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                self.samples.append((image_path, _identify_disease_class(image_path, self.split_dir, mapping)))
        if not self.samples:
            raise RuntimeError(f"No disease images found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": torch.tensor(label, dtype=torch.long)}


class PairedStressDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int, train: bool, normalize=None):
        self.root = stress_dir(root)
        self.split = split
        manifest = pd.read_csv(self.root / "METADATA" / "split_manifest.csv")
        self.dataframe = manifest[manifest["split"] == split].reset_index(drop=True)
        class_names = sorted(manifest["class"].unique())
        self.class_to_id = {name: index for index, name in enumerate(class_names)}
        self.train = train
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.dataframe)

    def _resolve(self, relative_path: str) -> Path:
        path = Path(str(relative_path).strip())
        if path.is_absolute() and path.exists():
            return path
        rooted_path = self.root / path
        if rooted_path.exists():
            return rooted_path

        parts = path.parts
        if "RGB_Dataset" in parts:
            split_path = self.root / "SPLITS" / self.split / "RGB" / path.name
        elif "Thermal_Dataset" in parts:
            split_path = self.root / "SPLITS" / self.split / "Thermal" / path.name
        else:
            split_path = self.root / "SPLITS" / self.split / path.name
        if split_path.exists():
            return split_path

        raise FileNotFoundError(f"Could not resolve stress image path from manifest entry: {relative_path}")

    def _transform_pair(self, rgb: Image.Image, thermal: Image.Image):
        if self.train:
            rgb = rgb.resize((256, 256), Image.Resampling.BILINEAR)
            thermal = thermal.resize((256, 256), Image.Resampling.BILINEAR)
            left = random.randint(0, 256 - self.image_size)
            top = random.randint(0, 256 - self.image_size)
            crop = (left, top, left + self.image_size, top + self.image_size)
            rgb = rgb.crop(crop)
            thermal = thermal.crop(crop)
            if random.random() < 0.5:
                rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                thermal = thermal.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            rgb = rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            thermal = thermal.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        rgb_tensor, thermal_tensor = self.to_tensor(rgb), self.to_tensor(thermal)
        if self.normalize is not None:
            rgb_tensor, thermal_tensor = self.normalize(rgb_tensor), self.normalize(thermal_tensor)
        return rgb_tensor, thermal_tensor

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        rgb = Image.open(self._resolve(row["rgb_path"])).convert("RGB")
        thermal = Image.open(self._resolve(row["thermal_path"])).convert("RGB")
        rgb, thermal = self._transform_pair(rgb, thermal)
        return {
            "rgb": rgb,
            "thermal": thermal,
            "label": torch.tensor(self.class_to_id[row["class"]], dtype=torch.long),
        }


class WaterDataset(Dataset):
    def __init__(self, root: Path, split: str, stats: dict | None = None):
        split_file = water_dir(root) / "SPLITS" / split / f"temporal_water_{split}.csv"
        self.df = pd.read_csv(split_file, low_memory=False)
        self.feature_columns = [column for column in self.df.columns if column not in WATER_NON_FEATURE_COLUMNS]
        for column in self.feature_columns + WATER_TARGET_COLUMNS:
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce")
        if self.df[WATER_TARGET_COLUMNS].isna().any().any():
            raise ValueError(f"Missing water targets in {split_file}")

        features = self.df[self.feature_columns].to_numpy(dtype=np.float32)
        targets = self.df[WATER_TARGET_COLUMNS].to_numpy(dtype=np.float32)
        if stats is None:
            feature_mean = np.nanmean(features, axis=0)
            feature_std = np.nanstd(features, axis=0)
            target_mean = targets.mean(axis=0)
            target_std = targets.std(axis=0)
            stats = {
                "feature_mean": feature_mean,
                "feature_std": np.where(feature_std == 0, 1.0, feature_std),
                "target_mean": target_mean,
                "target_std": np.where(target_std == 0, 1.0, target_std),
            }
        self.stats = stats
        features = np.where(np.isnan(features), stats["feature_mean"], features)
        self.features = ((features - stats["feature_mean"]) / stats["feature_std"]).astype(np.float32)
        self.targets = ((targets - stats["target_mean"]) / stats["target_std"]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "features": torch.tensor(self.features[index], dtype=torch.float32),
            "targets": torch.tensor(self.targets[index], dtype=torch.float32),
        }


def build_disease_loaders(
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    imagenet_normalize: bool = False,
) -> LoaderBundle:
    root = project_data_dir(data_root)
    normalize = (
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if imagenet_normalize
        else nn.Identity()
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])
    train = DiseaseDataset(root, "train", train_transform)
    validation = DiseaseDataset(root, "validation", eval_transform)
    test = DiseaseDataset(root, "test", eval_transform)
    metadata = {"num_classes": 21, "input_channels": 3}
    return LoaderBundle(
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        metadata,
    )


def build_stress_loaders(
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    imagenet_normalize: bool = False,
) -> LoaderBundle:
    root = project_data_dir(data_root)
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) if imagenet_normalize else None
    train = PairedStressDataset(root, "train", image_size, train=True, normalize=normalize)
    validation = PairedStressDataset(root, "validation", image_size, train=False, normalize=normalize)
    test = PairedStressDataset(root, "test", image_size, train=False, normalize=normalize)
    metadata = {"num_classes": len(train.class_to_id), "input_channels": 6, "class_to_id": train.class_to_id}
    return LoaderBundle(
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        metadata,
    )


def build_water_loaders(data_root: str | Path, batch_size: int, num_workers: int) -> LoaderBundle:
    root = project_data_dir(data_root)
    train = WaterDataset(root, "train")
    validation = WaterDataset(root, "validation", stats=train.stats)
    test = WaterDataset(root, "test", stats=train.stats)
    metadata = {
        "num_features": len(train.feature_columns),
        "num_targets": 2,
        "feature_columns": train.feature_columns,
        "target_stats": {
            "mean": train.stats["target_mean"].tolist(),
            "std": train.stats["target_std"].tolist(),
        },
    }
    return LoaderBundle(
        DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        metadata,
    )
