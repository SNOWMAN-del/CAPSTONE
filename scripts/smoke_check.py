from __future__ import annotations

import argparse

import torch

from capstone_ai.data import build_disease_loaders, build_stress_loaders, build_water_loaders
from capstone_ai.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-data", default="Dataset/PROJECT_DATA")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    checks = [
        ("disease", build_disease_loaders(args.project_data, args.image_size, args.batch_size, args.num_workers)),
        ("stress", build_stress_loaders(args.project_data, args.image_size, args.batch_size, args.num_workers)),
        ("water", build_water_loaders(args.project_data, args.batch_size, args.num_workers)),
    ]
    for task, loaders in checks:
        batch = next(iter(loaders.train))
        for family in ("a", "b"):
            model = build_model(family, task, loaders.metadata, args.image_size)
            if task == "disease":
                output = model(batch["image"])
            elif task == "stress":
                output = model(torch.cat([batch["rgb"], batch["thermal"]], dim=1))
            else:
                output = model(batch["features"])
            print(f"{task} family {family}: output shape {tuple(output.shape)}")


if __name__ == "__main__":
    main()
