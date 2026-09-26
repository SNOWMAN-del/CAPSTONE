# Model Development Plan

This project keeps the handoff architecture intact: exactly two model families cover all data components.

## Family A

- Disease: pretrained EfficientNetV2-S classifier over the 21 disease/health classes. The compact CNN remains available as the baseline.
- Multimodal stress: compact CNN classifier over concatenated RGB and thermal channels.
- Water: MLP regressor for `[ETa(t+1), Ks(t+1)]`.

## Family B

- Disease: patch-attention image classifier.
- Multimodal stress: patch-attention classifier over concatenated RGB and thermal channels.
- Water: feature-token transformer regressor for `[ETa(t+1), Ks(t+1)]`.

## Current Data Decisions

- Prepared `SPLITS` are used directly.
- Disease and stress images use on-the-fly augmentation only for training.
- Water keeps the chronological split and does not shuffle training rows.
- Water feature imputation is limited to train-split mean imputation because PyTorch models cannot consume `NaN` values.
- Water features and targets are standardized with train-split statistics only.

## First Commands

Install dependencies in the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run a fast shape check:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/smoke_check.py --project-data Dataset/PROJECT_DATA --batch-size 2 --num-workers 0
```

Run the recommended Family A disease experiment:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m capstone_ai.train --task disease --family a --backbone efficientnet_v2_s --image-size 224 --batch-size 64 --epochs 30 --learning-rate 1e-4 --freeze-epochs 3 --early-stopping-patience 6 --scheduler-patience 3 --num-workers 4 --output-dir runs_efficientnet
```

The trainer writes `history.json`, retains the checkpoint with lowest validation loss, reloads that checkpoint before final test evaluation, and applies early stopping when requested.

Run compact smoke tests for the remaining Family A components:

```powershell
.\.venv\Scripts\python.exe -m capstone_ai.train --task disease --family a --backbone efficientnet_v2_s --epochs 1 --limit-batches 1 --num-workers 0
.\.venv\Scripts\python.exe -m capstone_ai.train --task stress --family a --epochs 1 --limit-batches 1 --num-workers 0
.\.venv\Scripts\python.exe -m capstone_ai.train --task water --family a --epochs 1 --limit-batches 1 --num-workers 0
```

Then train each task/family without `--limit-batches`.
