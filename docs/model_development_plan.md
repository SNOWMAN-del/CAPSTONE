# Model Development Plan

This project keeps the handoff architecture intact: exactly two model families cover all data components.

## Family A

- Disease: pretrained EfficientNetV2-S classifier over the 21 disease/health classes. The compact CNN remains available as the baseline.
- Multimodal stress: separate paired RGB and thermal encoders, feature-level fusion, and a validation-selected CNN ensemble across EfficientNetV2-S, ResNet18, and MobileNetV2.
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

Run the three Family A RGB/thermal fusion members, then select ensemble weights using validation only:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m capstone_ai.train --task stress --family a --backbone efficientnet_v2_s --batch-size 16 --epochs 40 --learning-rate 1e-4 --freeze-epochs 3 --early-stopping-patience 8 --scheduler-patience 3 --num-workers 4 --output-dir runs_hybrid
.\.venv\Scripts\python.exe -m capstone_ai.train --task stress --family a --backbone resnet18 --batch-size 16 --epochs 40 --learning-rate 1e-4 --freeze-epochs 3 --early-stopping-patience 8 --scheduler-patience 3 --num-workers 4 --output-dir runs_hybrid
.\.venv\Scripts\python.exe -m capstone_ai.train --task stress --family a --backbone mobilenet_v2 --batch-size 16 --epochs 40 --learning-rate 1e-4 --freeze-epochs 3 --early-stopping-patience 8 --scheduler-patience 3 --num-workers 4 --output-dir runs_hybrid
.\.venv\Scripts\python.exe -m capstone_ai.ensemble_stress --efficientnet-checkpoint runs_hybrid/family_a_stress_efficientnet_v2_s/best.pt --resnet-checkpoint runs_hybrid/family_a_stress_resnet18/best.pt --mobilenet-checkpoint runs_hybrid/family_a_stress_mobilenet_v2/best.pt --batch-size 16 --num-workers 4 --output-dir runs_hybrid/ensemble_stress
```

Do not add `--evaluate-test` while selecting the ensemble or tuning its members.

Run compact smoke tests for the remaining Family A components:

```powershell
.\.venv\Scripts\python.exe -m capstone_ai.train --task disease --family a --backbone efficientnet_v2_s --epochs 1 --limit-batches 1 --num-workers 0
.\.venv\Scripts\python.exe -m capstone_ai.train --task stress --family a --epochs 1 --limit-batches 1 --num-workers 0
.\.venv\Scripts\python.exe -m capstone_ai.train --task water --family a --epochs 1 --limit-batches 1 --num-workers 0
```

Then train each task/family without `--limit-batches`.

## Completing the Family A Framework

The title makes four operational claims beyond prediction quality: multimodal learning, uncertainty awareness, edge deployment evidence, and irrigation decision support. The commands below preserve the chronological water split and keep the test split closed unless `--evaluate-test` is explicitly supplied after all selections are frozen.

### 1. Structured water baseline

Install the added dependency, then train the validation-only Random Forest comparator:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m capstone_ai.water_baselines --project-data Dataset/PROJECT_DATA --output-dir runs_hybrid/family_a_water_random_forest
```

Compare its `metrics.json` to the MLP on original ETa/Ks units using MAE, RMSE, and R2. Do not select a model using the test split.

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m capstone_ai.evaluate_water --checkpoint runs_hybrid/family_a_water_baseline/best.pt --output runs_hybrid/family_a_water_baseline/validation_metrics.json
```

### 2. Calibration and uncertainty

Fit temperature scaling on validation data and report ECE, NLL, Brier score, and ensemble disagreement:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m capstone_ai.calibrate --task disease --efficientnet-checkpoint runs_efficientnet/family_a_disease/best.pt --resnet-checkpoint runs_hybrid/family_a_disease_resnet18/best.pt --mobilenet-checkpoint runs_hybrid/family_a_disease_mobilenet_v2/best.pt --weights 0.5 0.1 0.4 --output runs_hybrid/disease_calibration.json
```

Repeat for stress with its three checkpoints and weights `0.6 0.3 0.1`. Water predictions use MC dropout and return a predictive standard deviation. Neither method substitutes for an external OOD dataset; OOD testing remains a required final experiment.

### 3. Advisory irrigation decision

Water inference returns ETa, Ks, MC-dropout uncertainty, and a transparent advisory irrigation amount. The policy defaults are deliberately conservative placeholders and must be field-calibrated. A JSON policy may set `effective_rainfall_fraction`, `application_efficiency`, `maximum_daily_irrigation_mm`, and `stress_ks_threshold`.

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m capstone_ai.predict_family_a water --input-csv sample_water_row.csv --rainfall-mm 2.0 --irrigation-config irrigation_policy.json
```

### 4. Edge measurements and final report

Profile every selected model on the actual intended deployment device. Use batch size one, record the device, and run the same command for disease, stress, and water.

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m capstone_ai.benchmark --task disease --backbone efficientnet_v2_s --checkpoint runs_efficientnet/family_a_disease/best.pt --output runs_hybrid/edge_disease_efficientnet.json
.\.venv\Scripts\python.exe scripts/family_a_report.py
```

The Family A title should only be presented as complete after the RF/MLP selection, representative XAI documentation, calibration/OOD checks, target-device benchmarks, irrigation-policy validation, and one locked held-out evaluation are complete.

### One validation-evidence run

The following script runs the RF comparison, calibration, and edge measurements. Use a CUDA/edge device for the image workloads. It never evaluates a reserved test split.

```powershell
.\scripts\run_family_a_completion.ps1
```

To add a declared external disease OOD image directory:

```powershell
.\scripts\run_family_a_completion.ps1 -OodDirectory D:\data\external_leaf_images
```

The OOD command measures maximum-softmax-probability AUROC and average precision. It is valid only for the external distribution you name and document.

### Representative XAI evidence

This command selects a correct disease validation prediction, a correct RGB-thermal stress validation prediction, and the lowest-error water validation row. It then writes Grad-CAM overlays, water Integrated Gradients attribution, and `xai_evidence.json` to one folder. It does not use held-out test inputs.

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\generate_family_a_xai_evidence.py
```

### Locked final evaluation

`configs/family_a_final.json` records the frozen checkpoints, ensemble weights, and validation-fitted temperatures. Run this exactly once after retaining a copy of the generated JSON result:

```powershell
.\scripts\run_family_a_final_evaluation.ps1
```

It reports held-out disease and stress accuracy plus calibrated ECE/NLL/Brier score, ensemble disagreement, and original-unit water regression metrics. It does not run OOD evaluation because that requires a separately documented external dataset.
