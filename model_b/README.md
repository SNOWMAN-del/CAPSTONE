# Model B — fast classical approach

This folder contains the simplified second model family requested for the capstone. It does not use Transformers.

- Disease: frozen ImageNet MobileNetV2 features + linear SVM.
- RGB/thermal stress: frozen MobileNetV2 features from each image, concatenated + linear SVM.
- Water need: Extra Trees regression.

The supplied dataset splits are preserved. Each command evaluates validation data only by default; do not add `--evaluate-test` until you have chosen the final settings.

## Run

From the repository root in a Python virtual environment (no `PYTHONPATH` setup is needed):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The source datasets are deliberately not stored in Git because of their size. Place
the supplied project data at `Dataset/PROJECT_DATA` before running training or final
evaluation.

Then run:

```powershell
.\.venv\Scripts\python.exe model_b\train_classical_model_b.py --task disease --project-data Dataset/PROJECT_DATA
.\.venv\Scripts\python.exe model_b\train_classical_model_b.py --task stress --project-data Dataset/PROJECT_DATA
.\.venv\Scripts\python.exe model_b\train_classical_model_b.py --task water --project-data Dataset/PROJECT_DATA
```

Results are written to `runs_classical_b\family_b_<task>_classical`. The script saves the fitted model and `metrics.json` for each task.

## Next: choose the water model

The first Extra Trees result should be compared before Model B water prediction is finalized. This command tests Extra Trees, Random Forest, Ridge, and two RBF-SVR settings on validation data only, then saves the validation-selected model. It never opens the test split.

```powershell
.\.venv\Scripts\python.exe model_b\compare_water_models.py --project-data Dataset/PROJECT_DATA
```

## Locked final test evaluation

After accepting the validation-selected models, run this command exactly once. It refuses to overwrite an existing test report.

```powershell
.\.venv\Scripts\python.exe model_b\evaluate_final_model_b.py --project-data Dataset/PROJECT_DATA --confirm-final-evaluation
```

## Model B image XAI

After running the disease and stress calibration commands, create occlusion-sensitivity overlays from correct validation predictions only:

```powershell
.\model_b\run_model_b_xai.ps1
```

The script writes disease and RGB/thermal stress overlays into each task's `runs_classical_b\...\evidence` folder. Red regions indicate patches whose occlusion reduced the calibrated probability of the selected class; this shows model sensitivity, not biological causation.

## Important reporting note

Because this is not a Transformer, call it a **classical transfer-learning/ML baseline** in the final report. It still remains Model/Family B, so the project has two model families rather than separate models for each dataset.
