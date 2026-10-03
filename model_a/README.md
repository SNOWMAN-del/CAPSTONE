# Family A irrigation-event retry

This is a separate, retrospective irrigation-event classification experiment built from the project's master water table. It is not a replacement for the earlier ETa/Ks regression task.

It predicts whether recorded irrigation occurs at the next observation for the same treatment. Inputs are limited to current-observation weather, soil-water, treatment, and calendar variables. Current or future irrigation is never an input.

Run the validation-only experiment:

```powershell
.\.venv\Scripts\python.exe model_a\train_irrigation_event.py --project-data Dataset/PROJECT_DATA
```

The chronological protocol is fixed: 2008-2013 training, 2014 validation, and 2015-2016 remains untouched until selection is finalized.

## Kaggle irrigation benchmark evidence

After training `train_kaggle_irrigation_a.py`, run the validation-only uncertainty and XAI analysis:

```powershell
.\model_a\run_kaggle_irrigation_evidence.ps1
```

It reports ECE, NLL, Brier score, and predictive entropy, then saves permutation feature importance. The locked Kaggle test split is not used.
