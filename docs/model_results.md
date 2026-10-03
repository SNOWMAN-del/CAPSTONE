# Model results

This page publishes the currently available evaluation results for both model
families. Classification figures are accuracy; water prediction is a regression
task, so MAE and RMSE are reported instead. Do not compare validation and test
results as though they were measured on the same split.

## Project-data models: locked held-out test results

| Family | Task | Model / selected approach | Samples | Accuracy | Macro F1 | Other metrics |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | Disease classification | Calibrated image ensemble | 1,884 | 99.68% | — | ECE after calibration: 0.0050 |
| A | Water-stress classification | Calibrated RGB/thermal ensemble | 322 | 97.52% | — | ECE after calibration: 0.0246 |
| A | Water prediction | Water baseline | — | N/A | N/A | Mean MAE: 0.6268; mean RMSE: 1.0279 |
| B | Disease classification | MobileNetV2 features + linear SVM | 1,884 | 94.90% | 95.88% | — |
| B | Water-stress classification | RGB/thermal MobileNetV2 features + linear SVM | 322 | 92.86% | 92.95% | — |
| B | Water prediction | RBF-SVR (C=10) | — | N/A | N/A | Mean MAE: 0.5384; mean RMSE: 0.8824 |

Family A's locked-test metrics come from the one-time held-out evaluation.
Family B's locked-test metrics are stored in
`runs_classical_b/final_model_b_test_metrics.json`.

## Kaggle irrigation classification: held-out test results

Both Kaggle models used a stratified 70/15/15 split (seed 42), with 1,500 test
rows. These are separate from the project-data water-prediction task above.

| Family | Method | Accuracy | Balanced accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| A | Random Forest | 98.27% | 85.43% | 90.01% |
| B | Agronomic-summary features + stacked tree ensemble | 99.67% | 97.94% | 98.50% |

The Kaggle dataset and Family B stacked model are not in the repository because
they are large. The reported figures are retained here so they remain visible
to collaborators without requiring a dataset download.

## Irrigation-event experiment: validation result

Family A's retrospective next-observation irrigation-event experiment was
evaluated on 336 validation rows from 2014. It achieved precision 30.61%,
recall 31.25%, F1 30.93%, average precision 0.2249, and ROC-AUC 0.5821. The
2015–2016 period was held out and has not been evaluated by that experiment.

## Interpretation notes

- Results are experimental and dataset-specific; they are not evidence of field
  validation or approval for operational irrigation decisions.
- The Family B water model was selected using validation data before the locked
  held-out test evaluation.
- A missing Macro F1 for Family A reflects the saved evaluation report, which
  recorded calibrated accuracy and calibration metrics instead.
