# Family A XAI Findings

## Protocol

All examples were selected from the validation split. No held-out test inputs were used. One correct disease prediction and one correct RGB-thermal stress prediction were selected automatically; the water example is the validation row with the lowest mean standardized absolute error. Attribution is evidence of model sensitivity, not causal diagnosis or agronomic mechanism.

## Disease

- Input: `Camphor/Bacterial Spot_1.jpg`
- Expected and predicted class: `Camphor_Bacterial Spot`
- Ensemble confidence: `0.999897`
- Evidence: EfficientNet, ResNet, MobileNet, and ensemble Grad-CAM overlays are in `runs_hybrid/xai_evidence/disease/`.

The overlays identify image regions used by the ensemble for this correct prediction. They should be inspected visually with the original image; they do not establish that the highlighted regions are biologically causal lesions.

## RGB-Thermal Water Stress

- Input pair: RGB `FLIR4146.jpg`, thermal `FLIR4145.jpg`
- Expected and predicted class: `Guava Healthy`
- Ensemble confidence: `0.999247`
- Evidence: RGB and thermal ensemble Grad-CAM overlays are in `runs_hybrid/xai_evidence/stress/`.

The paired overlays show that the multimodal model produces a spatial explanation for each sensing modality. This is a qualitative explanation only; it does not prove which modality is more important globally.

## Water Prediction

- Target explained: next-day ETa
- Selected validation row: `20`
- Top positive attributions: `HW_WindDir_STDD_Mean`, `WB_Irrigation`, `SWC_30_Std`, `DOY`
- Top negative attribution: `Year`

The water model's Integrated Gradients values are relative to the training-average baseline. Positive attribution increases the predicted ETa relative to that baseline; negative attribution decreases it. These values should not be interpreted as a field irrigation rule without agronomic validation.
