param(
    [string]$ProjectData = "Dataset/PROJECT_DATA",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$OodDirectory = "",
    [switch]$FinalTest
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

& $Python -m capstone_ai.water_baselines --project-data $ProjectData --output-dir runs_hybrid/family_a_water_random_forest
& $Python -m capstone_ai.evaluate_water --project-data $ProjectData --checkpoint runs_hybrid/family_a_water_baseline/best.pt --output runs_hybrid/family_a_water_baseline/validation_metrics.json

& $Python -m capstone_ai.calibrate --task disease --project-data $ProjectData --efficientnet-checkpoint runs_efficientnet/family_a_disease/best.pt --resnet-checkpoint runs_hybrid/family_a_disease_resnet18/best.pt --mobilenet-checkpoint runs_hybrid/family_a_disease_mobilenet_v2/best.pt --weights 0.5 0.1 0.4 --output runs_hybrid/disease_calibration.json
& $Python -m capstone_ai.calibrate --task stress --project-data $ProjectData --efficientnet-checkpoint runs_hybrid/family_a_stress_efficientnet_v2_s/best.pt --resnet-checkpoint runs_hybrid/family_a_stress_resnet18/best.pt --mobilenet-checkpoint runs_hybrid/family_a_stress_mobilenet_v2/best.pt --weights 0.6 0.3 0.1 --output runs_hybrid/stress_calibration.json

& $Python -m capstone_ai.benchmark --task disease --backbone efficientnet_v2_s --project-data $ProjectData --checkpoint runs_efficientnet/family_a_disease/best.pt --output runs_hybrid/edge_disease_efficientnet.json
& $Python -m capstone_ai.benchmark --task stress --backbone efficientnet_v2_s --project-data $ProjectData --checkpoint runs_hybrid/family_a_stress_efficientnet_v2_s/best.pt --output runs_hybrid/edge_stress_efficientnet.json
& $Python -m capstone_ai.benchmark --task water --backbone baseline --project-data $ProjectData --checkpoint runs_hybrid/family_a_water_baseline/best.pt --output runs_hybrid/edge_water_mlp.json

if ($OodDirectory) {
    & $Python -m capstone_ai.ood --ood-dir $OodDirectory --project-data $ProjectData --efficientnet-checkpoint runs_efficientnet/family_a_disease/best.pt --resnet-checkpoint runs_hybrid/family_a_disease_resnet18/best.pt --mobilenet-checkpoint runs_hybrid/family_a_disease_mobilenet_v2/best.pt --weights 0.5 0.1 0.4 --output runs_hybrid/disease_ood.json
}

if ($FinalTest) {
    Write-Host "Final test evaluation is intentionally not automated here. Run it once only after documenting all frozen selections."
}

& $Python scripts/family_a_report.py
