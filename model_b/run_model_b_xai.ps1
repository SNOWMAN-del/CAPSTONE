param(
    [string]$ProjectData = "Dataset/PROJECT_DATA",
    [int]$BatchSize = 96,
    [int]$NumWorkers = 4
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"

& $Python model_b\xai_occlusion_model_b.py --task disease --project-data $ProjectData --batch-size $BatchSize --num-workers $NumWorkers
& $Python model_b\xai_occlusion_model_b.py --task stress --project-data $ProjectData --batch-size $BatchSize --num-workers $NumWorkers
