param(
    [string]$ProjectData = "Dataset/PROJECT_DATA",
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
& $Python scripts\run_family_a_final_evaluation.py --project-data $ProjectData
