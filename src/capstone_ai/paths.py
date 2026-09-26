from pathlib import Path


def project_data_dir(path: str | Path = "Dataset/PROJECT_DATA") -> Path:
    root = Path(path)
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"PROJECT_DATA directory not found: {root}")
    return root


def disease_dir(root: Path) -> Path:
    return root / "01_disease"


def stress_dir(root: Path) -> Path:
    return root / "02_multimodal_stress"


def water_dir(root: Path) -> Path:
    return root / "04_water_need"
