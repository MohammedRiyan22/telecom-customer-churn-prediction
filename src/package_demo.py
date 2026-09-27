"""Build a clean, self-contained Streamlit demo folder and ZIP archive."""

from pathlib import Path
from shutil import copy2
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "Customer_Churn_Demo"
ARCHIVE_PATH = PROJECT_ROOT / "Customer_Churn_Demo.zip"

FILES_TO_COPY = {
    PROJECT_ROOT / "src" / "app.py": DEMO_ROOT / "src" / "app.py",
    PROJECT_ROOT / "models" / "customer_churn_production_model.joblib": (
        DEMO_ROOT / "models" / "customer_churn_production_model.joblib"
    ),
    PROJECT_ROOT / "models" / "final_model_metadata.json": (
        DEMO_ROOT / "models" / "final_model_metadata.json"
    ),
    PROJECT_ROOT / "requirements.txt": DEMO_ROOT / "requirements.txt",
}

DEMO_README = """# Customer Churn Intelligence Demo

## Run locally

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
streamlit run src/app.py
```

The demo includes the verified CatBoost artifact and matching metadata. The application checks the model type, ordered feature list, and threshold before enabling predictions.

To audit the artifact from the full project, run `python src/verify_final_model.py` at the project root.
"""


def main():
    for source, destination in FILES_TO_COPY.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required deployment file is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)

    (DEMO_ROOT / "README.md").write_text(DEMO_README, encoding="utf-8")

    with ZipFile(ARCHIVE_PATH, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(DEMO_ROOT.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            archive.write(path, Path(DEMO_ROOT.name) / path.relative_to(DEMO_ROOT))

    print(f"Deployment folder: {DEMO_ROOT}")
    print(f"Deployment archive: {ARCHIVE_PATH}")
    print(f"Archive size: {ARCHIVE_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
