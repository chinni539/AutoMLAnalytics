from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = ROOT_DIR / "uploads"

LOG_FOLDER = ROOT_DIR / "logs"

APP_TITLE = "AutoML Analytics"

MAX_UPLOAD_SIZE_MB = 200

SUPPORTED_FILES = [
    "csv",
    "xlsx"
]