import logging
from pathlib import Path

from config.config import LOG_FOLDER

LOG_FOLDER.mkdir(exist_ok=True)

LOG_FILE = Path(LOG_FOLDER, "application.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AutoML")