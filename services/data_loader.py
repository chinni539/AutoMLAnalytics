import pandas as pd
from pathlib import Path
from config.logger import logger

SUPPORTED_EXTENSIONS = [".csv", ".xlsx"]


class DataLoader:

    @staticmethod
    def load_data(uploaded_file):

        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported file format.")

        logger.info(f"Loading {uploaded_file.name}")

        if suffix == ".csv":
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        logger.info(f"Dataset Shape : {df.shape}")

        return df