import os
import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException


# =========================
# PROJECT ROOT
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# =========================
# CONFIG CLASS
# =========================
@dataclass
class DataIngestionConfig:
    train_data_path: Path = PROJECT_ROOT / "artifacts" / "train.csv"
    test_data_path: Path = PROJECT_ROOT / "artifacts" / "test.csv"
    raw_data_path: Path = PROJECT_ROOT / "artifacts" / "raw.csv"


# =========================
# DATA INGESTION CLASS
# =========================
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion method")

        try:
            # Correct dataset path
            data_path = PROJECT_ROOT / "notebook" / "data" / "stud.csv"
            df = pd.read_csv(data_path)

            logging.info("Dataset read successfully")

            # Create artifacts folder in ROOT
            self.ingestion_config.train_data_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # Train test split
            logging.info("Train test split started")

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# =========================
# RUN SCRIPT
# =========================
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()