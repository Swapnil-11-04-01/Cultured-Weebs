import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    data_path: str = os.path.join("artifacts", "anime.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            data = pd.read_csv(self.ingestion_config.data_path, delimiter='\t')
            logging.info(f"Successfully completed data ingestion")
            return data
        except Exception as e:
            raise CustomException(e, sys)
