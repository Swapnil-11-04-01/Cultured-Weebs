import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import data_splitter
from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
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
            data = pd.read_csv('notebook/data/anime.csv', delimiter='\t')
            logging.info(f"Successfully completed data ingestion")

            data.to_csv(self.ingestion_config.data_path, index=False, header=True)

            return data
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_obj = DataIngestion()
    dataframe = data_obj.initiate_data_ingestion()
    print(dataframe.shape)
    print(dataframe.columns)