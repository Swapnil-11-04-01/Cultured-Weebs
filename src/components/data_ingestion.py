import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.utils import cosine_similarity_matrix, load_object
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


# if __name__ == "__main__":
#     data_obj = DataIngestion()
#     data = data_obj.initiate_data_ingestion()
#     # data_transformed = pd.read_csv("artifacts/data_transformed.csv")
#     DataTransformer = DataTransformation()
#     data_transformed = DataTransformer.initialize_data_transformation(data)
#
#     print(data.shape, data_transformed.shape)
