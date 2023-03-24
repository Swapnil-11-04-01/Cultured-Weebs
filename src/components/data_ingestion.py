import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.utils import cosine_similarity_matrix
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
            data = pd.read_csv(self.ingestion_config.data_path, delimiter=',')
            logging.info(f"Successfully completed data ingestion")
            return data
        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
    # data_obj = DataIngestion()
    # dataframe = data_obj.initiate_data_ingestion()
    #
    # data_transformation = DataTransformation()
    # transformed_dataframe, vector = data_transformation.initialize_data_transformation(dataframe)
    #
    # similarity = cosine_similarity_matrix(vector)
    # print(similarity)


    # print(madel_name, result)