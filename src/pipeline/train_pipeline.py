from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import cosine_similarity_matrix
from src.utils import load_object, save_object
from src.exception import CustomException
import sys
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        pass

    @staticmethod
    def initiate_train_pipeline():
        logging.info("Initiating data ingestion")
        try:
            data_obj = DataIngestion()
            data = data_obj.initiate_data_ingestion()

            DataTransformer = DataTransformation()
            vector, name = DataTransformer.initialize_data_transformation(data)

            # vector_tfidf = load_object("artifacts/vector_tfidf.pkl")
            # vector_bow = load_object("artifacts/vector_bow.pkl")

            similarity_matrix_tfidf = cosine_similarity_matrix(vector, name)
            # similarity_matrix_bow = cosine_similarity_matrix(vector_bow, "bow")

            # save_object("artifacts/similarity_matrix_tfidf.pkl", similarity_matrix_tfidf)
            # save_object("artifacts/similarity_matrix_bow.pkl", similarity_matrix_bow)

            logging.info('File creation complete')
        except Exception as e:
            raise CustomException(e, sys)
