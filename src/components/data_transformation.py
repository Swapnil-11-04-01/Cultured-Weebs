import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.utils import dataframe_modifier
from src.utils import preprocess_text
from src.exception import CustomException
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class DataTransformationConfig:
    transformer_obj_file_path = os.path.join("artifacts", "transformer.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()

    # def get_data_transformer_object(self):
    #     try:
    #         text_preprocessor = make_pipeline(
    #             FunctionTransformer(preprocess_text, validate=False),
    #             TfidfVectorizer()
    #         )
    #
    #         preprocessor = ColumnTransformer(
    #             [
    #                 ("preprocess_text", text_preprocessor, ['cat_vector'])
    #             ]
    #         )
    #
    #         pipeline = make_pipeline(
    #             preprocessor,
    #             # Add other pipeline steps as needed
    #         )
    #
    #         save_object(
    #             file_path=self.data_transformer_config.transformer_obj_file_path,
    #             obj=preprocessor
    #         )
    #
    #         return pipeline
    #     except Exception as e:
    #         raise CustomException(e, sys)

    @staticmethod
    def initialize_data_transformation(data):
        try:
            logging.info("Initializing data transformation")

            data = dataframe_modifier(data)
            data['cat_vector'] = data['cat_vector'].apply(preprocess_text)
            # preprocessor_obj = self.get_data_transformer_object()
            # print(data.head())

            # tfidf_matrix = preprocessor_obj.fit_transform(data)

            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(data['cat_vector'])
            vector = tfidf_matrix.toarray()

            return data, vector
        except Exception as e:
            raise CustomException(e, sys)
