import sys
import os
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from src.exception import CustomException
from src.utils import save_object
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')


@dataclass
class DataTransformationConfig:
    transformer_obj_file_path = os.path.join("artifacts", "transformer.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()

    @staticmethod
    def preprocess_text(text):
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        tokens = re.sub(r'[^\w\s]', '', text).lower()
        tokens = nltk.word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in stop_words]
        stemmed_words = " ".join([stemmer.stem(word) for word in tokens])
        return stemmed_words

    @staticmethod
    def data_cleaner(data):
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        data = data = data[(data['score'] >= 6.5) & (data['synopsis'].str.split().str.len() >= 20)]
        data.reset_index(inplace=True, drop=True)
        return data

    @staticmethod
    def pre_text_transformer(obj):
        obj = obj.replace(" ", "_")
        obj = obj.split('|')
        return obj

    # @staticmethod
    def dataframe_modifier(self, dataframe):
        required_cols = ["anime_id", "title", "synopsis", "type", "studios", "genres"]
        df_new = dataframe[required_cols]
        for col in required_cols[2:]:
            if col == "synopsis":
                df_new[col] = df_new[col].apply(lambda x: x.split())
            else:
                df_new[col] = df_new[col].apply(self.pre_text_transformer)
        df_new["cat_vector"] = df_new["synopsis"] + df_new["type"] + df_new["studios"] + df_new["genres"]
        df_new["cat_vector"] = df_new["cat_vector"].apply(lambda x: " ".join(x))
        df_new = df_new[["anime_id", "title", "cat_vector"]]
        return df_new

    # @staticmethod
    def initialize_data_transformation(self, data):
        try:
            logging.info("Initializing data transformation")
            data = self.data_cleaner(data)
            data = self.dataframe_modifier(data)
            data['cat_vector'] = data['cat_vector'].apply(self.preprocess_text)
            data.to_csv("artifacts/data_transformed.csv", index=False, header=True)
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(data['cat_vector'])
            vector = tfidf_matrix.toarray()
            save_object("artifacts/vector.pkl", vector)
            return data
        except Exception as e:
            raise CustomException(e, sys)
