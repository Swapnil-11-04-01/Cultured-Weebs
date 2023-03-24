import os
import sys
from src.exception import CustomException
import dill
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('stopwords')


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def pre_text_transformer(obj):
    obj = obj.replace(" ", "_")
    obj = obj.split('|')
    return obj


def dataframe_modifier(dataframe):
    required_cols = ["anime_id", "title", "synopsis", "type", "studios", "genres"]
    df_new = dataframe[required_cols]
    df_new.dropna(inplace=True)
    df_new.reset_index(inplace=True, drop=True)

    for col in required_cols[2:]:
        if col == "synopsis":
            df_new[col] = df_new[col].apply(lambda x: x.split())
        else:
            df_new[col] = df_new[col].apply(pre_text_transformer)

    df_new["cat_vector"] = df_new["synopsis"] + df_new["type"] + df_new["studios"] + df_new["genres"]
    df_new["cat_vector"] = df_new["cat_vector"].apply(lambda x: " ".join(x))

    df_new = df_new[["anime_id", "title", "cat_vector"]]
    return df_new


def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    tokens = re.sub(r'[^\w\s]', '', text).lower()
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]

    stemmed_words = " ".join([stemmer.stem(word) for word in tokens])
    return stemmed_words


def cosine_similarity_matrix(matrix):
    similarity_matrix = cosine_similarity(matrix)
    save_object("artifacts/similarity_matrix.pkl", similarity_matrix)
    return similarity_matrix


def main_pic_linker(df, anime_id):
    url = df.loc[df['anime_id'] == anime_id, ["anime_url"]].iloc[0, 0]
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    image_tag = soup.find('img', {'class': "ac"})
    image_url = image_tag['data-src']
    return image_url
