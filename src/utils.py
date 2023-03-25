import os
import sys
from src.exception import CustomException
import dill
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup


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


def cosine_similarity_matrix(matrix, name):
    similarity_matrix = cosine_similarity(matrix)
    save_object(f"artifacts/similarity_matrix_{name}.pkl", similarity_matrix)


def main_pic_linker(df, anime_id):
    url = df.loc[df['anime_id'] == anime_id, ["anime_url"]].iloc[0, 0]
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    image_tag = soup.find('img', {'class': "ac"})
    image_url = image_tag['data-src']
    return image_url
