import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import cosine_similarity_matrix
from src.utils import load_object
from PIL import Image
from io import BytesIO
import requests

data_obj = DataIngestion()
data = data_obj.initiate_data_ingestion()

try:
    data_transformed = pd.read_csv("artifacts/data_transformed.csv")
    similarity_matrix = load_object(file_path="artifacts/similarity_matrix.pkl")
except:
    data_obj = DataIngestion()
    data = data_obj.initiate_data_ingestion()
    DataTransformer = DataTransformation()
    data_transformed = DataTransformer.initialize_data_transformation(data)
    vector = load_object("artifacts/vector.pkl")
    similarity_matrix = cosine_similarity_matrix(vector)


def recommend(anime):
    result = PredictPipeline.predict(anime, data, data_transformed, similarity_matrix)
    print(result['title'])
    return result['title'], result['pic_url'], result['synopsis']


st.set_page_config(page_title="My App", page_icon=":smiley:", layout="wide")

st.header('Anime Recommender System')
anime_list = data_transformed['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    anime_list
)

if st.button('Show Recommendation'):
    recommended_anime_names, recommended_anime_posters, recommended_anime_synopsis = recommend(selected_movie)
    for i in range(5):
        pic, title_synopsis = st.columns([2, 5])
        with pic:
            url = recommended_anime_posters[i]
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            # Resize the image
            resized_image = image.resize((400, 500))
            st.image(resized_image)
        with title_synopsis:
            st.title(recommended_anime_names[i])
            st.text_area(label="Synopsis",
                         key=f"ta_{i}",
                         height=300,
                         value=recommended_anime_synopsis[i])
