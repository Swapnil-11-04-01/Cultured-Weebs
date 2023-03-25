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
DataTransformer = DataTransformation()
data_transformed = DataTransformer.initialize_data_transformation(data)
vector_tfidf = load_object("artifacts/vector_tfidf.pkl")
vector_bow = load_object("artifacts/vector_bow.pkl")
similarity_matrix_tfidf = cosine_similarity_matrix(vector_tfidf)
similarity_matrix_bow = cosine_similarity_matrix(vector_bow)
print('\n\nFile creation complete\n\n')


def recommend(anime):
    result = PredictPipeline.predict(anime, data, data_transformed, similarity_matrix_tfidf)
    # print(result['title'])
    return (result['title'],
            result['pic_url'],
            result['synopsis'],
            result['genres'],
            result['studio'],
            result['type'],
            result['num_episodes'],
            result['status'],
            result['score'],
            result['start_date'],
            result['end_date'])


st.set_page_config(page_title="Cultured Weebs", page_icon=":smiley:", layout="wide")

st.header('Cultured Weebs')
st.subheader('Anime Recommender System')
anime_list = data_transformed['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    anime_list
)

if st.button('Show Recommendation'):
    titles, posters, synopses, genres, studios, types, num_episodes, statuses, scores, start_dates, end_dates = recommend(selected_movie)
    for i in range(15):
        pic, title_synopsis = st.columns([2, 5])
        with pic:
            url = posters[i]
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            # Resize the image
            resized_image = image.resize((400, 500))
            st.image(resized_image)
        with title_synopsis:
            st.title(titles[i])
            synopsis, rest = st.columns([2, 2])
            with synopsis:
                st.text_area(label="Synopsis",
                             key=f"ta_{i}",
                             height=320,
                             value=synopses[i])
            with rest:
                st.markdown(f'### **More Info**')
                st.markdown(f'- **TYPE :** *{types[i]}*')
                st.markdown(f'- **EPISODES :** *{num_episodes[i]}*')
                st.markdown(f'- **GENRES :** *{genres[i]}*')
                st.markdown(f'- **STUDIO :** *{studios[i]}*')
                st.markdown(f'- **RATING :** *{scores[i]}*')
                st.markdown(f'- **STATUS :** *{statuses[i]}*')
                st.markdown(f'- **AIRED :** *{start_dates[i]}* to *{end_dates[i]}*')