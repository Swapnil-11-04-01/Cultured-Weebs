import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.utils import load_object
from PIL import Image
from io import BytesIO
import requests
import base64


def recommend(anime, data, data_transformed, similarity_matrix_bow):
    result = PredictPipeline.predict(anime, data, data_transformed, similarity_matrix_bow)
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


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string});
        background-size: cover;
        backdrop-filter: blur(500px);
    }}
            .stApp::before {{
            content: "";
            background-color: rgba(0, 0, 0, 0.85);
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
        }}
    </style>
    """,
        unsafe_allow_html=True
    )


def display_recommendations(anime, data, data_transformed, similarity_matrix_bow):
    print(anime)
    titles, posters, synopses, genres, studios, types, num_episodes, statuses, scores, start_dates, end_dates = recommend(
        anime, data, data_transformed, similarity_matrix_bow)
    for i in range(11):
        if i == 0:
            _, pic, info = st.columns([2, 3.5, 4])
            with pic:
                url = posters[i]
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                # Resize the image
                resized_image = image.resize((480, 600))
                st.image(resized_image)
            with info:
                st.title(titles[i])
                st.markdown(f'#### **More Info**')
                st.markdown(f'- **TYPE :** *{types[i]}*')
                st.markdown(f'- **EPISODES :** *{num_episodes[i]}*')
                st.markdown(f'- **GENRES :** *{genres[i]}*')
                st.markdown(f'- **STUDIO :** *{studios[i]}*')
                st.markdown(f'- **RATING :** *{scores[i]}*')
                st.markdown(f'- **STATUS :** *{statuses[i]}*')
                if statuses[i] == 'Currently Airing':
                    st.markdown(f'- **AIRED :** *{start_dates[i][:10]}*   to   -')
                else:
                    st.markdown(f'- **AIRED :** *{start_dates[i][:10]}*   to   *{end_dates[i][:10]}*')
            _, synopsis, _ = st.columns([1.5, 6, 1.5])
            with synopsis:
                st.text_area("Description :",
                             key=f"ta_{i}",
                             height=200,
                             value=synopses[i])
                st.title(" ")
                st.title(" ")
                st.title(" ")
                st.title(" ")
            st.markdown("#### Recommendations:")
            st.title(" ")

        else:
            index, pic, title_synopsis = st.columns([0.3, 2, 4])
            with index:
                st.markdown(f"#### {i}")
            with pic:
                url = posters[i]
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                # Resize the image
                resized_image = image.resize((400, 500))
                st.image(resized_image)
            with title_synopsis:
                st.title(titles[i])
                synopsis, rest = st.columns([3, 2])
                with synopsis:
                    st.text_area("Description :",
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
                    if statuses[i] == 'Currently Airing':
                        st.markdown(f'- **AIRED :** *{start_dates[i][:10]}*   to   -')
                    else:
                        st.markdown(f'- **AIRED :** *{start_dates[i][:10]}*   to   *{end_dates[i][:10]}*')
                    st.title(" ")
                    st.title(" ")
                    st.title(" ")


def application():
    st.set_page_config(page_title="Cultured Weebs", page_icon=":smiley:", layout="wide", menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'Swapnil Sharma - https://swapnil-11-04-01.github.io/Personal-Portfolio/'})

    try:
        data = pd.read_csv("artifacts/anime.csv", delimiter='\t')
        data_transformed = pd.read_csv("artifacts//data_transformed.csv")
        # similarity_matrix_tfidf = load_object("artifacts/similarity_matrix_tfidf.pkl")
        similarity_matrix_bow = load_object("artifacts/similarity_matrix_bow.pkl")
        print('try')

    except:
        TrainPipeline.initiate_train_pipeline()

        data = pd.read_csv("artifacts/anime.csv", delimiter='\t')
        data_transformed = pd.read_csv("artifacts//data_transformed.csv")
        # similarity_matrix_tfidf = load_object("artifacts/similarity_matrix_tfidf.pkl")
        similarity_matrix_bow = load_object("artifacts/similarity_matrix_bow.pkl")
        print('except')

    add_bg_from_local('templates/wallpaper/1.jpg')

    st.write(
        '# <div style="text-align: center; font-size: 2.7em; font-family: cursive; margin-bottom: 0.7em"><I>'
        '\`Cultured Weebs\`</div>',
        unsafe_allow_html=True)

    st.subheader('Anime Recommendation System  : `7.5+ MAL Ratings`')
    anime_list = data_transformed['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        anime_list
    )

    if st.button('Show Recommendation'):
        display_recommendations(selected_movie, data, data_transformed, similarity_matrix_bow)


if __name__ == '__main__':
    application()
