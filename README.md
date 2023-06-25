# Cultured Weebs
 
Cultured Weebs is an Anime Recommendation System which recommends 10 similiar animes to the the selected one.

Link - http://culturedweebs-env-1.eba-dbmc36mu.us-east-1.elasticbeanstalk.com/](https://cultured-weebs.onrender.com/



## Resources Used:-
- **Programming language** -> `Python`
- **Web-Framework** -> `Streamlit`
- **Data Source Website** -> https://myanimelist.net/
- **Recommendation technique** -> `Content Based Filtering`
- **Text Vectorisation Technique** -> `Bag Of Words` & `Tf-idf` (Preferred BOW for its lower sized matrix but similar results with Tf-idf)
- **Similarity distance algorithm** -> `Cosine Similarity` 



## Steps To Run Locally (Windows):
1. Clone the repository: 
   ```
   git clone https://github.com/Swapnil-11-04-01/Cultured-Weebs.git
   ```
2. Enter the directory:
   ```
   cd Cultured-Weebs
   ```
3. Create Virtual Environment and then activate it:
   ```
   python -m venv my_venv
   ```
4. Activate Environment:
   ```
   ./my_venv/Scripts/activate
   ```
5. Intall Requirements:
   ```
   pip install -r requirements.txt
   ```
6. Run the Application:
   ```
   streamlit run application.py
   ```
