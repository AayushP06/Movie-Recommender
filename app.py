import os
import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ----------------------------
# Load and preprocess data
# ----------------------------
@st.cache_data
def load_data():

    base_path = os.path.dirname(__file__)

    local_movies_path = os.path.join(base_path,"data", "tmdb_5000_movies.csv")
    local_credits_path = os.path.join(base_path, "data", "tmdb_5000_credits.csv")

    # Online URLs (replace FILE_ID with your actual file IDs)
    online_movies_url = "https://drive.google.com/uc?id=1eyyqWpMl9_4_F-GDC0amtI-RvQwaPaLq"
    online_credits_url = "https://drive.google.com/uc?id=1ulU3ghueTbpDhCGHbZtmPrLg3i_9qPwn"


    try:
        # Try loading local files first
        movies = pd.read_csv(local_movies_path)
        credits = pd.read_csv(local_credits_path)
        st.write("Loaded dataset locally ✅")
    except FileNotFoundError:
        # Fallback to Google Drive
        movies = pd.read_csv(online_movies_url)
        credits = pd.read_csv(online_credits_url)
        st.write("Loaded dataset from Google Drive ✅")

    # Compute similarity or return placeholder
    similarity = None  # Replace with your similarity computation if needed


    # Merge both datasets
    movies = movies.merge(credits, on="title")

    # Keep only necessary columns
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    # Convert stringified lists into Python objects
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i["name"])
        return L

    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                L.append(i["name"])
        return L

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(lambda x: convert(x)[:3])  # top 3 actors
    movies["crew"] = movies["crew"].apply(fetch_director)

    # Create tags
    movies["tags"] = (
        movies["overview"].apply(lambda x: x.split())
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )
    new_df = movies[["movie_id", "title", "tags"]]

    # Convert tags into strings
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())

    # Vectorize
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    return new_df, similarity


movies, similarity = load_data()

# ----------------------------
# Recommendation Function
# ----------------------------
def recommend(movie):
    if movie not in movies["title"].values:
        return ["Movie not found in dataset."]
    idx = movies[movies["title"] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1]
    )[1:6]
    return [movies.iloc[i[0]].title for i in distances]


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎥 Movie Recommender System")
st.markdown("#### Find movies similar to your favorites! 🍿")

movie_name = st.selectbox("🎬 Select a movie:", movies["title"].values)

if st.button("✨ Recommend Movies"):
    recommendations = recommend(movie_name)
    st.markdown("---")
    st.subheader(f"Recommended Movies for **{movie_name}**:")
    for m in recommendations:
        st.markdown(f"👉 {m}")
