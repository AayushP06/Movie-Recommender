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
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")

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
