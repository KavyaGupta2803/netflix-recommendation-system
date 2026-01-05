import streamlit as st
import pandas as pd
import requests
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Netflix Movie Recommendation System",
    layout="wide"
)

# =====================================================
# THEME (NETFLIX STYLE)
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.92), rgba(0,0,0,0.92));
}
.block-container {
    background-color: rgba(20,20,20,0.96);
    padding: 2.5rem;
    border-radius: 16px;
}
h1, h2, h3, h4, h5, h6, p, label, span {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("clean_data.csv")

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# =====================================================
# TMDB CONFIG (SECRET)
# =====================================================
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# =====================================================
# BLACK POSTER PLACEHOLDER
# =====================================================
def poster_placeholder():
    st.markdown(
        """
        <div style="
            width:100%;
            height:580px;
            background-color:black;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            color:white;
            font-size:18px;
            border-radius:8px;
        ">
            <div>
                🎬<br>
                Poster<br>
                Not Available
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# POSTER FETCH (SAFE + TYPE AWARE)
# =====================================================
@st.cache_data(show_spinner=False)
def fetch_poster(title, content_type):
    try:
        if not TMDB_API_KEY:
            return None

        if content_type == "Movie":
            url = "https://api.themoviedb.org/3/search/movie"
        else:
            url = "https://api.themoviedb.org/3/search/tv"

        params = {
            "api_key": TMDB_API_KEY,
            "query": title
        }

        response = requests.get(url, params=params, timeout=5)
        if response.status_code != 200:
            return None

        data = response.json()
        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

        return None

    except requests.exceptions.RequestException:
        return None

# =====================================================
# COSINE SIMILARITY (DYNAMIC + CACHED)
# =====================================================
@st.cache_data(show_spinner=True)
def compute_cosine_similarity(data):
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )
    tfidf_matrix = tfidf.fit_transform(data['content'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_cosine_similarity(df)

# =====================================================
# RECOMMENDATION LOGIC (TITLE BASED)
# =====================================================
def recommend_by_title(title, n=6):
    idx = indices[title]
    input_type = df.loc[idx, 'type']

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    filtered = [
        i for i in sim_scores
        if df.iloc[i[0]]['type'] == input_type and i[0] != idx
    ]

    rec_indices = [i[0] for i in filtered[:n]]
    return df.iloc[rec_indices]

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.header("🎯 Recommendation Settings")

basis = st.sidebar.radio(
    "Recommendation Basis",
    ["Title"]
)

content_type = st.sidebar.selectbox(
    "Content Type",
    ["Movie", "TV Show"]
)

# =====================================================
# MAIN TITLE
# =====================================================
st.title("🎬 Netflix Movie Recommendation System")

# =====================================================
# TITLE MODE
# =====================================================
if basis == "Title":

    filtered_titles = df[df['type'] == content_type]['title'].sort_values()

    selected_title = st.selectbox(
        "🔍 Search & select a title",
        filtered_titles
    )

    movie = df[df['title'] == selected_title].iloc[0]

    col1, col2 = st.columns([1, 2])

    with col1:
        poster = fetch_poster(movie['title'], movie['type'])
        if poster:
            st.image(poster, use_container_width=True)
        else:
            poster_placeholder()

    with col2:
        st.markdown(f"## 🎬 {movie['title']}")
        st.write(f"**Type:** {movie['type']}")
        st.write(f"**Release Year:** {movie['release_year']}")
        st.write(f"**Rating:** {movie['rating']}")
        st.write(f"**Duration:** {movie['duration']}")
        st.write(f"**Genres:** {movie['listed_in']}")
        st.write(movie['description'])

    st.markdown("## 🔥 Recommended for You")

    recs = recommend_by_title(selected_title)
    cols = st.columns(3)

    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 3]:
            poster = fetch_poster(row['title'], row['type'])
            if poster:
                st.image(poster, use_container_width=True)
            else:
                poster_placeholder()
            st.caption(f"{row['title']} ({row['release_year']})")


st.caption("Posters fetched from TMDB • Some titles may not have posters available")
