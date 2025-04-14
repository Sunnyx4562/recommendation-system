import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stylable_container import stylable_container
import random
import base64

TMDB_API_KEY = 'cbe1f8e54cabf33493acde2f3a08c4ae'

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide", page_icon="🎥")

# UI Styling
st.markdown("""
    <style>
    .block-container {
        padding: 2rem 4rem;
    }
    .stSelectbox, .stTextInput, .stButton {
        font-size: 16px;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    .movie-poster {
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
        margin-bottom: 10px;
    }
    .movie-poster:hover {
        transform: scale(1.03);
    }
    .top-cast-img {
        border-radius: 15px;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stMarkdown, .stTextInput, .stSelectbox, .stButton {{
            background-color: rgba(0, 0, 0, 0.6) !important;
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("assets/backgroundimg.jpg")

@st.cache_data
def load_data(path='main_data.csv'):
    df = pd.read_csv(path)
    df = df[['movie_title', 'genres', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']]
    df.dropna(inplace=True)
    df['genres'] = df['genres'].apply(lambda g: g.split('|')[0])

    def combine_features(row):
        genre_weighted = (row['genres'] + " ") * 3
        return f"{row['movie_title']} {genre_weighted} {row['director_name']} {row['actor_1_name']} {row['actor_2_name']} {row['actor_3_name']}"

    df['combined_features'] = df.apply(combine_features, axis=1)
    return df

@st.cache_resource
def create_similarity_matrix(df):
    vectorizer = CountVectorizer(stop_words='english')
    count_matrix = vectorizer.fit_transform(df['combined_features'])
    return cosine_similarity(count_matrix)

def get_recommendations_by_title(title, df, sim_matrix):
    if title not in df['movie_title'].values:
        return pd.DataFrame()
    idx = df[df['movie_title'] == title].index[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    return df.iloc[[i[0] for i in sim_scores]]

def get_recommendations_by_genre(genre, df):
    subset = df[df['genres'] == genre]
    return subset.sample(min(10, len(subset)))

def get_recommendations_by_director(director, df):
    subset = df[df['director_name'] == director]
    return subset.sample(min(10, len(subset)))

def get_recommendations_by_actor(actor, df):
    mask = (df['actor_1_name'] == actor) | (df['actor_2_name'] == actor) | (df['actor_3_name'] == actor)
    subset = df[mask]
    return subset.sample(min(10, len(subset)))

def get_movie_details_tmdb(title):
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        search_res = requests.get(search_url).json()
        if not search_res.get("results"):
            return None

        movie = search_res['results'][0]
        movie_id = movie['id']
        poster_path = movie.get('poster_path')
        rating = movie.get('vote_average')
        overview = movie.get('overview', 'No description available.')
        release = movie.get('release_date', 'N/A')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

        detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        detail_res = requests.get(detail_url).json()
        genres = ', '.join([g['name'] for g in detail_res.get('genres', [])])
        runtime = detail_res.get('runtime', 'N/A')
        status = detail_res.get('status', 'N/A')

        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
        credits_res = requests.get(credits_url).json()
        cast = credits_res.get('cast', [])[:5]
        cast_info = []
        for actor in cast:
            actor_name = actor.get('name')
            profile_path = actor.get('profile_path')
            actor_img = f"https://image.tmdb.org/t/p/w200{profile_path}" if profile_path else None
            actor_id = actor.get('id')
            cast_info.append({'name': actor_name, 'image': actor_img, 'id': actor_id})

        trailer_url = None
        videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
        videos_res = requests.get(videos_url).json()
        for video in videos_res.get('results', []):
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                trailer_url = f"https://www.youtube.com/watch?v={video['key']}"
                break

        return {
            'title': movie['title'],
            'poster': poster_url,
            'rating': rating,
            'overview': overview,
            'genres': genres,
            'release_date': release,
            'runtime': runtime,
            'status': status,
            'cast': cast_info,
            'trailer': trailer_url
        }
    except:
        return None

def get_actor_details_tmdb(actor_id):
    try:
        actor_url = f"https://api.themoviedb.org/3/person/{actor_id}?api_key={TMDB_API_KEY}"
        actor_res = requests.get(actor_url).json()

        if actor_res:
            name = actor_res['name']
            biography = actor_res.get('biography', 'Biography not available.')
            birth_date = actor_res.get('birthday', 'N/A')
            place_of_birth = actor_res.get('place_of_birth', 'N/A')
            profile_path = actor_res.get('profile_path', None)
            profile_img = f"https://image.tmdb.org/t/p/w500{profile_path}" if profile_path else None

            return {
                'name': name,
                'biography': biography,
                'birth_date': birth_date,
                'place_of_birth': place_of_birth,
                'profile_img': profile_img
            }
        return None
    except Exception as e:
        print(f"Error fetching actor details: {e}")
        return None

st.title("🎥🍿 Movie Recommender System")
df = load_data()
sim_matrix = create_similarity_matrix(df)

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

search_mode = st.selectbox("🔍 Choose Recommendation Type", [
    "By Movie Title", "By Genre", "By Director Name", "By Actor Name"])

selection = None
if search_mode == "By Movie Title":
    selection = st.selectbox("🎬 Select a Movie Title", sorted(df['movie_title'].unique()))
elif search_mode == "By Genre":
    selection = st.selectbox("🎭 Select a Genre", sorted(df['genres'].unique()))
elif search_mode == "By Director Name":
    selection = st.selectbox("🎬 Select a Director", sorted(df['director_name'].unique()))
elif search_mode == "By Actor Name":
    actors = pd.unique(df[['actor_1_name', 'actor_2_name', 'actor_3_name']].values.ravel())
    selection = st.selectbox("🧑‍🎤 Select an Actor", sorted(actors))

if st.button("🔍 Recommend"):
    st.session_state.selected_movie = None
    if search_mode == "By Movie Title":
        st.session_state.recommendations = get_recommendations_by_title(selection, df, sim_matrix)
    elif search_mode == "By Genre":
        st.session_state.recommendations = get_recommendations_by_genre(selection, df)
    elif search_mode == "By Director Name":
        st.session_state.recommendations = get_recommendations_by_director(selection, df)
    elif search_mode == "By Actor Name":
        st.session_state.recommendations = get_recommendations_by_actor(selection, df)

if st.session_state.recommendations is not None:
    st.subheader("🎯 Recommended Movies:")
    rows = [st.columns(5), st.columns(5)]
    for i, (_, row) in enumerate(st.session_state.recommendations.iterrows()):
        col = rows[i // 5][i % 5]
        details = get_movie_details_tmdb(row['movie_title'])
        with col:
            if details and details['poster']:
                st.markdown(
                    f'<img src="{details["poster"]}" class="movie-poster" width="80%">',
                    unsafe_allow_html=True
                )
            st.markdown(f"**{row['movie_title']}**")
            st.markdown(f"⭐ {details['rating']}/10" if details else "")
            if st.button("More Details", key=f"info_{i}_{row['movie_title']}"):
                st.session_state.selected_movie = row['movie_title']

if st.session_state.selected_movie:
    details = get_movie_details_tmdb(st.session_state.selected_movie)
    if details:
        st.markdown(f"## 🎬 {details['title']}")
        col1, col2 = st.columns([1, 2])
        with col1:
            if details['poster']:
                st.image(details['poster'], width=300)
        with col2:
            st.markdown(f"**Overview:** {details['overview']}")
            st.markdown(f"**Rating:** ⭐ {details['rating']}/10")
            st.markdown(f"**Genres:** {details['genres']}")
            st.markdown(f"**Release Date:** 📅 {details['release_date']}")
            st.markdown(f"**Runtime:** ⏱️ {details['runtime']} min")
            st.markdown(f"**Status:** 🎬 {details['status']}")
            if details['trailer']:
                st.markdown(f"[▶️ Watch Trailer]({details['trailer']})", unsafe_allow_html=True)

        st.markdown("### 👥 Top Cast")
        cast_cols = st.columns(len(details['cast']))
        for i, actor in enumerate(details['cast']):
            with cast_cols[i]:
                if actor['image']:
                    st.markdown(
                        f'''
                        <div style="text-align:left">
                            <img src="{actor["image"]}" class="top-cast-img" width="100">
                            <div style="padding-top:4px;"><b>{actor["name"]}</b></div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                if st.button(f"Know more", key=f"actor_info_{i}"):
                    actor_details = get_actor_details_tmdb(actor['id'])
                    if actor_details:
                        st.markdown(f"**{actor_details['name']}**")
                        st.markdown(f"**Born:** {actor_details['birth_date']} in {actor_details['place_of_birth']}")
                        st.markdown(f"**Biography:** {actor_details['biography']}")
                        if actor_details['profile_img']:
                            st.image(actor_details['profile_img'], width=300)
