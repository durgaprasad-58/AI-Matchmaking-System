import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load Data
# -----------------------------
users = pd.read_csv("user.csv")
feedback = pd.read_csv("feedback.csv")

# -----------------------------
# NLP Processing
# -----------------------------
users["combined_text"] = (
    users["professional_summary"].fillna("")
    + " "
    + users["about_me"].fillna("")
)

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    users["combined_text"]
)

# -----------------------------
# MBTI Rules
# -----------------------------
MBTI_SCORES = {
    ("INTJ", "ENFP"): 100,
    ("ENFP", "INTJ"): 100,
    ("INFJ", "ENTP"): 95,
    ("ENTP", "INFJ"): 95
}

def mbti_score(a, b):
    return MBTI_SCORES.get((a, b), 60)

def location_score(a, b):
    if a == b:
        return 100
    return 50

# Initial weights
w1 = 0.5
w2 = 0.3
w3 = 0.2

# -----------------------------
# Compatibility Function
# -----------------------------
def compatibility(user1, user2):

    idx1 = users[
        users["user_id"] == user1
    ].index[0]

    idx2 = users[
        users["user_id"] == user2
    ].index[0]

    text_score = cosine_similarity(
        tfidf_matrix[idx1],
        tfidf_matrix[idx2]
    )[0][0] * 100

    mbti = mbti_score(
        users.loc[idx1, "MBTI"],
        users.loc[idx2, "MBTI"]
    )

    location = location_score(
        users.loc[idx1, "location"],
        users.loc[idx2, "location"]
    )

    total = (
        w1 * text_score
        + w2 * mbti
        + w3 * location
    )

    return round(total, 2)

# -----------------------------
# Train Feedback Model
# -----------------------------
X = []
y = []

for _, row in feedback.iterrows():

    idx1 = users[
        users["user_id"] == row["user_id"]
    ].index[0]

    idx2 = users[
        users["user_id"] == row["matched_user_id"]
    ].index[0]

    text_sim = cosine_similarity(
        tfidf_matrix[idx1],
        tfidf_matrix[idx2]
    )[0][0]

    mbti = mbti_score(
        users.loc[idx1, "MBTI"],
        users.loc[idx2, "MBTI"]
    ) / 100

    X.append([text_sim, mbti])
    y.append(row["action"])

model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Top Matches
# -----------------------------
def top_matches(user_id):

    scores = []

    for uid in users["user_id"]:

        if uid != user_id:

            score = compatibility(
                user_id,
                uid
            )

            scores.append(
                [uid, score]
            )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scores[:5]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="AI Matchmaking System",
    layout="wide"
)

st.title("💖 AI Matchmaking Recommendation System")

st.write(
    "Find the top 5 compatible matches."
)

selected_user = st.selectbox(
    "Select User ID",
    users["user_id"]
)

if st.button("Find Matches"):

    matches = top_matches(selected_user)

    result = pd.DataFrame(
        matches,
        columns=[
            "Matched User",
            "Compatibility Score"
        ]
    )

    st.success(
        f"Top Matches for {selected_user}"
    )

    st.dataframe(
        result,
        use_container_width=True
    )

    st.bar_chart(
        result.set_index(
            "Matched User"
        )
    )

    user_info = users[
        users["user_id"] == selected_user
    ]

    st.subheader(
        "Selected User Profile"
    )

    st.dataframe(
        user_info[
            [
                "name",
                "location",
                "profession",
                "MBTI"
            ]
        ]
    )