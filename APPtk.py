import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

# =========================
# Load Data
# =========================

users = pd.read_csv("user.csv")
feedback = pd.read_csv("feedback.csv")

# =========================
# NLP Processing
# =========================

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

# =========================
# MBTI Scores
# =========================

MBTI_SCORES = {
    ("INTJ","ENFP"):100,
    ("ENFP","INTJ"):100,
    ("INFJ","ENTP"):95,
    ("ENTP","INFJ"):95
}

def mbti_score(a,b):
    return MBTI_SCORES.get((a,b),60)

def location_score(a,b):

    if a == b:
        return 100

    return 50

# =========================
# Compatibility Weights
# =========================

w1 = 0.5
w2 = 0.3
w3 = 0.2

# =========================
# Compatibility Function
# =========================

def compatibility(user1,user2):

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
        users.loc[idx1,"MBTI"],
        users.loc[idx2,"MBTI"]
    )

    location = location_score(
        users.loc[idx1,"location"],
        users.loc[idx2,"location"]
    )

    total = (
        w1 * text_score +
        w2 * mbti +
        w3 * location
    )

    return round(total,2)

# =========================
# Top Matches
# =========================

def top_matches(user_id):

    scores = []

    for uid in users["user_id"]:

        if uid != user_id:

            score = compatibility(
                user_id,
                uid
            )

            scores.append(
                (uid,score)
            )

    scores.sort(
        key=lambda x:x[1],
        reverse=True
    )

    return scores[:5]

# =========================
# Train Feedback Model
# =========================

X = []
y = []

for _, row in feedback.iterrows():

    idx1 = users[
        users["user_id"] ==
        row["user_id"]
    ].index[0]

    idx2 = users[
        users["user_id"] ==
        row["matched_user_id"]
    ].index[0]

    text_sim = cosine_similarity(
        tfidf_matrix[idx1],
        tfidf_matrix[idx2]
    )[0][0]

    mbti = mbti_score(
        users.loc[idx1,"MBTI"],
        users.loc[idx2,"MBTI"]
    ) / 100

    X.append([
        text_sim,
        mbti
    ])

    y.append(
        row["action"]
    )

model = LinearRegression()
model.fit(X,y)

# =========================
# GUI Functions
# =========================

def find_matches():

    selected_user = user_combo.get()

    if selected_user == "":
        messagebox.showerror(
            "Error",
            "Select User ID"
        )
        return

    for row in tree.get_children():
        tree.delete(row)

    matches = top_matches(
        selected_user
    )

    for uid, score in matches:

        tree.insert(
            "",
            tk.END,
            values=(uid,score)
        )

    weight_label.config(
        text=
        f"Text Weight: {round(model.coef_[0],3)}      "
        f"MBTI Weight: {round(model.coef_[1],3)}"
    )

# =========================
# Main Window
# =========================

root = tk.Tk()

root.title(
    "AI Matchmaking System"
)

root.geometry(
    "900x600"
)

root.configure(
    bg="#f5f5f5"
)

# =========================
# Heading
# =========================

heading = tk.Label(
    root,
    text="AI Matchmaking Recommendation System",
    font=("Arial",20,"bold"),
    bg="#f5f5f5",
    fg="darkblue"
)

heading.pack(pady=15)

# =========================
# User Selection
# =========================

frame = tk.Frame(
    root,
    bg="#f5f5f5"
)

frame.pack()

tk.Label(
    frame,
    text="Select User ID:",
    font=("Arial",12),
    bg="#f5f5f5"
).grid(
    row=0,
    column=0,
    padx=10
)

user_combo = ttk.Combobox(
    frame,
    width=20,
    values=list(users["user_id"])
)

user_combo.grid(
    row=0,
    column=1,
    padx=10
)

btn = tk.Button(
    frame,
    text="Find Matches",
    font=("Arial",12,"bold"),
    bg="green",
    fg="white",
    command=find_matches
)

btn.grid(
    row=0,
    column=2,
    padx=10
)

# =========================
# Results Table
# =========================

columns = (
    "Matched User",
    "Compatibility Score"
)

tree = ttk.Treeview(
    root,
    columns=columns,
    show="headings",
    height=10
)

tree.heading(
    "Matched User",
    text="Matched User"
)

tree.heading(
    "Compatibility Score",
    text="Compatibility Score (%)"
)

tree.column(
    "Matched User",
    width=200
)

tree.column(
    "Compatibility Score",
    width=200
)

tree.pack(
    pady=20
)

# =========================
# Model Weights
# =========================

weight_label = tk.Label(
    root,
    text="",
    font=("Arial",12,"bold"),
    bg="#f5f5f5",
    fg="red"
)

weight_label.pack(
    pady=10
)

# =========================
# Footer
# =========================

footer = tk.Label(
    root,
    text="Machine Learning Based Matchmaking System",
    font=("Arial",10),
    bg="#f5f5f5"
)

footer.pack(
    side="bottom",
    pady=10
)

root.mainloop()