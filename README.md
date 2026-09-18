# AI Matchmaking Recommendation System

An AI-based matchmaking recommendation system that identifies compatible user matches using "Natural Language Processing (NLP), MBTI personality compatibility, location similarity, and Machine Learning".

The system recommends the "Top 5 compatible matches" for a selected user through interactive web and desktop applications.

##  Project Overview

Traditional matchmaking systems often rely on basic filters such as age and location. This project explores an explainable recommendation approach that combines multiple compatibility factors:

- Text similarity based on user profiles
- MBTI personality compatibility
- Location similarity
- Feedback-based Machine Learning validation

The project uses synthetic user data and is designed as an educational prototype for understanding recommendation systems and NLP.

##  Objectives

- Recommend the top 5 compatible users for a selected profile.
- Calculate compatibility using text, personality, and location.
- Use TF-IDF and cosine similarity for text matching.
- Validate matching factors using a Linear Regression model.
- Provide both web-based and desktop interfaces.
- Demonstrate an end-to-end Machine Learning pipeline.
- Keep the scoring process transparent and understandable.

##  Features

✅ AI-based compatibility recommendation

✅ Top 5 compatible user matches

✅ NLP-based profile text similarity

✅ MBTI personality compatibility scoring

✅ Location-based matching

✅ Weighted compatibility percentage

✅ Feedback-based Linear Regression validation

✅ Interactive Streamlit web application

✅ Tkinter desktop application

✅ Command-line prototype

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python     | Core programming language |
| Pandas     | Data loading and processing |
| Scikit-learn | NLP and Machine Learning |
| TF-IDF     | Text feature extraction |
| Cosine Similarity | Measuring text similarity |
| Linear Regression | Feedback-based validation |
| Streamlit         | Web application |
| Tkinter           | Desktop application |
| Google Colab      | Prototyping and experimentation |
| CSV               | Data storage |

## Methodology

### 1. Data Generation

Synthetic datasets are generated using Python scripts.

- `user.py` generates user profiles.
- `feedback.py` generates historical feedback records.

The current dataset contains:

- 80 synthetic user profiles
- 400 synthetic feedback records

### 2. Text Similarity Using NLP

The system combines the following user profile fields:

- `professional_summary`
- `about_me`

TF-IDF converts the combined text into numerical vectors.

Cosine similarity is then used to measure the similarity between two user profiles.

### 3. MBTI Compatibility

The system uses a rule-based MBTI compatibility lookup.

Currently modeled pairings include:

- INTJ ↔ ENFP
- INFJ ↔ ENTP

Other pairings use a default compatibility score.

### 4. Location Similarity

The system compares users' city names:

- Same city: 100
- Different city: 50

### 5. Weighted Compatibility Score

The final compatibility score is calculated using:

```text
Compatibility Score =
    0.5 × Text Similarity
  + 0.3 × MBTI Compatibility
  + 0.2 × Location Score
```

The result is rounded to two decimal places.

### 6. Machine Learning Validation

A Linear Regression model is trained using:

- Text similarity
- MBTI compatibility
- Historical feedback action

The feedback action is represented as:

- `1` = Accepted
- `0` = Rejected

The model is used to inspect the relationship between matching factors and feedback outcomes.

### 7. Top 5 Recommendation

The system calculates compatibility scores between the selected user and other users.

The scores are sorted in descending order, and the five highest-scoring matches are displayed.

---

## Project Structure

```text
AI-Matchmaking-System/
│
├── APP.py
├── APPtk.py
├── major_proj.py
├── MAJOR_proj.ipynb
│
├── user.py
├── feedback.py
│
├── user.csv
├── feedback.csv
│
├── .gitignore
└── README.md
```

### File Description

| File | Description |
|------|-------------|
| `APP.py`   | Streamlit web application |
| `APPtk.py` | Tkinter desktop application |
| `major_proj.py` | Command-line prototype |
| `MAJOR_proj.ipynb` | Project development notebook |
| `user.py`          | Generates synthetic user data |
| `feedback.py`      | Generates synthetic feedback data |
| `user.csv`         | User profile dataset |
| `feedback.csv`     | Feedback dataset |

##  Installation

### Prerequisites

- Python 3.8 or later
- pip
- Internet connection for installing Python packages

### Step 1: Clone the Repository

```bash
git clone https://github.com/durgaprasad-58/AI-Matchmaking-System.git
```

### Step 2: Navigate to the Project Folder

```bash
cd AI-Matchmaking-System
```

### Step 3: Install Dependencies

```bash
pip install pandas scikit-learn streamlit
```

##  How to Run

### Option 1: Streamlit Web Application

```bash
streamlit run APP.py
```

The application opens in your browser.

Select a User ID and click "Find Matches" to view the top 5 recommendations.

### Option 2: Tkinter Desktop Application

```bash
python APPtk.py
```

This launches the desktop application.

### Option 3: Command-Line Prototype

```bash
python major_proj.py
```

Enter a User ID when prompted to view the recommended matches.

##  Dataset Information

### User Dataset

The `user.csv` file contains synthetic user profiles with fields such as:

- User ID
- Name
- Age
- Location
- Profession
- Experience
- Professional Summary
- About Me
- MBTI
- Interests

### Feedback Dataset

The `feedback.csv` file contains:

- User ID
- Matched User ID
- Action
- Timestamp

The current feedback records are randomly generated for demonstration purposes.

## Limitations

This project is an educational prototype and is not a production-ready matchmaking platform.

Current limitations include:

- Feedback data is randomly generated.
- The dataset contains synthetic user profiles.
- User preferences and deal-breakers are not modeled.
- MBTI scoring covers only a small number of pairings.
- Location matching uses a simple same-city comparison.
- TF-IDF does not capture deep semantic meaning.
- Compatibility weights are manually selected.
- There is no database or user authentication.
- No automated testing suite is included.

"Important:" The Machine Learning validation results from randomly generated feedback should not be interpreted as reliable real-world predictive performance.

## Future Enhancements

- Replace synthetic feedback with genuine user interaction data.
- Add user preferences and deal-breakers.
- Use Sentence-BERT or contextual embeddings for semantic matching.
- Improve personality compatibility modeling.
- Implement distance-aware location matching.
- Learn compatibility weights from real feedback.
- Introduce a database and API backend.
- Add authentication, consent handling, and privacy safeguards.
- Create automated tests.
- Add caching for improved scalability.

## Learning Outcomes

This project demonstrates practical experience in:

- Python programming
- Natural Language Processing
- TF-IDF feature extraction
- Cosine similarity
- Recommendation systems
- Machine Learning
- Data preprocessing
- Streamlit application development
- Tkinter GUI development
- Git and GitHub

## Author

"Bogoju Durgaprasad"

GitHub: [durgaprasad-58](https://github.com/durgaprasad-58)

## Disclaimer

This project is developed for educational and demonstration purposes. It uses synthetic data and should not be used to make real-world relationship, psychological, or personal compatibility decisions.

This project is developed for educational and demonstration purposes. It uses synthetic data and should not be used to make real-world relationship, psychological, or personal compatibility decisions.
