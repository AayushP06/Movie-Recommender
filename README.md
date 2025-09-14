# Movie-Recommender
# 🎬 Movie Recommender System

A simple **content-based movie recommendation system** built with **Python, Scikit-learn, and Streamlit**.  
It suggests similar movies based on overview and genres using **cosine similarity**.

---

## 🚀 Features
- Search any movie title.
- Get top N similar movies.
- Clean and interactive UI built with Streamlit.
- Easy to deploy and extend with more data.

---
 Dataset:
 
---
LIVE DEMO :
https://movie-recommender-by-aayush.streamlit.app/


## 📊 Dataset

This project uses the **TMDB 5000 Movie Dataset** available on Kaggle:  
🔗 [Download Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

After downloading, place the files inside a folder named `data/` in your project directory:


## 🛠️ Tech Stack
- **Python 3.x**
- **Pandas, Numpy** → Data handling
- **Scikit-learn** → Vectorization & similarity
- **Streamlit** → Web app framework

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/movie-recommender.git
   cd movie-recommender

2.Create a virtual environment (recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate   # For Linux/Mac
  venv\Scripts\activate      # For Windows
  pip install -r requirements.txt
  streamlit run app.py
```

🧠 How It Works

Movie overviews + genres are converted into text "tags".

CountVectorizer creates vector embeddings.

cosine_similarity finds closest vectors.

Top-N similar movies are shown as recommendations.


🚀 Deployment

Streamlit Cloud: https://streamlit.io/cloud

Hugging Face Spaces: https://huggingface.co/spaces

Heroku/Render: Requires Procfile & Docker setup.




