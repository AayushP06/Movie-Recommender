import pandas as pd

# Load your full datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Save smaller samples (first 500 rows)
movies.head(500).to_csv("data/movies_sample.csv", index=False)
credits.head(500).to_csv("data/credits_sample.csv", index=False)

print("Sample datasets created!")
