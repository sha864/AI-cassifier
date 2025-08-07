import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

"""
Simple Collaborative Filtering Recommendation System

Approach:
1. Manually create a user-movie ratings matrix.
2. Compute user-user similarity using cosine similarity.
3. Recommend movies to a target user based on ratings from similar users.
"""

# Step 1: Create dummy ratings data (users x movies)
data = {
    'The Matrix':     [5, 4, np.nan, 1, 2],
    'Inception':      [4, np.nan, 5, 2, 1],
    'Titanic':        [1, 2, 1, 5, 4],
    'Avengers':       [4, 5, 4, np.nan, 3],
    'Interstellar':   [5, 5, 4, 1, 2],
    'The Notebook':   [1, 1, 1, 5, 5],
    'The Dark Knight':[5, 4, 4, 2, np.nan],
}

users = ['User1', 'User2', 'User3', 'User4', 'User5']
ratings_df = pd.DataFrame(data, index=users)

# Step 2: Fill NaNs with 0 (unrated movies)
ratings_filled = ratings_df.fillna(0)

# Step 3: Compute cosine similarity between users
similarity_matrix = cosine_similarity(ratings_filled)
similarity_df = pd.DataFrame(similarity_matrix, index=users, columns=users)

# Step 4: Get top similar users for a given user
def get_similar_users(user_id, top_n=2):
    similar_scores = similarity_df[user_id].drop(user_id)
    top_similars = similar_scores.sort_values(ascending=False).head(top_n)
    return top_similars.index.tolist()

# Step 5: Recommend movies
def recommend_movies_for_user(user_id, top_n=3):
    similar_users = get_similar_users(user_id)
    user_ratings = ratings_df.loc[user_id]
    unseen_movies = user_ratings[user_ratings.isna()].index.tolist()

    scores = {}
    for movie in unseen_movies:
        score = 0
        count = 0
        for sim_user in similar_users:
            rating = ratings_df.loc[sim_user, movie]
            if not np.isnan(rating):
                score += rating
                count += 1
        if count > 0:
            scores[movie] = score / count

    sorted_recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in sorted_recommendations[:top_n]]

# Step 6: Show recommendations for a sample user
target_user = 'User3'
recommendations = recommend_movies_for_user(target_user)

print(f"🎬 Recommendations for {target_user}:")
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie}")
