import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the CSV dataset
faq_df = pd.read_csv("ecommerce_faq.csv")

# Vectorize the questions using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(faq_df['question'])

# Chatbot function to get the most similar answer
def get_answer(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    best_match_idx = similarity.argmax()
    score = similarity[0][best_match_idx]
    
    if score < 0.3:
        return " Sorry, I couldn't find a relevant answer."
    return faq_df.iloc[best_match_idx]['answer']

# Streamlit UI
st.title(" E-commerce FAQ Chatbot")
query = st.text_input("Ask a question:")
if query:
    response = get_answer(query)
    st.write("**Answer:**", response)
