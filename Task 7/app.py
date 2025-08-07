# ✅ app.py
import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ✅ Sentiment labeling function
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# ✅ Load CSV
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Sentiment'] = df['review'].apply(get_sentiment)
    return df

# ✅ App layout
st.set_page_config(page_title="Sentiment Analysis Dashboard")
st.title("💬 Sentiment Analysis Dashboard")
st.markdown("Analyze text reviews and visualize sentiment distribution.")

# ✅ Upload dataset
uploaded_file = st.file_uploader("Upload CSV file with a 'review' column", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # ✅ Show stats
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts()

    # ✅ Pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # ✅ Sample texts
    st.subheader("📝 Sample Reviews by Sentiment")
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        st.markdown(f"**{sentiment}**")
        samples = df[df['Sentiment'] == sentiment]['review'].sample(min(2, len(df[df['Sentiment'] == sentiment])))
        for text in samples:
            st.write(f"👉 {text}")

    # ✅ Word Clouds
    st.subheader("☁️ Word Cloud by Sentiment (Optional)")
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        st.markdown(f"**{sentiment} Words**")
        text = ' '.join(df[df['Sentiment'] == sentiment]['review'])
        if text.strip():
            wc = WordCloud(width=400, height=200, background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info(f"No {sentiment} reviews to generate word cloud.")

# ✅ Real-time text input
st.subheader("🎯 Try Your Own Text")
user_input = st.text_area("Enter a sentence to analyze sentiment")
if st.button("Analyze"):
    if user_input.strip():
        result = get_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{result}**")
    else:
        st.warning("Please enter some text.")
