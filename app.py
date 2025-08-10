import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import requests
import joblib

nltk.download('stopwords')

# Load models from Hugging Face (cached for speed)
@st.cache_resource
def load_model(url):
    response = requests.get(url)
    with open("temp.pkl", "wb") as f:
        f.write(response.content)
    return joblib.load("temp.pkl")

vectorizer = load_model("https://huggingface.co/soniparanjay/TweetPulse-Twitter_Sentiment_Analysis/resolve/main/vectorizer.pkl")
best_model = load_model("https://huggingface.co/soniparanjay/TweetPulse-Twitter_Sentiment_Analysis/resolve/main/best_model.pkl")
le = load_model("https://huggingface.co/soniparanjay/TweetPulse-Twitter_Sentiment_Analysis/resolve/main/le.pkl")

# Stemming function
port_stem = PorterStemmer()
def stemming(text):
    FinalText = re.sub('[^a-zA-Z]', ' ', text)
    FinalText = FinalText.lower()
    FinalText = FinalText.split()
    FinalText = [port_stem.stem(word) for word in FinalText if not word in stopwords.words('english')]
    return ' '.join(FinalText)

# App UI
st.set_page_config(page_title="TweetPulse", page_icon="🐦", layout="centered")

st.title("🐦 TweetPulse - Twitter/X Sentiment Analysis")
st.markdown("**Analyze the sentiment of any tweet instantly!**")

# User input
tweet_input = st.text_area("Enter a tweet:", placeholder="Type or paste a tweet here...")

if st.button("Analyze Sentiment"):
    if tweet_input.strip() == "":
        st.warning("⚠️ Please enter a tweet before analyzing.")
    else:
        processed_tweet = stemming(tweet_input)
        vector_input = vectorizer.transform([processed_tweet])
        prediction = best_model.predict(vector_input)
        sentiment = le.inverse_transform(prediction)[0]

        # Color mapping
        colors = {
            "Positive": "green",
            "Negative": "red",
            "Neutral": "orange",
            "Irrelevant": "blue"
        }

        st.markdown(
            f"<h3 style='color:white;'>Sentiment: "
            f"<span style='color:{colors.get(sentiment, 'black')};'>{sentiment}</span></h3>",
            unsafe_allow_html=True
        )

# Footer Credit
st.markdown("---")
st.markdown(
    "<h5 style='text-align: center; color: grey;'>Built with  ❤️  by <a href='https://github.com/paranjaysoni' target='_blank' style='color: grey; text-decoration: none;'>Paranjay Soni</a></h5>",
    unsafe_allow_html=True
)
