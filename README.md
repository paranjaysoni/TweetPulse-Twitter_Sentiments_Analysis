# TweetPulse: Twitter Sentiment Analysis App 🎯

TweetPulse is a web application that analyzes tweets in real-time and classifies their sentiment into four categories: **Irrelevant**, **Negative**, **Neutral**, and **Positive**. It helps users quickly understand the overall sentiment behind any Twitter conversation.

---

## Live Demo

Try out the live app by [clicking here](https://tweetpulse.streamlit.app).

---

## Features

- Real-time sentiment analysis of tweets  
- Preprocessing to clean noisy social media text (slang, hashtags, URLs, emojis)  
- TF-IDF vectorization for effective feature extraction  
- Logistic Regression model trained on labeled Twitter data with **≈ 95% accuracy**  
- User-friendly interface built with Streamlit  
- Models and vectorizers hosted on Hugging Face for fast loading and scalability  
- Color-coded sentiment tags for easy interpretation

---

## How It Works

1. **Data Collection:**  
   Collects tweets for training and accepts user input for live analysis.

2. **Preprocessing:**  
   Cleans tweets by removing unwanted characters, converting to lowercase, removing stopwords, and applying stemming.

3. **Feature Extraction:**  
   Converts processed text into numerical features using TF-IDF vectorization.

4. **Model Training:**  
   Uses Logistic Regression to classify tweet sentiments based on the extracted features.

5. **Prediction:**  
   Takes user input tweets, preprocesses and vectorizes them, then predicts sentiment with clear, color-coded output.

---

## Technologies Used

- **Python** – Core programming language  
- **Streamlit** – For building the interactive web app  
- **NLTK** – Natural Language Toolkit for text preprocessing  
- **Scikit-learn** – Machine learning library for model building  
- **Hugging Face Hub** – Hosting models and vectorizers  
- **Pandas** – Data manipulation and analysis

---

## Run It Locally

To run this project locally, clone the repo and launch the app using Streamlit:  
```bash
git clone https://github.com/paranjaysoni/TweetPulse-Twitter_Sentiments_Analysis.git
cd TweetPulse-Twitter_Sentiments_Analysis
pip install -r requirements.txt
streamlit run app.py
