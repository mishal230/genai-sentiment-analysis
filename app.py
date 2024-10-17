import streamlit as st
from transformers import pipeline

# Title of the app
st.title("General Sentiment Analysis")

# Description
st.write("""
         This app uses a Hugging Face sentiment analysis model to classify text as positive, negative, or neutral.
         You can use it to analyze the sentiment of news articles, movie reviews, social media posts, and more.
         """)

# Create a text input field
user_input = st.text_area("Enter Text", "I had a wonderful day at the park today!")

# Load the general sentiment analysis pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

# Split sentence into parts for separate analysis
def split_and_analyze(text):
    sentences = text.split(" and ")  # Split by "and" for simplicity
    results = {}
    for sentence in sentences:
        result = pipe(sentence)
        results[sentence] = result
    return results

# Perform sentiment analysis when the button is pressed
if st.button("Analyze Sentiment"):
    pipe = load_model()
    
    # Analyze each part of the sentence separately
    results = split_and_analyze(user_input)
    
    # Display the result
    st.subheader("Sentiment Analysis Result")
    for sentence, result in results.items():
        for res in result:
            st.write(f"Sentence: **{sentence}**, Label: **{res['label']}**, Confidence: **{res['score']:.2f}**")
