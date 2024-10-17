import streamlit as st
from transformers import pipeline
import re

# Set up the page configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon=":bar_chart:", layout="centered")

# Title of the app with a professional look
st.title("🧠 AI Sentiment Analyzer")

# Add description and instructions
st.write("""
Welcome to the **AI Sentiment Analyzer**! This tool uses state-of-the-art NLP models to analyze the sentiment of your text input. 
You can input anything from news headlines, social media posts, or personal thoughts, and we will break down the sentiment for you!
""")

# Instructions
st.markdown("""
- **Enter text**: Input any text you'd like to analyze for sentiment.
- **Sentiment analysis**: We'll break down each part of your text and give you both individual and overall sentiment.
""")

# Input text field
user_input = st.text_area("Enter your text for sentiment analysis", placeholder="Type something like 'Pakistan is a Wonderful Country.'")

# Load the sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

# Enhanced function to split and analyze sentences, handling conjunctions and context
def split_and_analyze(text):
    # Split text by "but", "and", "when", "however" while preserving context
    delimiters = r'\s*(but|and|when|however)\s*'
    sentences = re.split(delimiters, text)
    
    results = {}
    full_text_result = pipe(text)  # Analyze the full text first
    results[text] = full_text_result

    if len(sentences) > 1:
        # Only split and analyze further if the full text analysis is ambiguous
        for sentence in sentences:
            if sentence.strip():
                sentence_result = pipe(sentence)
                results[sentence] = sentence_result
    return results

# Perform sentiment analysis
if st.button("Analyze Sentiment"):
    pipe = load_model()
    
    # Display a loading spinner while processing
    with st.spinner("Analyzing..."):
        results = split_and_analyze(user_input)

    # Display results with professional formatting
    st.subheader("🔍 Detailed Sentiment Analysis")

    positive_count = 0
    negative_count = 0

    for sentence, result in results.items():
        for res in result:
            label = res['label']
            score = res['score']
            
            if label == "POSITIVE":
                st.success(f"**Sentence:** \"{sentence}\" | **Sentiment:** {label} | **Confidence:** {score:.2f}")
                positive_count += 1
            elif label == "NEGATIVE":
                st.error(f"**Sentence:** \"{sentence}\" | **Sentiment:** {label} | **Confidence:** {score:.2f}")
                negative_count += 1

    # Overall sentiment summary based on the counts of positive and negative phrases
    st.subheader("📊 Overall Sentiment Summary")

    if positive_count > 0 and negative_count > 0:
        st.warning("Overall sentiment is **neutral** due to both positive and negative phrases being present.")
    elif positive_count > negative_count:
        st.success(f"Overall sentiment is **positive** based on {positive_count} positive phrases and {negative_count} negative phrases.")
    elif negative_count > positive_count:
        st.error(f"Overall sentiment is **negative** based on {positive_count} positive phrases and {negative_count} negative phrases.")
    else:
        st.info("Overall sentiment is **neutral**.")

# Footer for a professional touch
st.markdown("""
---
*Created with [Streamlit](https://streamlit.io/) and [Hugging Face Transformers](https://huggingface.co/transformers)* 
""")
