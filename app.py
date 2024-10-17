import streamlit as st
from transformers import pipeline

# Set up the page configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon=":bar_chart:", layout="centered")

# Title of the app with a professional look
st.title("🧠 AI Sentiment Analyzer")

# Add description and instructions
st.write("""
Welcome to the **AI Sentiment Analyzer**! This tool uses state-of-the-art NLP models to analyze the sentiment of your text input. 
You can input anything from movie reviews, news headlines, social media posts, or personal thoughts, and we will break down the sentiment for you!
""")

# Instructions
st.markdown("""
- **Enter text**: Input any text you'd like to analyze for sentiment.
- **Sentiment analysis**: We'll break down each part of your text and give you both individual and overall sentiment.
""")

# Input text field
user_input = st.text_area("Enter your text for sentiment analysis", placeholder="Type something like 'I love myself but sometimes I have self-doubt.'")

# Load the sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

# Function to split and analyze sentences, handling "but"
def split_and_analyze(text):
    # Split sentence by "but" or "and" for more nuanced analysis
    if "but" in text:
        sentences = text.split(" but ")
    else:
        sentences = text.split(" and ")
    
    results = {}
    for sentence in sentences:
        result = pipe(sentence)
        results[sentence] = result
    return results

# Perform sentiment analysis
if st.button("Analyze Sentiment"):
    pipe = load_model()
    
    # Display a loading spinner while processing
    with st.spinner("Analyzing..."):
        results = split_and_analyze(user_input)

    # Display results with professional formatting
    st.subheader("🔍 Detailed Sentiment Analysis")

    total_score = 0
    for sentence, result in results.items():
        for res in result:
            label = res['label']
            score = res['score']
            
            if label == "POSITIVE":
                st.success(f"**Sentence:** \"{sentence}\" | **Sentiment:** {label} | **Confidence:** {score:.2f}")
                total_score += score  # Positive score contributes positively
            elif label == "NEGATIVE":
                st.error(f"**Sentence:** \"{sentence}\" | **Sentiment:** {label} | **Confidence:** {score:.2f}")
                total_score -= score  # Negative score subtracts
            
            else:
                st.warning(f"**Sentence:** \"{sentence}\" | **Sentiment:** {label} | **Confidence:** {score:.2f}")

    # Overall sentiment based on the analysis
    st.subheader("📊 Overall Sentiment Summary")
    if total_score > 0:
        st.success("Overall sentiment is **positive** based on the balance of positive and negative phrases.")
    elif total_score < 0:
        st.error("Overall sentiment is **negative** based on the balance of positive and negative phrases.")
    else:
        st.warning("Overall sentiment is **neutral** with balanced positive and negative phrases.")
        
# Footer for a professional touch
st.markdown("""
---
*Created with [Streamlit](https://streamlit.io/) and [Hugging Face Transformers](https://huggingface.co/transformers)*
""")
