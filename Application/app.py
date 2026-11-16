
import streamlit as st
import tensorflow as tf
import json
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# Application configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# File paths
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR.parent / "Models" / "sentiment_lstm_model.keras"
WORD_INDEX_PATH = BASE_DIR.parent / "Models" / "word_index.json"

VOCAB_SIZE = 10000
MAX_LENGTH = 250

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Load word dictionary
@st.cache_data
def load_word_index():
    with open(WORD_INDEX_PATH, "r") as file:
        return json.load(file)

model = load_model()
word_index = load_word_index()

# Convert user text into numerical sequences
def preprocess_text(text):

    words = text_to_word_sequence(text)

    sequence = [1]

    for word in words:
        index = word_index.get(word, 2) + 3

        if index >= VOCAB_SIZE:
            index = 2

        sequence.append(index)

    padded_sequence = pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    return padded_sequence

# Application heading
st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below to find out "
    "whether its sentiment is positive or negative."
)

# User input
review = st.text_area(
    "Enter your movie review:",
    height=180,
    placeholder="Example: This movie was amazing..."
)

# Prediction button
if st.button("Analyze Sentiment"):

    if review.strip():

        processed_review = preprocess_text(review)

        prediction = model.predict(
            processed_review,
            verbose=0
        )[0][0]

        if prediction >= 0.5:

            st.success("Positive Sentiment 😊")
            st.write(
                f"Positive probability: {prediction * 100:.2f}%"
            )

        else:

            st.error("Negative Sentiment 😞")
            st.write(
                f"Negative probability: {(1 - prediction) * 100:.2f}%"
            )

    else:

        st.warning("Please enter a movie review first.")

st.divider()

st.caption(
    "Powered by Deep Learning and LSTM Neural Networks"
)
