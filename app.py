import re
import string
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("sentiment_model.keras")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    config = pickle.load(open("config.pkl", "rb"))
    return model, tokenizer, config

model, tokenizer, config = load_artifacts()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def predict_review(review):
    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=config["max_len"])
    pred = model.predict(pad)[0][0]

    if pred > 0.5:
        return "Positive Review 😊", float(pred)
    else:
        return "Negative Review 😞", float(1 - pred)

st.set_page_config(page_title="Sentiment Classifier", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Classifier")
st.write("Enter a movie review and this Keras deep learning model will predict whether it's positive or negative.")

review = st.text_area("Review Text", height=150, placeholder="Type or paste a movie review here...")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        label, confidence = predict_review(review)
        st.subheader(label)
        st.write(f"Confidence: {confidence:.2%}")
