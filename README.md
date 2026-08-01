# 🎬 Movie Review Sentiment Classifier

A deep learning web app that classifies movie reviews as **Positive** or **Negative**, built using a Keras/TensorFlow model trained on Kaggle.

---

## 📌 Overview

This project trains a binary sentiment classifier on movie review data using Keras, then wraps it in a Streamlit web app so users can type in a review and get an instant prediction with a confidence score.

**Deployment status:** Initially attempted deployment on **Streamlit Community Cloud**, but ran into a platform-side issue — Streamlit Cloud's build environment was using **Python 3.14**, and TensorFlow does not yet have compatible wheels for that Python version, causing dependency installation to fail. Currently working on an alternative deployment approach (e.g. Docker + Render) to resolve this.

---

## 🛠️ Tech Stack

- **Model:** Keras / TensorFlow (binary sentiment classifier, sigmoid output)
- **Text Preprocessing:** Keras `Tokenizer` + custom text cleaning (lowercasing, HTML tag removal, punctuation removal)
- **Web Framework:** Streamlit
- **Training Environment:** Kaggle Notebook

---

## 📂 Project Structure

---

## 🧠 How It Works

1. **Text Cleaning** — Input review is lowercased, HTML tags are stripped, punctuation is removed.
2. **Tokenization** — Cleaned text is converted into integer sequences using the same `Tokenizer` fitted during training.
3. **Padding** — Sequences are padded/truncated to the fixed `max_len` used during training.
4. **Prediction** — The Keras model outputs a probability (0–1) via a sigmoid output layer.
5. **Label Mapping** — `> 0.5` → **Positive Review 😊**, otherwise → **Negative Review 😞**, shown with a confidence percentage.

```python
def predict_review(review):
    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=config["max_len"])
    pred = model.predict(pad)[0][0]

    if pred > 0.5:
        return "Positive Review 😊", float(pred)
    else:
        return "Negative Review 😞", float(1 - pred)
```

---

## 🚀 Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/sentiment-app.git
cd sentiment-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ⚠️ Known Deployment Issue

While attempting to deploy on **Streamlit Community Cloud**, the build consistently failed with:

**Root cause:** Streamlit Cloud's environment was running **Python 3.14**, a version for which TensorFlow has not yet published compatible wheels — even after setting `runtime.txt` and the "Advanced settings" Python version option to 3.11.

**Next steps:** Exploring deployment via **Docker on Render**, which allows full control over the Python version through a `Dockerfile`, avoiding this platform-side limitation.

---

## 📝 Notes

- Labels were mapped manually (`positive` → `1`, `negative` → `0`) rather than using `sklearn.LabelEncoder`.
- Any new text input must go through the **exact same** `clean_text()` preprocessing used during training, or predictions will be unreliable.

---

## 📄 License

MIT License — feel free to fork, modify, and reuse.
