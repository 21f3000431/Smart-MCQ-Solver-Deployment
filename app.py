import joblib
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load trained model and vectorizer
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
model = joblib.load(BASE_DIR / "logistic_model.pkl")


def predict(question, option_a, option_b, option_c, option_d, option_e):
    combined_text = (
        f"{question} "
        f"A: {option_a} "
        f"B: {option_b} "
        f"C: {option_c} "
        f"D: {option_d} "
        f"E: {option_e}"
    )

    X = vectorizer.transform([combined_text])

    probabilities = model.predict_proba(X)[0]
    labels = model.classes_

    top3_idx = probabilities.argsort()[-3:][::-1]

    return labels, probabilities, top3_idx


st.set_page_config(page_title="Smart MCQ Solver")

st.title("Smart MCQ Solver")

st.write(
    "Predict the most appropriate answer using the trained TF-IDF + Logistic Regression model."
)

question = st.text_area("Question")

option_a = st.text_area("Option A")
option_b = st.text_area("Option B")
option_c = st.text_area("Option C")
option_d = st.text_area("Option D")
option_e = st.text_area("Option E")

if st.button("Predict"):

    labels, probabilities, top3_idx = predict(
        question,
        option_a,
        option_b,
        option_c,
        option_d,
        option_e,
    )

    st.success(f"Predicted Answer: **{labels[top3_idx[0]]}**")

    st.subheader("Top 3 Predictions")

    for i in top3_idx:
        st.write(f"{labels[i]} ({probabilities[i]:.2%})")