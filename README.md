# Smart MCQ Solver Deployment

This repository contains the deployment code for the **Smart MCQ Solver Challenge**.

The application uses a trained **TF-IDF + Logistic Regression** model to predict the most appropriate answer among five multiple-choice options through a simple Streamlit interface.

## Files

- `app.py` – Streamlit application
- `logistic_model.pkl` – Trained Logistic Regression model
- `tfidf_vectorizer.pkl` – Trained TF-IDF vectorizer
- `requirements.txt` – Python dependencies

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application accepts:

- Question
- Option A
- Option B
- Option C
- Option D
- Option E

and returns:

- Predicted Answer
- Top-3 Predictions with confidence scores