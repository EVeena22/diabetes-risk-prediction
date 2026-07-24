# Diabetes Risk Prediction Web App

A Flask web application that predicts diabetes risk from patient health data using a Support Vector Machine (SVM) classifier, with model explainability and batch prediction support.

## Features
- **Risk prediction** — enter patient data (glucose, insulin, BMI, age, etc.) and get a real-time diabetes risk prediction
- **SHAP explainability** — see which features most influenced each individual prediction
- **Batch prediction** — upload a CSV of multiple patients and get predictions for all of them at once
- **Model performance dashboard** — view accuracy, precision, recall, and F1 score for the trained model
- **AI-assisted risk simulator** — uses the Gemini API to generate plain-language explanations of prediction results

## Tech Stack
- **Backend:** Python, Flask
- **ML:** Scikit-learn (SVM classifier), SHAP (explainability)
- **Data:** Pandas, NumPy
- **AI integration:** Google Gemini API

## Model Performance
Trained on the Pima Indians Diabetes Dataset:

| Metric | Score |
|---|---|
| Accuracy | 73.4% |
| Precision | 65.9% |
| Recall | 50.0% |
| F1 Score | 56.8% |

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your own Gemini API key:
   ```bash
   cp .env.example .env
   ```

3. Run the app:
   ```bash
   python app.py
   ```

## Dataset
Uses the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Note
`model.joblib` contains the pre-trained SVM model. To retrain from scratch, run `model.py`.
