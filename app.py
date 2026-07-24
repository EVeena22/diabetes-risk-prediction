from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Optional: SHAP for explanations (install: pip install shap)
import shap

import os
from google import genai  # Gemini SDK

from dotenv import load_dotenv

load_dotenv()  # Load .env into environment variables

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

print("Gemini Key:", GEMINI_API_KEY)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
LOG_PREDICTIONS_PATH = BASE_DIR / "predictions_log.csv"

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# ---------- Utilities ----------

def load_model():
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    return model, feature_names

MODEL, FEATURE_NAMES = load_model()

def make_single_prediction(features: Dict[str, float]) -> Dict[str, Any]:
    x = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
    proba = MODEL.predict_proba(x)[0, 1]
    label = int(proba >= 0.5)
    return {
        "probability": float(proba),
        "label": label,
    }

def explain_prediction(features: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Return top 3 features by SHAP value.
    Handles both binary (single output) and multi-class outputs safely.
    """
    x = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)

    try:
        # For a Pipeline with scaler + linear SVC
        clf = MODEL.named_steps.get("clf", MODEL)
        scaler = MODEL.named_steps.get("scaler", None)

        if scaler is not None:
            background = scaler.transform(x)
            explainer = shap.LinearExplainer(clf, background)
            shap_values = explainer.shap_values(x)
        else:
            explainer = shap.Explainer(MODEL)
            shap_values = explainer(x)

        # shap_values can be:
        # - numpy array shape (1, n_features)
        # - list of arrays for each class
        if isinstance(shap_values, list):
            # Pick class 1 if it exists, else first class
            if len(shap_values) > 1:
                sv = shap_values[1][0]
            else:
                sv = shap_values[0][0]
        else:
            # shap.Explainer style object
            sv = np.array(shap_values.values)[0]
    except Exception:
        # Fallback: simple absolute coefficient-based ranking if SHAP fails
        try:
            clf = MODEL.named_steps.get("clf", MODEL)
            if hasattr(clf, "coef_"):
                sv = clf.coef_[0]
            else:
                sv = np.zeros(len(FEATURE_NAMES))
        except Exception:
            sv = np.zeros(len(FEATURE_NAMES))

    pairs = list(zip(FEATURE_NAMES, sv))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)

    top3 = [
        {
            "feature": name,
            "importance": float(val),
            "direction": "increases_risk" if val > 0 else "decreases_risk",
        }
        for name, val in pairs[:3]
    ]
    return top3

def log_prediction_row(row: Dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    if LOG_PREDICTIONS_PATH.exists():
        df.to_csv(LOG_PREDICTIONS_PATH, mode="a", index=False, header=False)
    else:
        df.to_csv(LOG_PREDICTIONS_PATH, index=False)

def compute_live_metrics() -> Dict[str, Any]:
    """
    If true labels are available in predictions_log.csv (column 'true_label'),
    compute metrics live from logged data; otherwise fall back to static metrics.json.
    """
    if LOG_PREDICTIONS_PATH.exists():
        df = pd.read_csv(LOG_PREDICTIONS_PATH)
        if "true_label" in df.columns:
            y_true = df["true_label"].values
            y_pred = df["prediction_label"].values

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred).tolist()

            return {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion_matrix": cm,
                "source": "live_logs",
            }

    # Fallback: static metrics from training
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            m = json.load(f)
        m["source"] = "training_metrics"
        return m

    return {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "confusion_matrix": None,
        "source": "none",
    }

# ---------- Page routes (frontend) ----------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/batch")
def batch_page():
    return render_template("batch-prediction.html")

@app.route("/risk-simulator")
def risk_simulator_page():
    return render_template("risk-simulator.html")

@app.route("/model-performance")
def model_performance_page():
    return render_template("model-performance.html")

# Optional: serve static files if you move CSS/JS to /static
@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

# ---------- APIs ----------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts form data (for existing single-patient home.html)
    or JSON body.
    Expected fields: Glucose, Insulin, BMI, Age
    """
    if request.is_json:
        data = request.get_json()
    else:
        data = {k: float(v) for k, v in request.form.items()}

    try:
        features = {name: float(data[name]) for name in FEATURE_NAMES}
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    pred = make_single_prediction(features)
    insights = explain_prediction(features)

    result = {
        "input": features,
        "probability": pred["probability"],
        "label": pred["label"],
        "insights": insights,
    }

    log_prediction_row({
        **features,
        "prediction_proba": pred["probability"],
        "prediction_label": pred["label"],
    })

    # For your existing HTML form, you can adapt the JS to use fetch
    return jsonify(result)

@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """
    Accepts CSV or JSON file.
    CSV: first row header; must contain the feature columns.
    JSON: list of objects with same feature keys.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith(".json"):
            content = file.read().decode("utf-8")
            data = json.loads(content)
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400

    for col in FEATURE_NAMES:
        if col not in df.columns:
            return jsonify({"error": f"Missing column in file: {col}"}), 400

    X = df[FEATURE_NAMES].astype(float).values
    proba = MODEL.predict_proba(X)[:, 1]
    labels = (proba >= 0.5).astype(int)

    results = []
    for idx, row in df.iterrows():
        features = {name: float(row[name]) for name in FEATURE_NAMES}
        insights = explain_prediction(features)
        results.append({
            "id": int(idx + 1),
            **features,
            "probability": float(proba[idx]),
            "label": int(labels[idx]),
            "insights": insights,
        })

        log_prediction_row({
            **features,
            "prediction_proba": float(proba[idx]),
            "prediction_label": int(labels[idx]),
        })

    return jsonify({"results": results})

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Used by risk-simulator.html.
    Body: JSON with Glucose, Insulin, BMI, Age.
    """
    data = request.get_json(force=True)
    try:
        features = {name: float(data[name]) for name in FEATURE_NAMES}
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    pred = make_single_prediction(features)
    insights = explain_prediction(features)

    return jsonify({
        "input": features,
        "probability": pred["probability"],
        "label": pred["label"],
        "insights": insights,
    })

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    metrics = compute_live_metrics()
    return jsonify(metrics)

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """
    Admin endpoint to retrain model with a new CSV dataset.
    Expects a CSV file with same schema as diabetes.csv.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    from model import prepare_data, train_model  # reuse functions
    X, y, _ = prepare_data(df)
    new_model, new_metrics = train_model(X, y)

    joblib.dump(
        {
            "model": new_model,
            "feature_names": FEATURE_NAMES,
        },
        MODEL_PATH
    )
    with open(METRICS_PATH, "w") as f:
        json.dump(new_metrics, f, indent=2)

    global MODEL
    MODEL = new_model

    return jsonify({
        "message": "Model retrained and updated.",
        "metrics": new_metrics,
    })

import re

@app.route("/api/advice", methods=["POST"])
def api_advice():
    """
    Uses Gemini to generate patient-friendly lifestyle advice as JSON.
    Requires GEMINI_API_KEY to be set in environment.
    Sends rich context so the LLM can give more meaningful insights.
    """
    if gemini_client is None:
        return jsonify({
            "advice": (
                "LLM advice service is not configured. "
                "Please set the GEMINI_API_KEY environment variable to enable it."
            ),
            "risk_label": None,
            "risk_probability": None,
            "top_factors": [],
        }), 200

    data = request.get_json(force=True)

    # From frontend
    metrics = data.get("metrics", {})
    probability = data.get("probability")
    label = data.get("label")
    insights = data.get("insights", [])

    # Derive some extra context
    risk_label = "high" if label == 1 else "low"

    top_factors_text = ", ".join(
        f"{f.get('feature')} ({f.get('direction', '').replace('_', ' ')}, importance={round(float(f.get('importance', 0.0)), 3)})"
        for f in insights[:5]
    ) or "not specified"

    # Simple domain hints (non-diagnostic, high-level)
    metric_hints = {
        "Glucose": (
            "Higher fasting glucose is generally associated with higher diabetes risk. "
            "Very low values can also be concerning and should be interpreted by a clinician."
        ),
        "Insulin": (
            "Insulin levels can reflect how the body responds to blood sugar. "
            "Persistent abnormalities may relate to insulin resistance or other conditions."
        ),
        "BMI": (
            "Higher BMI often correlates with increased risk for type 2 diabetes, "
            "although waist circumference and body composition also matter."
        ),
        "Age": (
            "Risk of type 2 diabetes tends to increase with age, especially after mid-adulthood."
        ),
    }

    metric_details = {
        name: {
            "value": metrics.get(name),
            "hint": metric_hints.get(name, "No specific hint available for this metric."),
        }
        for name in FEATURE_NAMES
    }

    # Optionally include current model performance to frame reliability
    model_perf = compute_live_metrics()

    patient_summary = {
        "estimated_risk_probability": probability,
        "risk_label": risk_label,
        "top_risk_drivers": top_factors_text,
        "metrics_raw": metrics,
        "metrics_with_hints": metric_details,
        "model_performance_snapshot": {
            "accuracy": model_perf.get("accuracy"),
            "precision": model_perf.get("precision"),
            "recall": model_perf.get("recall"),
            "f1": model_perf.get("f1"),
            "note": "These values describe how the model performs on historical data, "
                    "not a guarantee for any single person.",
        },
    }

    system_instructions = (
        "You are a helpful healthcare assistant supporting diabetes prevention counseling. "
        "You must NOT provide diagnosis, treatment decisions, or medication changes. "
        "You can only give general, high-level lifestyle and habit recommendations. "
        "Always remind the user to consult a licensed healthcare professional for medical decisions. "
        "Use simple, empathetic, non-judgmental language. "
        "Focus on diet, physical activity, weight management, sleep, stress management, "
        "and regular medical check-ups, consistent with widely accepted diabetes prevention guidance. "
        "Do not reference specific copyrighted sources or quote them verbatim. "
        "Do not mention that you used Gemini or an API key. "
        "Treat the model performance numbers as general reliability indicators, not guarantees. "
        "Always respond ONLY with valid JSON. Do not include markdown, explanations, or extra text."
    )

    # Ask explicitly for a JSON object
    user_prompt = (
        "You are given a diabetes risk model prediction for one person.\n"
        "Use the structured data below to generate advice.\n"
        "Respond ONLY with a single valid JSON object of this shape (no markdown, no extra text):\n"
        '{\n'
        '  "message": "short summary in plain language",\n'
        '  "risk_explanation": "what this risk means in general terms",\n'
        '  "lifestyle_recommendations": [\n'
        '    "bullet point 1",\n'
        '    "bullet point 2",\n'
        '    "bullet point 3"\n'
        '  ],\n'
        '  "follow_up_advice": "encouragement to talk to a healthcare professional"\n'
        '}\n\n'
        "Do NOT provide diagnosis or medication advice.\n\n"
        f"STRUCTURED INPUT:\n{json.dumps(patient_summary, indent=2)}"
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            config={
                "system_instruction": system_instructions,
                "max_output_tokens": 512,
                "temperature": 0.7,
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                    },
                ],
            },
        )

        # Try to extract JSON from the response text
        raw_text = response.text or ""
        # Look for the first JSON object in the response
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)

        try:
            advice_json = json.loads(raw_text)
            advice = f"{advice_json.get('message', '')}\n\n{advice_json.get('follow_up_advice', '')}"
        except Exception:
            advice = (
                "There was an issue generating detailed advice at this time. "
                "In general, focus on a balanced eating pattern with vegetables, whole grains, "
                "and lean proteins, regular physical activity on most days of the week, "
                "maintaining a healthy weight, getting enough sleep, managing stress, and "
                "following up with your healthcare provider for personalized guidance."
            )

    except Exception as e:
        # Log the error for debugging
        print("Gemini API error in /api/advice:", repr(e))
        advice = (
            "There was an issue generating detailed advice at this time. "
            "In general, focus on a balanced eating pattern with vegetables, whole grains, "
            "and lean proteins, regular physical activity on most days of the week, "
            "maintaining a healthy weight, getting enough sleep, managing stress, and "
            "following up with your healthcare provider for personalized guidance."
        )

    return jsonify({
        "advice": advice,
        "risk_label": label,
        "risk_probability": probability,
        "top_factors": insights,
        "raw_input": metrics,
    })

if __name__ == "__main__":
    app.run(debug=True)
