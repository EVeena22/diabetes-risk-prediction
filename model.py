import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_PATH = Path("diabetes.csv")
MODEL_PATH = Path("model.joblib")
METRICS_PATH = Path("metrics.json")

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def prepare_data(df: pd.DataFrame):
    # Adjust feature columns to match your dataset
    # Example assumes Pima Indians format
    feature_cols = ["Glucose", "Insulin", "BMI", "Age"]
    target_col = "Outcome"

    X = df[feature_cols].values
    y = df[target_col].values

    return X, y, feature_cols

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler(feature_range=(0, 1))
    clf = SVC(kernel="linear", probability=True, random_state=42)

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("scaler", scaler),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }

    return pipe, metrics

def main():
    df = load_data(DATA_PATH)
    X, y, feature_cols = prepare_data(df)

    model, metrics = train_model(X, y)
    joblib.dump(
        {
            "model": model,
            "feature_names": ["Glucose", "Insulin", "BMI", "Age"]
        },
        MODEL_PATH
    )

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Model and metrics saved.")

if __name__ == "__main__":
    main()
