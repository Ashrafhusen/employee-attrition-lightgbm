import joblib
import numpy as np
import os
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from src.preprocess import load_data

def evaluate_model():
    df = load_data()
    X = df.drop("Attrition_Yes", axis=1)
    y = df["Attrition_Yes"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load("models/model.pkl")
    y_proba = model.predict_proba(X_test)[:, 1]  

    # Threshold tuning
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = [f1_score(y_test, y_proba > t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (y_proba > best_threshold).astype(int)

    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    print(" Model Evaluation Results")
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    
    os.makedirs("metrics", exist_ok=True)
    metrics = {
        "threshold": round(best_threshold, 2),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report
    }
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate_model()
