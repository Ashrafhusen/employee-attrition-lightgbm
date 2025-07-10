# 🚀 LightGBM Attrition Prediction Pipeline

This project builds a machine learning pipeline to predict employee attrition using LightGBM. It includes class imbalance handling, threshold tuning, evaluation logging, and model saving.

---

## ✅ What This Project Does

- Handles class imbalance using `scale_pos_weight`
- Trains a LightGBM classifier on employee attrition data
- Tunes the prediction threshold for best F1 score
- Saves:
  - Trained model → `models/model.pkl`
  - Evaluation metrics → `metrics/metrics.json`

---

## 🔧 Pipeline Steps

### 1. Preprocess Data (`preprocess.py`)
- Loads dataset
- Splits into features `X` and target `y`

### 2. Train Model (`train.py`)
- Splits into training and test sets (80/20)
- Uses `scale_pos_weight` to handle class imbalance
- Saves model to `models/model.pkl`

### 3. Evaluate Model (`evaluate.py`)
- Predicts probabilities using `predict_proba`
- Tunes decision threshold for best F1 score
- Logs metrics to `metrics/metrics.json`

---

## 📊 Latest Results (Threshold = 0.2)

- **Accuracy:** 76.87%
- **Precision:** 32.94%
- **Recall:** 71.79%
- **F1 Score:** 45.16%

**Confusion Matrix**:
[[198 57]
[ 11 28]]
