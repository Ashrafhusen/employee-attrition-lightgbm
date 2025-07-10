
---

```markdown
# 🧠 LightGBM Employee Attrition Prediction

This project builds a machine learning pipeline to predict **employee attrition** using **LightGBM**, with a focus on:

- Real-world **class imbalance handling**
- **Threshold tuning** for imbalanced classification
- **CI/CD-friendly metric logging**
- Model versioning and evaluation tracking

---

## 📂 Project Structure

```

LightBGMAction/
│
├── data/                   # (Optional) raw or cleaned dataset
├── models/                 # Stores trained LightGBM model (`model.pkl`)
├── metrics/                # Stores evaluation metrics (`metrics.json`)
├── src/
│   ├── preprocess.py       # Loads and preprocesses data
│   ├── train.py            # Trains the LightGBM model
│   └── evaluate.py         # Evaluates the model with threshold tuning
├── requirements.txt
└── README.md

```

---

## 🔧 What We've Done So Far

### ✅ Step 1: Data Loading and Preprocessing
- Loaded HR attrition dataset
- Encoded target column: `Attrition_Yes`
- Feature matrix `X` and target vector `y` prepared

### ✅ Step 2: Model Training (`src/train.py`)
- Split data into train/test sets (80/20)
- Applied **LightGBM Classifier** with:
  - `scale_pos_weight` for class imbalance
  - Basic tuning (`num_leaves`, `learning_rate`, etc.)
- Trained model saved to `models/model.pkl`

### ✅ Step 3: Model Evaluation (`src/evaluate.py`)
- Predicted with **probabilities** using `predict_proba`
- Tuned decision **threshold** for best F1 score
- Final evaluation at **threshold = 0.2**:
```

Accuracy:  76.87%
Precision: 32.94%
Recall:    71.79%
F1 Score:  45.16%

````
- Metrics saved to `metrics/metrics.json` (CI/CD-ready)

---

## 📊 Key Insights

- Default threshold (0.5) **underperformed on minority class** (`Attrition = Yes`)
- Using **threshold = 0.2** gave a **+40% boost in recall**
- Final model catches **~72% of attrition cases** with modest drop in precision

---

## 📁 Files of Interest

| File | Description |
|------|-------------|
| `src/preprocess.py` | Loads and prepares data |
| `src/train.py`      | Trains the model, handles imbalance |
| `src/evaluate.py`   | Tunes threshold, evaluates metrics, saves JSON |
| `metrics/metrics.json` | Contains evaluation metrics (accuracy, precision, recall, F1, confusion matrix) |
| `models/model.pkl`  | Trained LightGBM model |

---

## 🚀 Next Steps

- 📈 Add ROC-AUC and PR-AUC metrics
- 📊 Visualize F1 vs threshold curve
- 🧪 Add confusion matrix heatmap and feature importance plot
- 🔁 Add GitHub Actions for automated training/testing
- 📦 Add DVC for dataset/model versioning

---

## 📌 Requirements

- Python 3.8+
- LightGBM
- scikit-learn
- joblib
- numpy
- pandas

```bash
pip install -r requirements.txt
````

---

## 🏁 Run the Pipeline

```bash
# Train model
python src/train.py

# Evaluate model (with threshold tuning)
python src/evaluate.py
```

---

## 🧪 Example Output (`metrics.json`)

```json
{
  "threshold": 0.2,
  "accuracy": 0.7687,
  "precision": 0.3294,
  "recall": 0.7179,
  "f1_score": 0.4516,
  "confusion_matrix": [[198, 57], [11, 28]]
}
```

---

## 👨‍💻 Author

* Built and maintained by **Ashraf Raj**

