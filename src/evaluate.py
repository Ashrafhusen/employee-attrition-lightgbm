import joblib 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from src.preprocess import load_data

def evaluate_model():

    df = load_data()
    X = df.drop("Attrition_Yes", axis = 1)
    y = df['Attrition_Yes']

    _, X_test, _, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    model = joblib.load('models/model.pkl')

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(" Model Evaluation Results")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()


