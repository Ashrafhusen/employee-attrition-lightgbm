import joblib 
import lightgbm as lgb 
from sklearn.model_selection import train_test_split
from src.reprocess import load_data 
import numpy as np

def train_model():
    df = load_data()
    x = df.drop("Attrition_Yes", axis = 1)
    y = df['Attrition_Yes']

    X_train,_,y_train,_ = train_test_split(x, y, test_size = 0.2, random_state = 42)

    num_pos = np.sum(y_train)
    num_neg = len(y_train) - num_pos
    scale_pos_weight = num_neg / num_pos

    model = lgb.LGBMClassifier(
        is_unbalance = False,
        scale_pos_weight=scale_pos_weight,
        num_leaves = 31,
        learning_rate = 0.05,
        n_estimators = 100,
        random_state = 42

    )
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    train_model()

