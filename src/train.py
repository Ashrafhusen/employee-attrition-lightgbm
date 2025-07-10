import joblib 
import lightgbm as lgb 
from sklearn.model_selection import train_test_split
from src.preprocess import load_data 

def train_model():
    df = load_data()
    x = df.drop("Attrition", axis = 1)
    y = df['Attrition']

    X_train,_,y_train,_ = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    train_model()

    