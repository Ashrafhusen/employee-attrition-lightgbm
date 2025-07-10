from train import train_model 
import os 

def test_model_training():
    train_model()
    assert os.path.exists("models/model.pkl")

