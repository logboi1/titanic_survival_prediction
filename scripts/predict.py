# src/predict.py
import pandas as pd
from preprocess import preprocess_data

def load_model(model_path):
    model = pd.read_pickle(model_path)
    return model

def predict_survival(test_file, model_file):
    # Load and preprocess test data
    data = pd.read_csv(test_file)
    ids = data['PassengerId']
    data = preprocess_data(data)
    
    # Load the trained model
    model = load_model(model_file)
    
    # Make predictions
    predictions = model.predict(data)
    
    # Output results
    results = pd.DataFrame({
        'PassengerId': ids,
        'Survived': predictions
    })
    results.to_csv('data/predictions.csv', index=False)
    print("Predictions saved to data/predictions.csv")

if __name__ == "__main__":
    predict_survival('data/test.csv', 'models/model.pkl')
