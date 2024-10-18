# src/train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import load_data, preprocess_data, split_data

def train_model(train_file):
    # Load and preprocess data
    data = load_data(train_file)
    data = preprocess_data(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Feature importance
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print("\nFeature Importances:")
    print(feature_importances.sort_values(ascending=False))
    
    # Save the model
    pd.to_pickle(model, 'models/model.pkl')
    
if __name__ == "__main__":
    train_model('data/train.csv')
