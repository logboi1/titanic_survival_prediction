# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Fill missing values in 'Age' with median
    data['Age'] = data['Age'].fillna(data['Age'].median())
    
    # Fill missing values in 'Embarked' with the most common value
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # Drop the 'Cabin' column as it has too many missing values
    data = data.drop(columns=['Cabin', 'Ticket', 'Name'])
    
    # Encode 'Sex' and 'Embarked' columns
    le = LabelEncoder()
    data['Sex'] = le.fit_transform(data['Sex'])
    data['Embarked'] = le.fit_transform(data['Embarked'])
    
    return data

def split_data(data):
    X = data.drop(columns='Survived')
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
