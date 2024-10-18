# Titanic Survival Prediction using Machine Learning

## Project Overview
This project applies supervised machine learning techniques to predict the survival of passengers aboard the Titanic based on various features such as age, gender, class, and fare.

## Folder Structure
- `data/`: Contains the Titanic dataset (`train.csv` and `test.csv`).
- `models/`: Contains the trained machine learning model (`model.pkl`).
- `scripts/`: Contains Python scripts for preprocessing data, training the model, and making predictions.
- `notebooks/`: Jupyter Notebook for exploratory data analysis.
- `requirements.txt`: Lists the required Python packages.

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

2. **Preprocess data and train the model**:
    ```bash
    python scripts/train.py

3. **Make predictions on new data**:
    ```bash
    python scripts/predict.py

4. **Perform exploratory data analysis**:
Open the ```notebooks/data_analysis.ipynb``` Jupyter Notebook for data visualization.


### Conclusion
This project provides a comprehensive implementation of Titanic survival prediction using supervised machine learning. The structured approach includes data preprocessing, model training, prediction, and exploratory analysis. You can extend the project by experimenting with other machine learning algorithms or optimizing hyperparameters further.
