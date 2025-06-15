import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import pandas as pd
import pickle 
from src.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier


def main():
    # Load and preprocess data
    df = pd.read_csv("data/Train.csv")
    df = preprocess_data(df)

    # Split features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Initialize CatBoost model
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.03,
        depth=5,
        class_weights=[2, 1],  # Handling class imbalance
        random_seed=42,
        verbose=0
    )

    # Fit model
    model.fit(X_train, y_train)

    # Predict with threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    cat_thresh = 0.36
    y_pred = (y_proba >= cat_thresh).astype(int)

    # Print classification metrics
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

   
    # Save model
    with open("models/catboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(" Model saved to 'models/catboost_model.pkl'")


if __name__ == "__main__":
    main()
