import pandas as pd
import pickle
import os
from src.preprocessing import preprocess_data

def main():
    # Load test data
    test_path = "data/Train.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError("Train.csv not found in /data")
    
    df= pd.read_csv(test_path)
    original_ids = df['Loan_ID']  # Save for output

    # Preprocess test data
    df = preprocess_data(df)

    # Drop target column 
    if 'Loan_Status' in df_test.columns:
        df_test = df_test.drop('Loan_Status', axis=1)

    # Load trained model from .pkl
    model_path = "models/catboost_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found at /models")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predict
    y_proba = model.predict_proba(df_test)[:, 1]
    threshold = 0.36
    y_pred = (y_proba >= threshold).astype(int)

    # Prepare output
    output = pd.DataFrame({
        'Loan_ID': original_ids,
        'Loan_Status_Predicted': y_pred
    })

    # Convert 0/1 to No or Yes
    output['Loan_Status_Predicted'] = output['Loan_Status_Predicted'].map({0: 'N', 1: 'Y'})

    # Save predictions
    os.makedirs("outputs", exist_ok=True)
    output.to_csv("outputs/predictions.csv", index=False)
    print("✅ Predictions saved to outputs/predictions.csv")


if __name__ == "__main__":
    main()
