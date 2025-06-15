import pandas as pd
from src.preprocessing import preprocess_data

df = pd.read_csv('data/Train.csv')
df_cleaned = preprocess_data(df)
df_cleaned.to_csv('data/cleaned_loan_data.csv', index=False)
