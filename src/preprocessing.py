import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def preprocess_data(df):
    df = df.copy()

    # Fill missing values
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

    # Label Encoding
    le = LabelEncoder()
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed']:
        df[col] = le.fit_transform(df[col])

    # Encode Loan_Status if present (only during training)
    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})

    # One-Hot Encoding for Property_Area
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    area_encoded = ohe.fit_transform(df[['Property_Area']])
    encoded_cols = ohe.get_feature_names_out(['Property_Area'])
    encoded_df = pd.DataFrame(area_encoded, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop('Property_Area', axis=1), encoded_df], axis=1)

    # Drop ID column
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    # Feature engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Log_Total_Income'] = np.log(df['Total_Income'] + 1)
    df['Log_LoanAmount'] = np.log(df['LoanAmount'] + 1)
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['Balance_Income'] = df['Total_Income'] - df['EMI']
    df['Credit_Loan_Ratio'] = df['Credit_History'] * df['Log_LoanAmount']
    df['Married_Educated'] = df['Married'] * df['Education']
    df['SelfEmp_Dependents'] = df['Self_Employed'] * df['Dependents']
    df['Term_Income_Ratio'] = df['Loan_Amount_Term'] / df['Log_Total_Income']
    df['LoanAmount_Bin'] = pd.cut(
        df['Log_LoanAmount'],
        bins=[0, 3, 3.5, 5],
        labels=[0, 1, 2],
        include_lowest=True,
        right=True
    )
    df['LoanAmount_Bin'] = df['LoanAmount_Bin'].astype(float).fillna(-1).astype(int)

    # Drop now-unnecessary columns
    df.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income'], axis=1, inplace=True)

    return df
