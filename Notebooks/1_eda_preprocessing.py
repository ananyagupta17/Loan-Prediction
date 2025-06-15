#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
pip install pandas matplotlib seaborn numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("../Train.csv")

df.head()


# Display the shape of the dataset
df.info()

print("\n Column names:")
print(df.columns.tolist())

#identify numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
print(numerical_cols)

# Identify Categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("\nCategorical columns:")
print(categorical_cols)

# # Summary Statistics

print("\nDescriptive statistics for numerical columns:")
print(df[numerical_cols].describe())

# Value counts for categorical columns
print("\nValue counts for categorical columns (top 10 for brevity):")
for col in categorical_cols:
    print(f"\n--- Column: {col} ---")
# Displaying top 10 for columns with many unique values
    print(df[col].value_counts().head(10))


# # Missing Values

print("\n Missing values per column")
missing_values=df.isnull().sum()
print(missing_values)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(df['LoanAmount'].dropna(), bins=30, kde=True)
plt.title('LoanAmount Distribution')

plt.subplot(1,2,2)
sns.boxplot(x=df['LoanAmount'])
plt.title('LoanAmount Boxplot')

plt.show()

# the plot shows that we have right skewed data for loan amount so we will fill it using median

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

missing_values=df.isnull().sum()
print(missing_values)

# # Outlier detection using box plots

plt.figure(figsize=(15, 8))
for i, col in enumerate(numerical_cols):
    plt.subplot(5, 5, i + 1) # Adjust subplot grid based on number of numerical columns
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()


# # Univariate Analysis

#removing Loan_ID as it is an identifier
categorical_cols.remove('Loan_ID')
#for categorical columns
for col in categorical_cols:
    sns.countplot(data=df, x=col)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

#for numerical columns
for col in numerical_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# # Bivarariate Analysis- Categorical Vs Target

for col in categorical_cols[:-1]:  # exclude Loan_Status itself
    pd.crosstab(df[col], df['Loan_Status'], normalize='index').plot(kind='bar', stacked=True)
    plt.title(f'{col} vs Loan_Status')
    plt.ylabel('Proportion of Applicants')
    plt.show()

# Pairplot for scatter plots between all numerical columns
print("\nPairplot (Scatter plots for numerical columns):")
sns.pairplot(df[numerical_cols])
plt.suptitle('Pairplot of Numerical Features', y=1.02) # Add a title to the pairplot
plt.show()

#check correlation matrix
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Between Numerical Features')
plt.show()


# # Encoding the Data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le=LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Married'] = le.fit_transform(df['Married'])
df['Dependents'] = le.fit_transform(df['Dependents'])
df['Education'] = le.fit_transform(df['Education'])
df['Self_Employed'] = le.fit_transform(df['Self_Employed'])

df['Loan_Status'] = df['Loan_Status'].map({'N':0, 'Y':1})

data = df[['Property_Area']]

encoder = OneHotEncoder(sparse_output=False)

encoded_data = encoder.fit_transform(data)

encoded_columns = encoder.get_feature_names_out(['Property_Area'])

encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

df = pd.concat([df.drop('Property_Area', axis=1), encoded_df], axis=1)


# # Feature selection
df.head()
f = df.drop('Loan_ID', axis=1)
df.head()

#creating new features
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
#Smooth out income skew
df['Log_Total_Income'] = np.log(df['Total_Income'] + 1)
#Reduces skew in loan values
df['Log_LoanAmount'] = np.log(df['LoanAmount'] + 1)
#Estimate of monthly burden- LoanAmount / Loan_Amount_Term
df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
#Income left after EMI (will show ability to pay)
df['Balance_Income'] = df['Total_Income'] - df['EMI']

df.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income'], axis=1, inplace=True)

df.head()

df.to_csv('loan_data_cleaned.csv', index=False)





