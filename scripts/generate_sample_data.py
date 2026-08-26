import pandas as pd
import numpy as np
np.random.seed(42)

n = 7043

data = {
    'customerID': [f'CUST-{i:05d}' for i in range(n)],
    'gender': np.random.choice(['Male', 'Female'], n),
    'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
    'Partner': np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52]),
    'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70]),
    'tenure': np.random.randint(0, 73, n),
    'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.42, 0.48, 0.10]),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.28, 0.50, 0.22]),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.27, 0.51, 0.22]),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.28, 0.50, 0.22]),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.27, 0.51, 0.22]),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.30, 0.48, 0.22]),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.30, 0.48, 0.22]),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41]),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n, p=[0.34, 0.23, 0.22, 0.21]),
    'MonthlyCharges': np.round(np.random.uniform(18.25, 118.75, n), 2),
    'TotalCharges': np.round(np.random.uniform(18.80, 8684.80, n), 2),
}

df = pd.DataFrame(data)

churn_prob = np.zeros(n)
for i in range(n):
    p = 0.15
    if df.loc[i, 'Contract'] == 'Month-to-month':
        p += 0.12
    if df.loc[i, 'tenure'] < 12:
        p += 0.10
    if df.loc[i, 'InternetService'] == 'Fiber optic':
        p += 0.06
    if df.loc[i, 'PaymentMethod'] == 'Electronic check':
        p += 0.06
    if df.loc[i, 'OnlineSecurity'] == 'No':
        p += 0.03
    if df.loc[i, 'TechSupport'] == 'No':
        p += 0.03
    churn_prob[i] = min(p, 0.85)

df['Churn'] = ['Yes' if np.random.random() < churn_prob[i] else 'No' for i in range(n)]

df.to_csv('data/raw/telco_customer_churn.csv', index=False)
print(f'Dataset created: {df.shape}')
print(f'Churn rate: {df["Churn"].value_counts(normalize=True).to_dict()}')
