import pandas as pd
import numpy as np

def generate_credit_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 200000, n_samples),
        'employment_years': np.random.uniform(0, 30, n_samples),
        'debt': np.random.randint(0, 50000, n_samples),
        'credit_lines': np.random.randint(0, 10, n_samples),
        'loan_amount': np.random.randint(1000, 100000, n_samples),
    }
    df = pd.DataFrame(data)
    # Create target (default) based on a non-linear rule: debt/income > 0.4 AND credit_lines > 5
    df['debt_to_income'] = df['debt'] / df['income']
    df['default'] = ((df['debt_to_income'] > 0.4) & (df['credit_lines'] > 5)).astype(int)
    # Add 5% random noise
    noise = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    df['default'] = np.maximum(df['default'], noise)
    return df.drop('debt_to_income', axis=1)

if __name__ == '__main__':
    df = generate_credit_data()
    print(df.head())
    print("\nDefault rate:", df['default'].mean())