import pandas as pd
import numpy as np
import joblib # To save the model efficiently

def optimize_floats(df):
    floats = df.select_dtypes(include=['float64']).columns
    df[floats] = df[floats].astype('float32')
    return df

def optimize_ints(df):
    ints = df.select_dtypes(include=['int64']).columns
    df[ints] = df[ints].astype('int32')
    return df

def generate_low_ram_data(n_records=1000):
    # Base data
    data = {
        'tenure_months': np.random.randint(1, 61, size=n_records, dtype='int32'),
        'cs_calls': np.random.poisson(lam=2.0, size=n_records).astype('int32'),
        'complains': np.random.choice([0, 1], size=n_records, p=[0.8, 0.2]).astype('int32'),
        'discount_usage': np.random.uniform(0, 1, size=n_records).astype('float32')
    }
    
    df = pd.DataFrame(data)

    # Logic-based features
    df['frequency'] = (np.random.poisson(lam=5 + (df['tenure_months'] / 12))).astype('int32')
    df['recency_days'] = np.random.randint(1, 150, size=n_records, dtype='int32')
    
    # Simulating Churn Logic
    # We use a simple linear combination to keep CPU usage low
    churn_prob = (
        0.4 * (df['recency_days'] / 150) + 
        0.4 * (df['cs_calls'] / 10) - 
        0.2 * (df['tenure_months'] / 60)
    )
    df['churn_status'] = (churn_prob > 0.5).astype('int32')

    # Memory Optimization
    df = optimize_floats(df)
    df = optimize_ints(df)
    
    return df

# Generate and Save
df = generate_low_ram_data(2000) # Increased to 2k, still tiny for 4GB
df.to_csv('churn_data.csv', index=False)

print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print("Data Generated Successfully.")