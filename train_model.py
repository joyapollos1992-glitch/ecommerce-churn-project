import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import gc

def train_system():
    print("--- Phase 1: Loading Data ---")
    try:
        # Loading with specific types to save RAM immediately
        df = pd.read_csv('churn_data.csv')
    except FileNotFoundError:
        print("Error: 'churn_data.csv' not found. Please run your data generator script first.")
        return

    # 1. Feature/Target Split
    # features: tenure_months, cs_calls, complains, discount_usage, frequency, recency_days
    X = df.drop('churn_status', axis=1)
    y = df['churn_status']

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("--- Phase 2: Preprocessing (Scaling) ---")
    # Scaling is mandatory for Logistic Regression to perform well
    scaler = StandardScaler()
    
    # Fit only on training data to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("--- Phase 3: Training Lightweight Model ---")
    # Logistic Regression is extremely RAM-efficient
    # solver='lbfgs' is memory-optimized for small/medium datasets
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test_scaled)
    print("\n[MODEL PERFORMANCE]")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred))

    print("--- Phase 4: Exporting Artifacts ---")
    # Save the model and the scaler
    # Your dashboard NEEDS the scaler to process new user input
    joblib.dump(model, 'churn_model.pkl')
    joblib.dump(scaler, 'scalar.pkl')
    print("Saved: churn_model.pkl and scalar.pkl")

    # 5. Explicit Memory Cleanup
    print("\n--- Phase 5: Cleaning RAM ---")
    del df, X, y, X_train, X_test, X_train_scaled, X_test_scaled, model
    gc.collect()
    print("RAM released. Process complete.")

if __name__ == "__main__":
    train_system()