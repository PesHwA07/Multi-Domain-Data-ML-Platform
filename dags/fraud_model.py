import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Database connection URL for the Airflow container environment
DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"

def preprocess_and_smote():
    """
    Day 16: Preprocessing & SMOTE
    Fetches raw transaction data, expands the PostgreSQL feature array, 
    splits into Train/Test, and applies SMOTE ONLY to the training set 
    to handle severe class imbalance without data leakage.
    """
    print("Connecting to database to fetch fraud transactions...")
    engine = create_engine(DB_URL)
    
    df = pd.read_sql("SELECT amount, features, is_fraud FROM fraud.transactions", engine)
    
    if len(df) == 0:
        print("No data found in fraud.transactions. Please run the ETL script first.")
        return None, None, None, None
        
    print(f"Loaded {len(df)} transactions.")
    
    # The features column is a PostgreSQL array (mapped to a Python list by psycopg2). 
    # We expand it into a dense 2D numpy array for scikit-learn.
    print("Expanding PCA feature array...")
    features_matrix = np.stack(df['features'].values)
    
    # X matrix combines 'amount' and the 28 PCA features
    amount_array = df['amount'].values.reshape(-1, 1)
    X = np.hstack((amount_array, features_matrix))
    y = df['is_fraud'].values.astype(int)
    
    # Train/Test Split (80/20)
    # CRITICAL: We use 'stratify=y' to ensure the test set accurately reflects the original imbalance ratio
    print("Splitting dataset (80% Train, 20% Test) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Original Training Distribution -> Legit: {sum(y_train==0)}, Fraud: {sum(y_train==1)}")
    
    # Apply SMOTE (Synthetic Minority Over-sampling Technique)
    # CRITICAL: We only apply this to the training data! Applying before the split leaks data.
    print("Applying SMOTE to training data to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled Training Distribution -> Legit: {sum(y_train_resampled==0)}, Fraud: {sum(y_train_resampled==1)}")
    
    # Return the balanced training set and the untouched testing set
    return X_train_resampled, X_test, y_train_resampled, y_test

if __name__ == "__main__":
    # Test execution block
    # preprocess_and_smote()
    pass
