from airflow import DAG
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import os
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
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

    df = pd.read_sql(
        "SELECT amount, features, is_fraud FROM fraud.transactions", engine)

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

    print(
        f"Original Training Distribution -> Legit: {sum(y_train==0)}, Fraud: {sum(y_train==1)}")

    # Apply SMOTE (Synthetic Minority Over-sampling Technique)
    # CRITICAL: We only apply this to the training data! Applying before the split leaks data.
    print("Applying SMOTE to training data to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(
        f"Resampled Training Distribution -> Legit: {sum(y_train_resampled==0)}, Fraud: {sum(y_train_resampled==1)}")

    # Return the balanced training set and the untouched testing set
    return X_train_resampled, X_test, y_train_resampled, y_test


def train_and_evaluate_fraud_model():
    """
    Day 17: Modeling & Evaluation
    Trains a Random Forest classifier on the SMOTE-balanced training data,
    evaluates it on the untouched test set using strict fraud metrics (F1, PR-AUC),
    and saves the model artifact for the FastAPI serving layer.
    """
    X_train_resampled, X_test, y_train_resampled, y_test = preprocess_and_smote()

    if X_train_resampled is None:
        return

    print("Initializing Random Forest Classifier...")
    # Limiting depth and estimators to keep Airflow task execution time reasonable
    clf = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

    print("Fitting model to SMOTE-balanced training data (this may take a moment)...")
    clf.fit(X_train_resampled, y_train_resampled)

    print("Generating predictions on untouched test set...")
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1]

    print("Calculating strict fraud evaluation metrics...")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_probs)

    print(f"--- Fraud Model Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"---------------------------")

    # Save the model artifact to the shared data volume for FastAPI to consume
    model_path = "/opt/airflow/data/fraud_rf_model.joblib"
    if not os.path.exists(os.path.dirname(model_path)):
        # Local fallback
        model_path = os.path.join(os.path.dirname(
            __file__), '../data/fraud_rf_model.joblib')

    print(f"Saving model artifact to {model_path}...")
    joblib.dump(clf, model_path)
    print("Model successfully saved!")

    return clf


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# Wrap the model retraining process into an automated weekly DAG
with DAG(
    'fraud_model_retraining_weekly',
    default_args=default_args,
    description='A weekly DAG to retrain the Random Forest fraud classification model with SMOTE',
    schedule_interval='@weekly',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['fraud', 'ml', 'random-forest'],
) as dag:

    retrain_model_task = PythonOperator(
        task_id='train_and_evaluate_fraud_model',
        python_callable=train_and_evaluate_fraud_model,
    )
