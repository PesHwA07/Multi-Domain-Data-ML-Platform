"""
Fraud Detection Model Training DAG (XGBoost + GridSearchCV)
============================================================
This module handles preprocessing, SMOTE balancing, feature engineering,
hyperparameter tuning via GridSearchCV, and model evaluation for credit card
fraud detection. It is orchestrated as a weekly Airflow DAG.
"""

from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, make_scorer
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Database connection URL for the Airflow container environment
DB_URL = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"


def engineer_features(X, amount_col_idx=0):
    """
    Feature Engineering: Adds amount-based velocity features.
    - amount_log: Log-transformed amount (reduces skew, helps generalization)
    - amount_zscore: Z-score of amount (flags statistically extreme transactions)

    Parameters:
        X: numpy array where column 0 is 'amount'
        amount_col_idx: index of the amount column (default 0)
    Returns:
        X with two additional columns appended
    """
    amounts = X[:, amount_col_idx]

    # Log transform (add 1 to avoid log(0))
    amount_log = np.log1p(amounts).reshape(-1, 1)

    # Z-score normalization
    mean_amt = amounts.mean()
    std_amt = amounts.std()
    if std_amt == 0:
        amount_zscore = np.zeros_like(amounts).reshape(-1, 1)
    else:
        amount_zscore = ((amounts - mean_amt) / std_amt).reshape(-1, 1)

    return np.hstack((X, amount_log, amount_zscore))


def preprocess_and_smote():
    """
    Preprocessing & SMOTE
    Fetches raw transaction data, expands the PostgreSQL feature array,
    engineers velocity features, splits into Train/Test, and applies SMOTE
    ONLY to the training set to handle severe class imbalance without
    data leakage.
    """
    print("Connecting to database to fetch fraud transactions...")
    engine = create_engine(DB_URL)

    df = pd.read_sql(
        "SELECT amount, features, is_fraud FROM fraud.transactions", engine)

    if len(df) == 0:
        print("No data found in fraud.transactions. "
              "Please run the ETL script first.")
        return None, None, None, None

    print(f"Loaded {len(df)} transactions.")

    # The features column is a PostgreSQL array (mapped to a Python list).
    # Expand it into a dense 2D numpy array for scikit-learn.
    print("Expanding PCA feature array...")
    features_matrix = np.stack(df['features'].values)

    # X matrix combines 'amount' and the 28 PCA features
    amount_array = df['amount'].values.reshape(-1, 1)
    X = np.hstack((amount_array, features_matrix))
    y = df['is_fraud'].values.astype(int)

    # Feature Engineering: Add amount_log and amount_zscore
    print("Engineering velocity features (amount_log, amount_zscore)...")
    X = engineer_features(X, amount_col_idx=0)
    print(f"Feature matrix shape after engineering: {X.shape}")

    # Train/Test Split (80/20)
    # CRITICAL: stratify=y ensures the test set reflects the original
    # imbalance ratio
    print("Splitting dataset (80% Train, 20% Test) with stratification...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(
        f"Original Training Distribution -> "
        f"Legit: {sum(y_train == 0)}, Fraud: {sum(y_train == 1)}")

    # Apply SMOTE (Synthetic Minority Over-sampling Technique)
    # CRITICAL: Only apply to training data to prevent data leakage
    print("Applying SMOTE to training data to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train, y_train)

    print(
        f"Resampled Training Distribution -> "
        f"Legit: {sum(y_train_resampled == 0)}, "
        f"Fraud: {sum(y_train_resampled == 1)}")

    return X_train_resampled, X_test, y_train_resampled, y_test


def train_and_evaluate_fraud_model():
    """
    Model Training & Evaluation with XGBoost + GridSearchCV
    Trains an XGBoost classifier with automated hyperparameter tuning,
    evaluates on the untouched test set using strict fraud metrics
    (F1, PR-AUC), and saves the best model artifact for FastAPI serving.
    """
    X_train_resampled, X_test, y_train_resampled, y_test = (
        preprocess_and_smote()
    )

    if X_train_resampled is None:
        return

    # Calculate scale_pos_weight from the ORIGINAL (pre-SMOTE) class ratio
    # This gives XGBoost an additional signal about class imbalance
    n_legit = sum(y_train_resampled == 0)
    n_fraud = sum(y_train_resampled == 1)
    scale_weight = n_legit / max(n_fraud, 1)
    print(f"Calculated scale_pos_weight: {scale_weight:.2f}")

    # Define the XGBoost classifier
    print("Initializing XGBoost Classifier with GridSearchCV...")
    base_clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Fast histogram-based method
    )

    # Hyperparameter grid for GridSearchCV
    param_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'scale_pos_weight': [1.0, scale_weight],
    }

    # PR-AUC scorer (average_precision_score)
    pr_auc_scorer = make_scorer(
        average_precision_score,
        needs_proba=True,
        response_method='predict_proba'
    )

    # 3-fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring=pr_auc_scorer,
        cv=cv,
        verbose=1,
        n_jobs=-1,
        refit=True  # Automatically refit the best model on full train set
    )

    print("Running GridSearchCV (this may take several minutes)...")
    grid_search.fit(X_train_resampled, y_train_resampled)

    print(f"\n--- GridSearchCV Results ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV PR-AUC:  {grid_search.best_score_:.4f}")
    print(f"----------------------------\n")

    # Use the best estimator for final evaluation
    best_clf = grid_search.best_estimator_

    print("Generating predictions on untouched test set...")
    y_pred = best_clf.predict(X_test)
    y_probs = best_clf.predict_proba(X_test)[:, 1]

    print("Calculating strict fraud evaluation metrics...")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_probs)

    print(f"--- XGBoost Fraud Model Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"-----------------------------------")

    # Save the best model artifact to the shared data volume
    model_path = "/opt/airflow/data/fraud_xgb_model.joblib"
    if not os.path.exists(os.path.dirname(model_path)):
        model_path = os.path.join(os.path.dirname(
            __file__), '../data/fraud_xgb_model.joblib')

    print(f"Saving best XGBoost model to {model_path}...")
    joblib.dump(best_clf, model_path)
    print("Model successfully saved!")

    return best_clf


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
    description='Weekly DAG: XGBoost fraud classifier with GridSearchCV tuning',
    schedule_interval='@weekly',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['fraud', 'ml', 'xgboost', 'gridsearch'],
) as dag:

    retrain_model_task = PythonOperator(
        task_id='train_and_evaluate_fraud_model',
        python_callable=train_and_evaluate_fraud_model,
    )
