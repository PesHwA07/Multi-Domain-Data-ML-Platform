# %%
import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.metrics import average_precision_score
from pycaret.classification import setup, create_model, tune_model, add_metric, pull

print("Fetching Credit Card Fraud dataset...")
cached_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_file = os.path.join(cached_path, "creditcard.csv")

print("Loading data...")
df = pd.read_csv(csv_file)

# Recreate the data structure expected by the original pipeline
amounts = df['Amount'].values.reshape(-1, 1)
features_matrix = df[[f'V{i}' for i in range(1, 29)]].values
y = df['Class'].values.astype(int)
X = np.hstack((amounts, features_matrix))

# Engineer features
def engineer_features(X, amount_col_idx=0):
    amounts = X[:, amount_col_idx]
    amount_log = np.log1p(amounts).reshape(-1, 1)
    mean_amt = amounts.mean()
    std_amt = amounts.std()
    amount_zscore = (np.zeros_like(amounts) if std_amt == 0 else (amounts - mean_amt) / std_amt).reshape(-1, 1)
    return np.hstack((X, amount_log, amount_zscore))

print("Engineering velocity features...")
X = engineer_features(X, amount_col_idx=0)

# Build a dataframe for PyCaret
cols = ['Amount'] + [f'V{i}' for i in range(1, 29)] + ['Amount_log', 'Amount_zscore']
df_processed = pd.DataFrame(X, columns=cols)
df_processed['is_fraud'] = y

print("Setting up PyCaret...")
# Initialize PyCaret setup
clf_setup = setup(
    data=df_processed,
    target='is_fraud',
    fix_imbalance=True,
    session_id=42,
    verbose=False
)

add_metric('pr_auc', 'PR-AUC', average_precision_score, target='pred_proba')

print("Creating Extra Trees Classifier (the best model from Day 2)...")
et_model = create_model('et')

print("Tuning Extra Trees Classifier, optimizing for PR-AUC...")
# n_iter=10 specifies 10 iterations of random grid search
tuned_et = tune_model(et_model, optimize='PR-AUC', n_iter=10)

print("\n--- Final Metrics for Tuned Model ---")
results = pull()
print(results)

print("\n--- Tuned Hyperparameters ---")
print(tuned_et)
