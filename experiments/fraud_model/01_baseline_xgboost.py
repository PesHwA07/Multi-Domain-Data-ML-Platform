# %%
import os
import pandas as pd
import numpy as np
import kagglehub
import shutil
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, make_scorer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("Fetching Credit Card Fraud dataset...")
cached_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_file = os.path.join(cached_path, "creditcard.csv")

if not os.path.exists(csv_file):
    print("Dataset not found in cache. Something went wrong.")
    exit(1)

print("Loading data...")
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} transactions.")

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
print(f"Feature matrix shape: {X.shape}")

print("Splitting dataset (80% Train, 20% Test) with stratification...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Applying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

n_legit = sum(y_train_resampled == 0)
n_fraud = sum(y_train_resampled == 1)
scale_weight = n_legit / max(n_fraud, 1)

print("Initializing XGBoost Classifier with GridSearchCV...")
base_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

param_grid = {
    'max_depth': [4, 6, 8],
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'scale_pos_weight': [1.0, scale_weight],
}

pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_clf,
    param_grid=param_grid,
    scoring=pr_auc_scorer,
    cv=cv,
    verbose=1,
    n_jobs=-1,
    refit=True
)

print("Running GridSearchCV...")
grid_search.fit(X_train_resampled, y_train_resampled)

print(f"\n--- GridSearchCV Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV PR-AUC:  {grid_search.best_score_:.4f}")

best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)
y_probs = best_clf.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_probs)

print(f"--- Final Test Metrics ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
