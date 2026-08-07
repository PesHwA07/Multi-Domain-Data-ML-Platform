import os, numpy as np, pandas as pd, kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

cached_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(os.path.join(cached_path, "creditcard.csv"))
amounts = df['Amount'].values.reshape(-1, 1)
features_matrix = df[[f'V{i}' for i in range(1, 29)]].values
y = df['Class'].values.astype(int)
X = np.hstack((amounts, features_matrix))

def engineer_features(X, amount_col_idx=0):
    amounts = X[:, amount_col_idx]
    amount_log = np.log1p(amounts).reshape(-1, 1)
    mean_amt = amounts.mean(); std_amt = amounts.std()
    amount_zscore = (np.zeros_like(amounts) if std_amt == 0
                     else (amounts - mean_amt) / std_amt).reshape(-1, 1)
    return np.hstack((X, amount_log, amount_zscore))

X = engineer_features(X, amount_col_idx=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train XGBoost (use best params from earlier steps)
print("Training XGBoost...")
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='aucpr',
                          max_depth=6, n_estimators=200, learning_rate=0.1,
                          scale_pos_weight=1.0, random_state=42,
                          n_jobs=-1, tree_method='hist')
xgb_model.fit(X_train_res, y_train_res)

# Train Extra Trees
print("Training Extra Trees...")
et_model = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
et_model.fit(X_train_res, y_train_res)

# Get probabilities
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
et_probs = et_model.predict_proba(X_test)[:, 1]

# Evaluate individual + ensemble
def evaluate(name, probs, threshold=0.5):
    y_pred = (probs >= threshold).astype(int)
    print(f"\n--- {name} (threshold={threshold}) ---")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_test, probs):.4f}")

evaluate("XGBoost (alone)", xgb_probs)
evaluate("Extra Trees (alone)", et_probs)

# Simple average
ensemble_avg = (xgb_probs + et_probs) / 2
evaluate("Ensemble (50/50)", ensemble_avg)

# Weighted average (70% XGB, 30% ET)
ensemble_weighted = 0.7 * xgb_probs + 0.3 * et_probs
evaluate("Ensemble (70/30 XGB/ET)", ensemble_weighted)

# Disagreement analysis
xgb_pred = (xgb_probs >= 0.5).astype(int)
et_pred = (et_probs >= 0.5).astype(int)
disagree = xgb_pred != et_pred
print(f"\nDisagreement rate: {disagree.mean()*100:.2f}% ({disagree.sum()} samples)")
