import os, numpy as np, pandas as pd, kagglehub, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, make_scorer,
                             precision_recall_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- Data loading (same as run_day1_baseline.py) ---
cached_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_file = os.path.join(cached_path, "creditcard.csv")
df = pd.read_csv(csv_file)

amounts = df['Amount'].values.reshape(-1, 1)
features_matrix = df[[f'V{i}' for i in range(1, 29)]].values
y = df['Class'].values.astype(int)
X = np.hstack((amounts, features_matrix))

def engineer_features(X, amount_col_idx=0):
    amounts = X[:, amount_col_idx]
    amount_log = np.log1p(amounts).reshape(-1, 1)
    mean_amt = amounts.mean()
    std_amt = amounts.std()
    amount_zscore = (np.zeros_like(amounts) if std_amt == 0
                     else (amounts - mean_amt) / std_amt).reshape(-1, 1)
    return np.hstack((X, amount_log, amount_zscore))

X = engineer_features(X, amount_col_idx=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

n_legit = sum(y_train_res == 0)
n_fraud = sum(y_train_res == 1)
scale_weight = n_legit / max(n_fraud, 1)

# --- Train baseline (same grid as Day 1) ---
base_clf = XGBClassifier(objective='binary:logistic', eval_metric='aucpr',
                         random_state=42, n_jobs=-1, tree_method='hist')
param_grid = {
    'max_depth': [4, 6, 8], 'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1], 'scale_pos_weight': [1.0, scale_weight],
}
pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=base_clf, param_grid=param_grid,
                           scoring=pr_auc_scorer, cv=cv, verbose=1,
                           n_jobs=-1, refit=True)
print("Running GridSearchCV (baseline)...")
grid_search.fit(X_train_res, y_train_res)
best_clf = grid_search.best_estimator_

# --- Step 1: Threshold tuning ---
y_probs = best_clf.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Plot precision vs recall vs threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
plt.plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
plt.xlabel('Threshold'); plt.ylabel('Score'); plt.title('Precision & Recall vs Threshold')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('threshold_tuning_plot.png', dpi=150)
print("Saved: threshold_tuning_plot.png")

# Find best threshold where precision >= 0.6
precision_floor = 0.6
valid_mask = precisions[:-1] >= precision_floor
if valid_mask.any():
    best_idx = np.argmax(recalls[:-1][valid_mask])
    valid_thresholds = thresholds[valid_mask]
    best_threshold = valid_thresholds[best_idx]
else:
    best_threshold = 0.5

print(f"\n{'='*50}")
print(f"Precision floor: {precision_floor}")
print(f"Best threshold:  {best_threshold:.4f}")

# Compare default vs tuned threshold
for label, thresh in [("Default (0.5)", 0.5), (f"Tuned ({best_threshold:.4f})", best_threshold)]:
    y_pred = (y_probs >= thresh).astype(int)
    print(f"\n--- {label} ---")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_test, y_probs):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
