import os, numpy as np, pandas as pd, kagglehub
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, make_scorer)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

cached_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(os.path.join(cached_path, "creditcard.csv"))

# --- Extended feature engineering ---
print("Building extended feature set...")

# Base features
amounts = df['Amount'].values
time_vals = df['Time'].values
pca_features = df[[f'V{i}' for i in range(1, 29)]].values
y = df['Class'].values.astype(int)

# Original velocity features
amount_log = np.log1p(amounts)
mean_amt = amounts.mean(); std_amt = amounts.std()
amount_zscore = (amounts - mean_amt) / std_amt

# NEW: Ratio features — amount / rolling mean (approximated as global mean here)
amount_ratio = amounts / (mean_amt + 1e-8)

# NEW: Recency — time since previous transaction (sorted by Time)
time_diff = np.diff(time_vals, prepend=time_vals[0])

# NEW: Interaction terms — top SHAP features are typically V14, V17, V12
# (common knowledge for this dataset)
interaction_v14_amount = pca_features[:, 13] * amounts   # V14 x Amount
interaction_v17_v12 = pca_features[:, 16] * pca_features[:, 11]  # V17 x V12

X = np.column_stack([
    amounts, pca_features,
    amount_log, amount_zscore,       # existing velocity features
    amount_ratio,                     # NEW ratio feature
    time_diff,                        # NEW recency feature
    interaction_v14_amount,           # NEW interaction
    interaction_v17_v12,              # NEW interaction
])

print(f"Extended feature matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Use best params from Steps 1-2 (adjust if your results differ)
base_clf = XGBClassifier(objective='binary:logistic', eval_metric='aucpr',
                         random_state=42, n_jobs=-1, tree_method='hist',
                         max_depth=6, n_estimators=200, learning_rate=0.1)

pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Quick search over scale_pos_weight with new features
ratio = (y_train == 0).sum() / (y_train == 1).sum()
param_grid = {'scale_pos_weight': [1.0, ratio, ratio * 2.0, ratio * 5.0]}
grid_search = GridSearchCV(estimator=base_clf, param_grid=param_grid,
                           scoring=pr_auc_scorer, cv=cv, verbose=1,
                           n_jobs=-1, refit=True)

print("Training with extended features...")
grid_search.fit(X_train_res, y_train_res)

print(f"\nBest params: {grid_search.best_params_}")
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)
y_probs = best_clf.predict_proba(X_test)[:, 1]

print(f"\n--- Test Metrics (extended features) ---")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_probs):.4f}")
