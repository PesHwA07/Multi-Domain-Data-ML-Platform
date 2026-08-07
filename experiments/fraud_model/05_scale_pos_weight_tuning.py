import os, numpy as np, pandas as pd, kagglehub
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, make_scorer)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- Data loading ---
cached_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
df = pd.read_csv(os.path.join(cached_path, "creditcard.csv"))
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

from imblearn.pipeline import Pipeline as ImbPipeline

# --- Step 2: Widened scale_pos_weight ---
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance ratio: {ratio:.2f}")

param_grid = {
    'xgb__scale_pos_weight': [ratio * 0.5, ratio * 1.0, ratio * 2.0,
                         ratio * 5.0, ratio * 10.0],
    'xgb__max_depth': [6],         # use best from Day 1
    'xgb__n_estimators': [200],    # use best from Day 1
    'xgb__learning_rate': [0.1],   # use best from Day 1
}

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(objective='binary:logistic', eval_metric='aucpr',
                          random_state=42, n_jobs=-1, tree_method='hist'))
])

pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                           scoring=pr_auc_scorer, cv=cv, verbose=1,
                           n_jobs=-1, refit=True)
print("Running GridSearchCV with widened scale_pos_weight...")
grid_search.fit(X_train, y_train)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV PR-AUC: {grid_search.best_score_:.4f}")

best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)
y_probs = best_clf.predict_proba(X_test)[:, 1]

print(f"\n--- Test Metrics (widened scale_pos_weight) ---")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_probs):.4f}")
