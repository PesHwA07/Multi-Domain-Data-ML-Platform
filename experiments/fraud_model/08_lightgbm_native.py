import os, numpy as np, pandas as pd, kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              average_precision_score)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

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

# Split validation from the RAW training data, BEFORE SMOTE
X_tr_raw, X_val, y_tr_raw, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

smote = SMOTE(random_state=42)
X_tr, y_tr = smote.fit_resample(X_tr_raw, y_tr_raw)

# Clean arrays for ALL sets, including test
X_tr = np.ascontiguousarray(np.asarray(X_tr), dtype=np.float64)
y_tr = np.ascontiguousarray(np.asarray(y_tr), dtype=np.float32)
X_val = np.ascontiguousarray(np.asarray(X_val), dtype=np.float64)
y_val = np.ascontiguousarray(np.asarray(y_val), dtype=np.float32)
X_test = np.ascontiguousarray(np.asarray(X_test), dtype=np.float64)

print("X_tr shape:", X_tr.shape, "dtype:", X_tr.dtype)

# --- DIAGNOSTIC: try a small subsample first, before the full dataset ---
print("\n--- Testing on a 2,000-row subsample first ---")
try:
    small_train = lgb.Dataset(X_tr[:2000], label=y_tr[:2000])
    small_model = lgb.train(
        {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'num_threads': 1, 'seed': 42},
        small_train,
        num_boost_round=20
    )
    print("Subsample (2,000 rows) succeeded — issue is likely dataset SIZE, not data itself.")
except Exception as e:
    print("Subsample also failed:", e)
    print("Issue is NOT size-related — likely a broken LightGBM install. Stop here and reinstall/downgrade.")
    raise SystemExit(1)

# --- If subsample worked, try progressively larger sizes to find the breaking point ---
for n in [10000, 50000, 100000, len(X_tr)]:
    print(f"\n--- Testing on {min(n, len(X_tr))} rows ---")
    try:
        test_train = lgb.Dataset(X_tr[:n], label=y_tr[:n])
        test_model = lgb.train(
            {'objective': 'binary', 'metric': 'auc', 'verbose': -1, 'num_threads': 1, 'seed': 42},
            test_train,
            num_boost_round=20
        )
        print(f"{min(n, len(X_tr))} rows succeeded.")
    except Exception as e:
        print(f"CRASHED at {min(n, len(X_tr))} rows:", e)
        print("This is your breaking point — likely a size/memory bug in this LightGBM build.")
        break
train_data = lgb.Dataset(X_tr, label=y_tr)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

base_params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': True,   # add this line
    'verbose': -1,
    'num_threads': 1,
    'seed': 42,
}

param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6],
    'num_leaves': [31, 63],
}

best_score = -1
best_params = None
best_model = None

for lr in param_grid['learning_rate']:
    for depth in param_grid['max_depth']:
        for leaves in param_grid['num_leaves']:
            params_try = {**base_params, 'learning_rate': lr, 'max_depth': depth, 'num_leaves': leaves}
            model = lgb.train(
                params_try,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            val_probs = model.predict(X_val, num_iteration=model.best_iteration)
            score = average_precision_score(y_val, val_probs)  # compute real PR-AUC via sklearn
            print(f"lr={lr}, depth={depth}, leaves={leaves} -> PR-AUC={score:.4f}")
            if score > best_score:
                best_score = score
                best_params = params_try
                best_model = model

print("\nBest params:", best_params)
print("Best validation PR-AUC:", best_score)

X_test_clean = np.ascontiguousarray(np.asarray(X_test), dtype=np.float64)
test_probs = best_model.predict(X_test_clean, num_iteration=best_model.best_iteration)
test_preds = (test_probs >= 0.5).astype(int)

print("\n--- Test Metrics (LightGBM, native API) ---")
print("Precision:", precision_score(y_test, test_preds))
print("Recall:", recall_score(y_test, test_preds))
print("F1:", f1_score(y_test, test_preds))
print("PR-AUC:", average_precision_score(y_test, test_probs))