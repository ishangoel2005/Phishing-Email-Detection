# train_val_test_full_pipeline.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from lightgbm.callback import record_evaluation, early_stopping

# -----------------------
# EDIT THIS
# -----------------------
CSV_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/preprocessed.csv"   # <-- your single CSV path here
OUTPUT_DIR = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/final output/final_output2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------

# -----------------------
# Load CSV and basic checks
# -----------------------
df = pd.read_csv(CSV_PATH)
print("Total rows:", len(df))

# Basic required columns check
required = {'sender', 'subject_clean', 'body_clean', 'label',
            'num_redirect_tokens', 'contains_redirect', 'has_credential_request', 'has_attached_word'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Input CSV missing required columns: {missing}")

# Replace NaNs in text fields with empty string (very important)
df['subject_clean'] = df['subject_clean'].fillna("")
df['body_clean'] = df['body_clean'].fillna("")

# -----------------------
# Train / Val / Test split (70/15/15) stratified
# -----------------------
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Optional: print label distribution
print("Label dist (train):\n", train_df['label'].value_counts(normalize=True).to_dict())
print("Label dist (val):\n", val_df['label'].value_counts(normalize=True).to_dict())
print("Label dist (test):\n", test_df['label'].value_counts(normalize=True).to_dict())

# -----------------------
# Text features: TF-IDF (subject_clean + body_clean)
# -----------------------
tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2)

X_train_text = tfidf.fit_transform((train_df["subject_clean"] + " " + train_df["body_clean"]).astype(str))
X_val_text   = tfidf.transform((val_df["subject_clean"] + " " + val_df["body_clean"]).astype(str))
X_test_text  = tfidf.transform((test_df["subject_clean"] + " " + test_df["body_clean"]).astype(str))

# -----------------------
# Numeric features
# -----------------------
numeric_cols = [
    "num_redirect_tokens",
    "contains_redirect",
    "has_credential_request",
    "has_attached_word"
]

# If some numeric cols are missing or not in expected dtype, coerce
for c in numeric_cols:
    if c not in train_df.columns:
        train_df[c] = 0
        val_df[c] = 0
        test_df[c] = 0

scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_df[numeric_cols].fillna(0).astype(float))
X_val_num   = scaler.transform(val_df[numeric_cols].fillna(0).astype(float))
X_test_num  = scaler.transform(test_df[numeric_cols].fillna(0).astype(float))

# Combine sparse tfidf + dense numeric -> sparse
X_train = hstack([X_train_text, csr_matrix(X_train_num)], format='csr')
X_val   = hstack([X_val_text, csr_matrix(X_val_num)], format='csr')
X_test  = hstack([X_test_text, csr_matrix(X_test_num)], format='csr')

y_train = train_df["label"].astype(int).values
y_val   = val_df["label"].astype(int).values
y_test  = test_df["label"].astype(int).values

print("Feature shapes:", X_train.shape, X_val.shape, X_test.shape)

# -----------------------
# Train LightGBM with callbacks (early stopping + record_evaluation)
# -----------------------
LGB_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 1000,
    'random_state': 42,
    'verbosity': -1
}

model = lgb.LGBMClassifier(**LGB_PARAMS)

evals_result = {}
# Use callbacks for compatibility
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='binary_logloss',
    callbacks=[record_evaluation(evals_result), early_stopping(stopping_rounds=50, verbose=False)],
)

# -----------------------
# Plot training vs validation loss and accuracy per iteration
# -----------------------
# evals_result keys may be like 'training' and 'valid_1' or 'training' and 'valid_0'.
# Print keys for clarity
print("evals_result keys:", list(evals_result.keys()))

# Attempt to find train/val keys robustly
eval_keys = list(evals_result.keys())
train_key = eval_keys[0]  # usually 'training'
val_key = eval_keys[1] if len(eval_keys) > 1 else None

# Binary logloss arrays
train_loss = evals_result[train_key].get('binary_logloss', [])
val_loss = evals_result[val_key].get('binary_logloss', []) if val_key else []

# Plot Loss
plt.figure(figsize=(8,4))
plt.plot(train_loss, label="Train Loss")
if len(val_loss) > 0:
    plt.plot(val_loss, label="Val Loss")
plt.xlabel("Iteration")
plt.ylabel("Binary Log Loss")
plt.title("Train vs Val Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loss.png")
plt.close()

# Plot Accuracy per iteration (compute by setting n_estimators progressively)
train_acc = []
val_acc = []
n_iters = len(train_loss) if len(train_loss)>0 else getattr(model, "n_estimators", 1)

# To avoid extremely slow loops on very large models, cap to reasonable iterations if huge
MAX_PLOT_ITERS = 500
iters_to_eval = min(n_iters, MAX_PLOT_ITERS)

for i in range(1, iters_to_eval + 1):
    model.set_params(n_estimators=i)
    # predict_proba on sparse matrices; may be heavy for large datasets
    train_preds = (model.predict_proba(X_train)[:,1] >= 0.5).astype(int)
    train_acc.append(accuracy_score(y_train, train_preds))
    if X_val.shape[0] > 0:
        val_preds = (model.predict_proba(X_val)[:,1] >= 0.5).astype(int)
        val_acc.append(accuracy_score(y_val, val_preds))

plt.figure(figsize=(8,4))
plt.plot(train_acc, label="Train Accuracy")
if len(val_acc) > 0:
    plt.plot(val_acc, label="Val Accuracy")
plt.xlabel("Iteration (capped)")
plt.ylabel("Accuracy")
plt.title("Train vs Val Accuracy (per iteration)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_accuracy.png")
plt.close()

# -----------------------
# Final evaluation: metrics for train, val, test using model (with chosen n_estimators)
# -----------------------
# If model has best_iteration_, use it; else use current n_estimators
best_iter = getattr(model, "best_iteration_", None)
if best_iter is not None and best_iter > 0:
    model.set_params(n_estimators=best_iter)
else:
    # keep as-is
    pass

def eval_set(X, y):
    probs = model.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, preds).tolist()
    }

results = {
    "train": eval_set(X_train, y_train),
    "validation": eval_set(X_val, y_val),
    "test": eval_set(X_test, y_test)
}

# Save metrics and classification reports
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

# Save classification reports separately
(OUTPUT_DIR / "classification_report_train.json").write_text(json.dumps(classification_report(y_train, (model.predict_proba(X_train)[:,1]>=0.5).astype(int), output_dict=True, zero_division=0), indent=2))
(OUTPUT_DIR / "classification_report_val.json").write_text(json.dumps(classification_report(y_val, (model.predict_proba(X_val)[:,1]>=0.5).astype(int), output_dict=True, zero_division=0), indent=2))
(OUTPUT_DIR / "classification_report_test.json").write_text(json.dumps(classification_report(y_test, (model.predict_proba(X_test)[:,1]>=0.5).astype(int), output_dict=True, zero_division=0), indent=2))

# Save model and preprocessors
joblib.dump(model, OUTPUT_DIR / "model.pkl")
joblib.dump(tfidf, OUTPUT_DIR / "tfidf_vectorizer.joblib")
joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")

# Save a small summary to console
print("\n=== FINAL METRICS SUMMARY ===")
for split in ["train", "validation", "test"]:
    m = results[split]
    print(f"{split.upper()}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}")

print("\nSaved artifacts and plots to:", OUTPUT_DIR)