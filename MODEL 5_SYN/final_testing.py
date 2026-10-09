import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from lightgbm.callback import record_evaluation
from lightgbm.callback import early_stopping


# =========================
# PATHS (EDIT IF NEEDED)
# =========================
CSV_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/preprocessed.csv"   # <-- YOUR CSV
OUTPUT_DIR = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)
print("Total rows:", len(df))

# =========================
# SPLIT: 70 / 15 / 15
# =========================
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# =========================
# TEXT FEATURES (TF-IDF)
# =========================
tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))

X_train_text = tfidf.fit_transform(
    train_df["subject_clean"].fillna("") + " " + train_df["body_clean"].fillna("")
)
X_val_text = tfidf.transform(
    val_df["subject_clean"].fillna("") + " " + val_df["body_clean"].fillna("")
)

X_test_text = tfidf.transform(
    test_df["subject_clean"].fillna("") + " " + test_df["body_clean"].fillna("")
)

# =========================
# NUMERIC FEATURES
# =========================
numeric_cols = [
    "num_redirect_tokens",
    "contains_redirect",
    "has_credential_request",
    "has_attached_word"
]

scaler = StandardScaler()

X_train_num = scaler.fit_transform(train_df[numeric_cols])
X_val_num   = scaler.transform(val_df[numeric_cols])
X_test_num  = scaler.transform(test_df[numeric_cols])

X_train = hstack([X_train_text, csr_matrix(X_train_num)])
X_val   = hstack([X_val_text, csr_matrix(X_val_num)])
X_test  = hstack([X_test_text, csr_matrix(X_test_num)])

y_train = train_df["label"].values
y_val   = val_df["label"].values
y_test  = test_df["label"].values

# =========================
# TRAIN LIGHTGBM
# =========================
model = lgb.LGBMClassifier(
    objective="binary",
    learning_rate=0.05,
    num_leaves=64,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42
)

evals_result = {}

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric="binary_logloss",
    callbacks=[
        record_evaluation(evals_result),
        early_stopping(stopping_rounds=50, verbose=False)
    ]
)

# =========================
# PLOT TRAINING CURVES
# =========================
train_loss = evals_result["training"]["binary_logloss"]
val_loss   = evals_result["valid_1"]["binary_logloss"]

# ---- LOSS PLOT ----
plt.figure(figsize=(8, 4))
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Binary Log Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loss.png")
plt.close()

# ---- ACCURACY PLOT ----
train_acc = []
val_acc = []

for i in range(1, len(train_loss) + 1):
    model.set_params(n_estimators=i)

    train_preds = (model.predict_proba(X_train)[:, 1] >= 0.5).astype(int)
    val_preds   = (model.predict_proba(X_val)[:, 1] >= 0.5).astype(int)

    train_acc.append(accuracy_score(y_train, train_preds))
    val_acc.append(accuracy_score(y_val, val_preds))

plt.figure(figsize=(8, 4))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_accuracy.png")
plt.close()

# =========================
# FINAL EVALUATION
# =========================
def evaluate(X, y, name):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        f"{name}_accuracy": accuracy_score(y, preds),
        f"{name}_precision": precision_score(y, preds),
        f"{name}_recall": recall_score(y, preds),
        f"{name}_f1": f1_score(y, preds)
    }

metrics = {}
metrics.update(evaluate(X_train, y_train, "train"))
metrics.update(evaluate(X_val, y_val, "validation"))
metrics.update(evaluate(X_test, y_test, "test"))

# =========================
# SAVE OUTPUTS
# =========================
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(model, OUTPUT_DIR / "model.pkl")
joblib.dump(tfidf, OUTPUT_DIR / "tfidf_vectorizer.joblib")
joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")

print("\n=== FINAL METRICS ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\nSaved outputs to:", OUTPUT_DIR)