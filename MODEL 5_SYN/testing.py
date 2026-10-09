# scripts/evaluate_on_csv.py
import re, html, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# ----------------------------
# EDIT PATHS IF NEEDED
# ----------------------------
TEST_CSV = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/unknown_test_data_200.csv")   # your test CSV with true labels
ARTIFACT_DIR = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts")
MODEL_DIR = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/model_artifacts")
OUTPUT_DIR = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------

# Load model + artifacts
model = joblib.load("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/model_artifacts/model.pkl")
tfidf = joblib.load("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/tfidf_vectorizer.joblib")
scaler = joblib.load("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/scaler.joblib")
domain_encoder = joblib.load("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/domain_encoder.joblib")
with open("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/feature_names.json", "r") as f:
    fnames = json.load(f)
numeric_names = fnames.get("numeric", [])
# load top_domains if needed
# load top_domains if needed (use Path)
top_domains = []
dm_path = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/domain_mapping.json")
if dm_path.exists():
    with dm_path.open("r", encoding="latin1") as f:
        top_domains = json.load(f).get("top_domains", [])


# decision threshold
thresh_path = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/model_artifacts/decision_threshold.txt")
if thresh_path.exists():
    DECISION_THRESHOLD = float(thresh_path.read_text(encoding="latin1").strip())
else:
    DECISION_THRESHOLD = 0.5


# ---------- helpers (match your preprocessing/predict pipeline) ----------
URL_RE = re.compile(r'(https?://\S+|\bwww\.\S+)', flags=re.IGNORECASE)
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', flags=re.IGNORECASE)
NUM_RE = re.compile(r'\b\d{3,}\b')
HTML_TAG_RE = re.compile(r'<[^>]+>')
MULTI_WS_RE = re.compile(r'\s+')
CRED_PATTERNS = re.compile(r'(password|credentials|verify.*account|enter.*password|login.*now|update.*payment|provide.*bank)', flags=re.IGNORECASE)

def normalize_text(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""
    text = html.unescape(str(text))
    text = HTML_TAG_RE.sub(' ', text)
    text = URL_RE.sub(' [url] ', text)
    text = EMAIL_RE.sub(' [email] ', text)
    text = NUM_RE.sub(' [num] ', text)
    text = MULTI_WS_RE.sub(' ', text).strip()
    return text.lower()

def extract_sender_parts(sender: str):
    s = str(sender or "").strip()
    if "@" in s:
        local, domain = s.split("@", 1)
        return local.lower(), domain.lower()
    else:
        return s.lower(), ""

def uppercase_ratio(text: str) -> float:
    if not text:
        return 0.0
    up = sum(1 for c in text if c.isupper())
    return up / max(1, len(text))

def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())

def punctuation_count(text: str) -> int:
    return sum(text.count(c) for c in "?!.,;:")

def count_token(text: str, token: str) -> int:
    if not text:
        return 0
    return text.count(token.lower())

def build_numeric_rows(df):
    rows = []
    for idx, r in df.iterrows():
        subj_clean = normalize_text(r['subject'])
        body_clean = normalize_text(r['body'])
        local, domain = extract_sender_parts(r['sender'])
        d = {}
        d['body_len'] = len(body_clean)
        d['subject_len'] = len(subj_clean)
        d['num_words'] = count_words(body_clean)
        d['exclamation_count'] = (body_clean.count('!') + subj_clean.count('!'))
        d['uppercase_ratio'] = uppercase_ratio(body_clean)
        d['punctuation_count'] = punctuation_count(body_clean)
        d['num_urls'] = count_token(body_clean, '[url]')
        d['num_redirect_tokens'] = count_token(body_clean, '[redirect_link]')
        d['has_request_for_credentials'] = int(bool(CRED_PATTERNS.search(body_clean)))
        d['has_attached_word'] = int(bool(re.search(r'\b(attached|attachment|see attached)\b', body_clean, flags=re.IGNORECASE)))
        d['sender_domain_freq'] = 0.0  # training-only info not available here
        domain_cat = domain if domain in top_domains else ("missing" if domain=="" else "other")
        try:
            enc_val = domain_encoder.transform([[domain_cat]])[0][0]
            d['sender_domain_encoded'] = int(enc_val)
        except Exception:
            d['sender_domain_encoded'] = -1
        d['localpart_has_digits'] = int(bool(re.search(r'\d', local)))
        d['localpart_len'] = len(local)
        d['localpart_contains_suspicious'] = int(bool(re.search(r'(admin|support|billing|no-reply|help|service)', local, flags=re.IGNORECASE)))
        # order
        row = [float(d.get(name, 0.0)) for name in numeric_names]
        rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, len(numeric_names)))

# ----------------------------
# Load test CSV
# ----------------------------
if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

df = pd.read_csv(TEST_CSV, encoding="latin1")
required_cols = {'sender','subject','body','label'}
if not required_cols.issubset(set(df.columns)):
    raise ValueError(f"Test CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}")

# build text TF-IDF
texts = (df['subject'].fillna("") + " " + df['body'].fillna("")).apply(lambda t: normalize_text(t)).tolist()
X_text = tfidf.transform(texts)   # sparse matrix (n_test x n_text_features)

# numeric matrix
numeric_arr = build_numeric_rows(df)   # shape (n_test, n_numeric)
numeric_scaled = scaler.transform(numeric_arr)
X_num = csr_matrix(numeric_scaled)

# combine
X = hstack([X_text, X_num], format='csr')

# predict
probs = model.predict_proba(X)[:,1]
preds = (probs >= DECISION_THRESHOLD).astype(int)

# metrics
y_true = df['label'].astype(int).values
acc = accuracy_score(y_true, preds)
prec = precision_score(y_true, preds, zero_division=0)
rec = recall_score(y_true, preds, zero_division=0)
f1 = f1_score(y_true, preds, zero_division=0)
report = classification_report(y_true, preds, zero_division=0, output_dict=True)
cm = confusion_matrix(y_true, preds)

# print results
print("Evaluation results on:", TEST_CSV)
print(f"Samples: {len(df)}")
print(f"Decision threshold: {DECISION_THRESHOLD}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nConfusion matrix (rows=true, cols=pred):")
print(cm)
print("\nClassification report:")
print(pd.DataFrame(report).T)

# save results CSV with probs and preds
out = df.copy()
out['pred_prob'] = probs
out['pred_label'] = preds
out.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
print(f"\nSaved predictions to: {OUTPUT_DIR/'test_predictions.csv'}")
# save summary metrics
summary = {
    "n": int(len(df)),
    "threshold": float(DECISION_THRESHOLD),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "confusion_matrix": cm.tolist()
}
with open(OUTPUT_DIR / "test_metrics.json","w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved metrics to: {OUTPUT_DIR/'test_metrics.json'}")