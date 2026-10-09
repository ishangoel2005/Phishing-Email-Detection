# scripts/predict_loop.py
import re
import html
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix, load_npz
from sklearn.exceptions import NotFittedError
from pathlib import Path

# ----------------------------
# EDIT THESE PATHS IF NEEDED
# ----------------------------
ARTIFACT_DIR = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts")               # where tfidf/scaler/domain_encoder/feature_names live
MODEL_DIR = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/model_artifacts")            # where model.pkl, raw_lightgbm.pkl, oof_preds.npy live
# ----------------------------

# Load artifacts
tfidf = joblib.load("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/tfidf_vectorizer.joblib")
scaler = joblib.load("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/scaler.joblib")
domain_encoder = joblib.load("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/domain_encoder.joblib")
with open("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/domain_mapping.json", "r") as f:
    domain_meta = json.load(f)
top_domains = domain_meta.get("top_domains", [])

with open("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/feature_names.json", "r") as f:
    fnames = json.load(f)
numeric_names = fnames.get("numeric", [])

# Load model (calibrated)
model = joblib.load("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/model_artifacts/model.pkl")

# Load OOF preds and labels if available to show historical accuracy
oof_path = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/model_artifacts/oof_preds.npy")
labels_path = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts/labels.npy")
oof_accuracy = None
if oof_path.exists() and labels_path.exists():
    try:
        oof = np.load(oof_path)
        y = np.load(labels_path)
        # load threshold
        thresh_path = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/model_artifacts/decision_threshold.txt")
        if thresh_path.exists():
            best_t = float(thresh_path.read_text().strip())
        else:
            best_t = 0.5
        preds_oof = (oof >= best_t).astype(int)
        oof_accuracy = (preds_oof == y).mean()
    except Exception:
        oof_accuracy = None

# Decision threshold: try to load chosen threshold, fallback 0.5
try:
    thresh_path = Path("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/model_artifacts/decision_threshold.txt")
    if thresh_path.exists():
        DECISION_THRESHOLD = float(thresh_path.read_text().strip())
    else:
        DECISION_THRESHOLD = 0.5
except Exception:
    DECISION_THRESHOLD = 0.5

# Regex & helper funcs (match preprocessing.py)
URL_RE = re.compile(r'(https?://\S+|\bwww\.\S+)', flags=re.IGNORECASE)
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', flags=re.IGNORECASE)
NUM_RE = re.compile(r'\b\d{3,}\b')
HTML_TAG_RE = re.compile(r'<[^>]+>')
MULTI_WS_RE = re.compile(r'\s+')
CRED_PATTERNS = re.compile(r'(password|credentials|verify.*account|enter.*password|login.*now|update.*payment|provide.*bank)', flags=re.IGNORECASE)

def normalize_text(text: str) -> str:
    if text is None:
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

def build_numeric_row(sender_localpart, sender_domain, subject_clean, body_clean):
    # Build a dict matching numeric_names order expected by scaler
    # Provide defaults; sender_domain_freq is not available so set 0.0
    d = {}
    # compute values
    d['body_len'] = len(body_clean)
    d['subject_len'] = len(subject_clean)
    d['num_words'] = count_words(body_clean)
    d['exclamation_count'] = (body_clean.count('!') + subject_clean.count('!'))
    # uppercase_ratio measured on original body but our preproc lowercases; we approximate from original case absent -> 0
    d['uppercase_ratio'] = uppercase_ratio(body_clean)  # likely 0
    d['punctuation_count'] = punctuation_count(body_clean)
    d['num_urls'] = count_token(body_clean, '[url]')
    d['num_redirect_tokens'] = count_token(body_clean, '[redirect_link]')
    d['has_request_for_credentials'] = int(bool(CRED_PATTERNS.search(body_clean)))
    d['has_attached_word'] = int(bool(re.search(r'\b(attached|attachment|see attached)\b', body_clean, flags=re.IGNORECASE)))
    # sender_domain_freq: unavailable; set 0.0
    d['sender_domain_freq'] = 0.0
    # domain categorical encoded numeric
    # map domain -> category ('other' if not in top_domains)
    domain_cat = sender_domain if sender_domain in top_domains else ("missing" if sender_domain=="" else "other")
    try:
        # domain_encoder expects 2D array
        enc_val = domain_encoder.transform([[domain_cat]])[0][0]
        # ensure integer
        d['sender_domain_encoded'] = int(enc_val)
    except Exception:
        # fallback to  -1
        try:
            d['sender_domain_encoded'] = -1
        except Exception:
            d['sender_domain_encoded'] = 0
    # localpart features
    d['localpart_has_digits'] = int(bool(re.search(r'\d', sender_localpart)))
    d['localpart_len'] = len(sender_localpart)
    d['localpart_contains_suspicious'] = int(bool(re.search(r'(admin|support|billing|no-reply|help|service)', sender_localpart, flags=re.IGNORECASE)))
    # Now order them according to numeric_names
    numeric_row = [float(d.get(name, 0.0)) for name in numeric_names]
    return np.array(numeric_row).reshape(1, -1)

def predict_from_input(sender, subject, body):
    # normalize text
    subj_clean = normalize_text(subject)
    body_clean = normalize_text(body)
    local, domain = extract_sender_parts(sender)
    # text -> tfidf
    text = (subj_clean + " " + body_clean).strip()
    X_text = tfidf.transform([text])  # sparse
    # numeric
    numeric_row = build_numeric_row(local, domain, subj_clean, body_clean)
    # scale numeric
    numeric_scaled = scaler.transform(numeric_row)
    X_num_sparse = csr_matrix(numeric_scaled)
    X = hstack([X_text, X_num_sparse], format='csr')
    # predict
    prob = float(model.predict_proba(X)[:,1][0])
    pred_label = int(prob >= DECISION_THRESHOLD)
    return prob, pred_label

# ---------------------------
# Interactive loop
# ---------------------------
print("=== Spam predictor interactive ===")
print("Enter sender email, subject, and body. Type 'quit' at sender prompt to exit.")
if oof_accuracy is not None:
    print(f"Model historical OOF accuracy (on training-time OOF): {oof_accuracy:.4f}")
else:
    print("Model historical OOF accuracy: not available")

while True:
    sender = input("\nSender (or 'quit'): ").strip()
    if sender.lower() in ("quit", "q", "exit"):
        print("Exiting.")
        break
    subject = input("Subject: ").strip()
    body_lines = []
    print("Enter email body. End with a blank line on its own.")
    while True:
        line = input()
        if line == "":
            break
        body_lines.append(line)
    body = "\n".join(body_lines).strip()
    try:
        prob, pred = predict_from_input(sender, subject, body)
        label_str = "SPAM (1)" if pred == 1 else "LEGIT (0)"
        print(f"\nPredicted probability of SPAM: {prob:.4f}")
        print(f"Predicted label: {label_str}  [threshold={DECISION_THRESHOLD}]")
        if oof_accuracy is not None:
            print(f"Historical OOF accuracy: {oof_accuracy:.4f} (for reference)")
        else:
            pass
    except NotFittedError as e:
        print("Model / transformer not fitted correctly:", e)
    except Exception as e:
        print("Error during prediction:", str(e))