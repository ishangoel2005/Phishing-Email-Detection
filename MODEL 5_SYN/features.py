# src/features.py
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import joblib
import re

TOP_K_DOMAINS = 200
TFIDF_MAX_FEATURES = 30000

# helper feature functions
def uppercase_ratio(text: str) -> float:
    if not text:
        return 0.0
    up = sum(1 for c in text if c.isupper())
    return up / max(1, len(text))

def count_words(text: str) -> int:
    return len(text.split())

def punctuation_count(text: str) -> int:
    return sum(text.count(c) for c in "?!.,;:")

def count_token(text: str, token: str) -> int:
    if not text:
        return 0
    return text.count(token.lower())

def build_texts(df):
    # subject + body concatenation
    return (df['subject_clean'].fillna('') + ' ' + df['body_clean'].fillna('')).astype(str)

def get_top_k_domains(df, k=TOP_K_DOMAINS):
    top = df['sender_domain'].value_counts().nlargest(k).index.tolist()
    return top

def domain_to_category(domain, top_list):
    if not domain:
        return "missing"
    return domain if domain in top_list else "other"

def make_numeric_features(df, top_domains):
    # numeric features as described
    body = df['body_clean'].fillna('')
    subject = df['subject_clean'].fillna('')

    feat_dict = {
        'body_len': body.str.len().fillna(0).astype(int),
        'subject_len': subject.str.len().fillna(0).astype(int),
        'num_words': body.apply(count_words).astype(int),
        'exclamation_count': (body.str.count('!').fillna(0) + subject.str.count('!').fillna(0)).astype(int),
        'uppercase_ratio': body.apply(uppercase_ratio).astype(float),
        'punctuation_count': body.apply(punctuation_count).astype(int),
        'num_urls': body.apply(lambda t: count_token(t, '[url]')).astype(int),
        'num_redirect_tokens': body.apply(lambda t: count_token(t, '[redirect_link]')).astype(int),
        'has_request_for_credentials': df['has_credential_request'].fillna(0).astype(int),
        'has_attached_word': df['has_attached_word'].fillna(0).astype(int),
    }

    # domain frequency numeric (simple): relative frequency in dataset
    domain_counts = df['sender_domain'].value_counts(normalize=True)
    feat_dict['sender_domain_freq'] = df['sender_domain'].map(domain_counts).fillna(0.0).astype(float)

    # domain categorical mapped to top_k / other
    domain_cat = df['sender_domain'].apply(lambda d: domain_to_category(d, top_domains)).astype(str)
    # use OrdinalEncoder for a compact encoding (order is arbitrary but stable)
    # create encoder that maps unseen categories to -1 (safe, non-colliding)
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    domain_cat_reshaped = domain_cat.values.reshape(-1, 1)
    # fit using the explicit list of known categories so transform is deterministic
    enc.fit([[d] for d in (top_domains + ['other', 'missing'])])
    domain_encoded = enc.transform(domain_cat_reshaped).astype(int).ravel()
    feat_dict['sender_domain_encoded'] = domain_encoded

    # localpart features
    local = df['sender_localpart'].fillna('')
    feat_dict['localpart_has_digits'] = local.str.contains(r'\d', regex=True).astype(int)
    feat_dict['localpart_len'] = local.str.len().fillna(0).astype(int)
    feat_dict['localpart_contains_suspicious'] = local.str.contains(r'(admin|support|billing|no-reply|help|service)', case=False, na=False).astype(int)

    numeric_df = pd.DataFrame(feat_dict)
    return numeric_df, enc

def main(input_path: str, output_dir_override: str = None):
    input_path = Path(input_path)

    # Choose output directory: if output_dir_override provided, use it; otherwise default to "artifacts"
    if output_dir_override:
        out_dir = Path(output_dir_override)
    else:
        out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # 1) TF-IDF on subject+body
    texts = build_texts(df)
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1,2), min_df=3)
    X_text = tfidf.fit_transform(texts)
    joblib.dump(tfidf, out_dir / "tfidf_vectorizer.joblib")
    print(f"Saved TF-IDF vectorizer to {out_dir / 'tfidf_vectorizer.joblib'}")

    # 2) numeric features + domain encoding
    top_domains = get_top_k_domains(df, TOP_K_DOMAINS)
    numeric_df, domain_encoder = make_numeric_features(df, top_domains)
    # scale numeric features
    scaler = StandardScaler()
    numeric_vals = scaler.fit_transform(numeric_df.values.astype(float))
    joblib.dump(scaler, out_dir / "scaler.joblib")
    print(f"Saved numeric scaler to {out_dir / 'scaler.joblib'}")

    # save domain mapping for inference (top domains list)
    domain_meta = {"top_domains": top_domains}
    (out_dir / "domain_mapping.json").write_text(json.dumps(domain_meta))
    joblib.dump(domain_encoder, out_dir / "domain_encoder.joblib")
    print(f"Saved domain mapping and encoder to {out_dir}")

    # 3) combine sparse tfidf + dense numeric into one sparse matrix
    X_numeric_sparse = csr_matrix(numeric_vals)
    X = sparse.hstack([X_text, X_numeric_sparse], format='csr')
    # Save features and labels
    save_npz(out_dir / "features.npz", X)
    np.save(out_dir / "labels.npy", df['label'].values)
    print(f"Saved features.npz and labels.npy to {out_dir}")

    # 4) feature names for bookkeeping
    # construct a feature_names list (tfidf feature names + numeric column names)
    tfidf_feature_names = tfidf.get_feature_names_out().tolist()
    numeric_names = numeric_df.columns.tolist()
    feature_names = {"tfidf": tfidf_feature_names, "numeric": numeric_names}
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names))
    print(f"Saved feature_names.json to {out_dir}")

if __name__ == "__main__":
    # ----- EDIT THESE PATHS DIRECTLY IN THE CODE IF YOU PREFER -----
    # Set INPUT_CSV_PATH to the path of your preprocessed CSV.
    # Set OUTPUT_DIR to the folder where you want all artifacts saved.
    # If INPUT_CSV_PATH is empty (""), the script will fall back to argparse and use CLI args instead.
    INPUT_CSV_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/preprocessed.csv"
    OUTPUT_DIR = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/artifacts"  # e.g., "artifacts" or "/full/path/to/output_folder"
    # ----------------------------------------------------------------

    if INPUT_CSV_PATH:
        main(INPUT_CSV_PATH, OUTPUT_DIR)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", default="artifacts/preprocessed.csv")
        parser.add_argument("--output_dir", default=None)
        args = parser.parse_args()
        main(args.input, args.output_dir)