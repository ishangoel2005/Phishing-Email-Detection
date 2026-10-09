# src/preprocessing.py
import re
import html
import argparse
import pandas as pd
from pathlib import Path

# Regex patterns
URL_RE = re.compile(r'(https?://\S+|\bwww\.\S+)', flags=re.IGNORECASE)
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', flags=re.IGNORECASE)
NUM_RE = re.compile(r'\b\d{3,}\b')  # replace long numbers (invoice, phone, large ids)
HTML_TAG_RE = re.compile(r'<[^>]+>')
MULTI_WS_RE = re.compile(r'\s+')

# Credential-ish patterns (used to create a flag)
CRED_PATTERNS = re.compile(r'(password|credentials|verify.*account|enter.*password|login.*now|update.*payment|provide.*bank)', flags=re.IGNORECASE)

def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    # unescape HTML entities
    text = html.unescape(str(text))
    # remove tags
    text = HTML_TAG_RE.sub(' ', text)
    # convert URLs/emails/numbers to tokens
    text = URL_RE.sub(' [URL] ', text)
    text = EMAIL_RE.sub(' [EMAIL] ', text)
    text = NUM_RE.sub(' [NUM] ', text)
    # normalize whitespace and strip
    text = MULTI_WS_RE.sub(' ', text).strip()
    # keep case for certain heuristics? we'll lowercase here
    return text.lower()

def extract_sender_parts(sender: str):
    sender = str(sender or "").strip()
    if "@" in sender:
        local, domain = sender.split("@", 1)
        return local.lower(), domain.lower()
    else:
        return sender.lower(), ""

def add_basic_flags(df: pd.DataFrame) -> pd.DataFrame:
    # flags derived from raw/clean text
    df['num_redirect_tokens'] = df['body'].str.count(r'\[redirect_link\]', flags=re.IGNORECASE).fillna(0).astype(int)
    df['contains_redirect'] = (df['num_redirect_tokens'] > 0).astype(int)
    df['has_credential_request'] = df['body'].str.contains(CRED_PATTERNS, regex=True, na=False).astype(int)
    df['has_attached_word'] = df['body'].str.contains(r'\b(attached|attachment|see attached)\b', case=False, na=False).astype(int)
    return df

def main(in_path: str, out_path: str):
    in_path = Path(in_path)
    out_path = Path(out_path)
    df = pd.read_csv(in_path)

    # basic sanity
    print(f"Loaded {len(df)} rows from {in_path}")

    # clean subject and body
    df['subject_clean'] = df['subject'].fillna("").apply(normalize_text)
    df['body_clean'] = df['body'].fillna("").apply(normalize_text)

    # sender parts
    df[['sender_localpart', 'sender_domain']] = df['sender'].apply(lambda s: pd.Series(extract_sender_parts(s)))

    # create flags based on raw 'body' (before or after cleaning both ok; using cleaned)
    df['body'] = df['body_clean']  # ensure later code uses cleaned body
    df = add_basic_flags(df)

    # Save preprocessed CSV
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved preprocessed CSV to {out_path}")
    # small QA
    print("Top sender domains:", df['sender_domain'].value_counts().head(10).to_dict())
    print("Label distribution:", df['label'].value_counts().to_dict())

if __name__ == "__main__":
    # --- EDIT THESE PATHS DIRECTLY IN THE CODE IF YOU PREFER ---
    # Set INPUT_CSV_PATH to the path of your raw CSV and OUTPUT_CSV_PATH to where you want the preprocessed CSV saved.
    # If INPUT_CSV_PATH is empty (""), the script will fall back to argparse and use CLI args instead.
    INPUT_CSV_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/custom_combined_dataset.csv"
    OUTPUT_CSV_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 5_SYN/preprocessed.csv"
    # ----------------------------------------------------------------

    if INPUT_CSV_PATH:
        # Use the hard-coded paths you set above
        main(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    else:
        # Fallback to argparse (unchanged behavior)
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", default="data/phishing_dataset_40000.csv")
        parser.add_argument("--output", default="artifacts/preprocessed.csv")
        args = parser.parse_args()
        main(args.input, args.output)