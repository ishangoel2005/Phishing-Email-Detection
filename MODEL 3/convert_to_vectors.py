import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- CONFIG ----------------
INPUT_CSV = "C:/Users/Dell/Downloads/Research Paper_Literature Review/archive/CLEANED/combined.csv"
OUTPUT_CSV = "C:/Users/Dell/Downloads/Research Paper_Literature Review/archive/CLEANED/email_vectors_tfidf.csv"
VECTORIZER_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/archive/CLEANED/tfidf_vectorizer.pkl"

MAX_FEATURES = 10000
NGRAM_RANGE = (1, 2)

# ---------------- CLEANING ----------------
def clean_text(text):
    text = str(text).lower()

    # normalize patterns
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\d+", " number ", text)

    # remove special characters
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ---------------- LOAD DATA ----------------
print("📥 Loading dataset...")
df = pd.read_csv(
    INPUT_CSV,
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

# fill missing values
df.fillna("", inplace=True)

# ---------------- COMBINE IMPORTANT FIELDS ----------------
print("🔗 Combining text fields...")

df["combined_text"] = (
    df["sender"] + " " +
    df["subject"] + " " +
    df["body"] + " " +
    df["urls"].astype(str)
)

# clean text
df["combined_text"] = df["combined_text"].apply(clean_text)

# ---------------- TF-IDF VECTORIZATION ----------------
print("🧠 Vectorizing text using TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    stop_words="english",
    min_df=3
)

X = tfidf.fit_transform(df["combined_text"])

# ---------------- SAVE VECTORS ----------------
print("💾 Saving vectorized CSV...")

X_df = pd.DataFrame(
    X.toarray(),
    columns=tfidf.get_feature_names_out()
)

X_df["label"] = df["label"].values

X_df.to_csv(OUTPUT_CSV, index=False)

# ---------------- SAVE VECTORIZER ----------------
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(tfidf, f)

print("✅ DONE!")
print(f"📁 Vectors saved as: {OUTPUT_CSV}")
print(f"📦 Vectorizer saved as: {VECTORIZER_PATH}")