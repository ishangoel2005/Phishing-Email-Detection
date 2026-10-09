# ======================= IMPORTS =======================
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize
import zipfile

# ======================= CONFIG =======================
INPUT_CSV = "C:/Users/Dell/Downloads/Research Paper_Literature Review/archive/CLEANED/combined.csv"  # Path to your CSV
OUTPUT_DIR = "ml_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "final_email_model.pkl"
VECTORIZER_NAME = "tfidf_vectorizer.pkl"
ZIP_NAME = "ml_output.zip"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ======================= NLTK DOWNLOADS =======================
nltk.download('punkt')

# ======================= LOAD DATA =======================
print("Loading dataset...")
df = pd.read_csv(INPUT_CSV, encoding='latin1')

required_cols = {"subject", "body", "label"}
if not required_cols.issubset(set(df.columns)):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# Combine subject + body
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
X = df['text']
y = df['label']

# ======================= SPLIT DATA =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ======================= TEXT PREPROCESSING =======================
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join(tokens)

X_train = X_train.apply(preprocess)
X_test = X_test.apply(preprocess)

# ======================= TF-IDF VECTORIZE =======================
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english', sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ======================= MODEL TRAINING =======================
# You can switch between MultinomialNB, LogisticRegression, LinearSVC
model = MultinomialNB()
# model = LogisticRegression(max_iter=1000)
# model = LinearSVC()

print("Training model...")
model.fit(X_train_vec, y_train)

# ======================= EVALUATION =======================
y_pred = model.predict(X_test_vec)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Spam'], yticklabels=['Legit', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# ======================= SAVE MODEL & VECTORIZER =======================
with open(os.path.join(OUTPUT_DIR, MODEL_NAME), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(OUTPUT_DIR, VECTORIZER_NAME), "wb") as f:
    pickle.dump(vectorizer, f)

print(f"Model and vectorizer saved to {OUTPUT_DIR}")

# ======================= ZIP OUTPUT =======================
with zipfile.ZipFile(ZIP_NAME, 'w') as zipf:
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            zipf.write(os.path.join(root, file), arcname=file)

print(f"All outputs zipped into {ZIP_NAME}")