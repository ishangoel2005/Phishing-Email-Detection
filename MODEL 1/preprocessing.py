import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 50000
MAX_LEN = 200

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " <URL> ", text)
    text = re.sub(r"\S+@\S+", " <EMAIL> ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    csv_path = input("Enter path of cleaned CSV file: ")

    df = pd.read_csv(csv_path)

    # Combine important fields
    df["text"] = (
        df["sender"].fillna("") + " " +
        df["receiver"].fillna("") + " " +
        df["subject"].fillna("") + " " +
        df["body"].fillna("")
    )

    df["text"] = df["text"].apply(clean_text)
    labels = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    # Save everything
    np.savez("processed_data.npz",
             X_train=X_train_pad,
             X_test=X_test_pad,
             y_train=y_train,
             y_test=y_test)

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("✅ Preprocessing complete. Files saved:")
    print(" - processed_data.npz")
    print(" - tokenizer.pkl")

if __name__ == "__main__":
    main()
