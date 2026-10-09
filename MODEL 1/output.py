import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

# ===============================
# CONSTANTS
# ===============================
MAX_LEN = 200
MODEL_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 1/spam_cnn_glove.h5"       # Change to /kaggle/working/... if on Kaggle
TOKENIZER_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 1/tokenizer.pkl"

# ===============================
# TEXT CLEANING FUNCTION
# ===============================
def clean_text(text):
    """
    Same preprocessing as training:
    - Lowercase
    - Remove punctuation
    - Strip extra spaces
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

# ===============================
# MAIN INFERENCE FUNCTION
# ===============================
def predict_email():
    # Load model & tokenizer
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Get input from user
    sender = input("Enter sender email: ")
    receiver = input("Enter receiver email: ")
    subject = input("Enter email subject: ")
    body = input("Enter email body: ")

    # Combine all text attributes
    email_text = f"{sender} {receiver} {subject} {body}"
    email_text = clean_text(email_text)

    # Tokenize + pad
    seq = tokenizer.texts_to_sequences([email_text])
    padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # Predict
    prob = model.predict(padded_seq)[0][0]   # probability of spam
    confidence = prob * 100

    # Decide label
    if prob >= 0.7:
        label = "Spam / Phishy / Fraud"
    else:
        label = "Not Spam / Safe"

    print("\n===============================")
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print("===============================\n")

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    predict_email()