# ======================= IMPORTS =======================
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# ======================= CONFIG =======================
MODEL_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 3/mlp_tfidf_full_output/mlp_tfidf_model.h5"       # path to your saved model
VECTORIZER_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 3/mlp_tfidf_full_output/tfidf_vectorizer.pkl"  # path to your saved vectorizer

# ======================= LOAD MODEL AND VECTORIZER =======================
print("Loading model and vectorizer...")
model = load_model(MODEL_PATH)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Model and vectorizer loaded successfully!\n")

# ======================= USER INPUT =======================
print("Enter email details for spam detection:\n")

subject = input("Email Subject: ").strip()
body = input("Email Body: ").strip()

if not subject and not body:
    print("⚠️ Both subject and body are empty. Cannot predict.")
    exit()

# ======================= PREPROCESS INPUT =======================
combined_text = subject + " " + body
X_input = vectorizer.transform([combined_text])  # ⚠️ DO NOT FIT AGAIN

# ======================= PREDICTION =======================
prob = model.predict(X_input, verbose=0)[0][0]
label = "SPAM" if prob >= 0.5 else "LEGITIMATE"

# ======================= OUTPUT =======================
print("\n--- Prediction Result ---")
print(f"Spam Probability: {prob:.4f}")
print(f"Predicted Label: {label}")