# ======================= IMPORTS =======================
import pickle
from tensorflow.keras.models import load_model

# ======================= CONFIG =======================
MODEL_PATH = "C:\\Users\\Dell\\Downloads\\Research Paper_Literature Review\\ml_output\\final_email_model.pkl"       # your saved model
VECTORIZER_PATH = "C:\\Users\\Dell\\Downloads\\Research Paper_Literature Review\\ml_output\\tfidf_vectorizer.pkl"   # your saved vectorizer

# ======================= LOAD MODEL & VECTORIZER =======================
print("Loading model and vectorizer...")
model = load_model(MODEL_PATH)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Model and vectorizer loaded successfully!\n")

# ======================= PREDICTION FUNCTION =======================
def predict_email(subject: str, body: str):
    combined_text = f"{subject} {body}"  # combine subject + body

    # Transform using saved vectorizer (DO NOT fit again!)
    X_input = vectorizer.transform([combined_text])

    # Predict probability
    prob = model.predict(X_input, verbose=0)[0][0]

    # Convert probability to label
    label = "SPAM" if prob >= 0.5 else "LEGITIMATE"

    print("\nPrediction Results:")
    print(f"Probability of SPAM: {prob:.4f}")
    print(f"Predicted Label: {label}")

# ======================= USER INPUT =======================
print("Enter email details for spam detection:\n")
subject = input("Email Subject: ")
body = input("Email Body: ")

predict_email(subject, body)