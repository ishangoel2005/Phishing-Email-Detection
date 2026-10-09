import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- CONFIG ----------------
INPUT_CSV = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 3/email_vectors_tfidf.csv"          # Your CSV with TF-IDF vectors
VECTORIZER_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 3/tfidf_vectorizer.pkl" # Saved TF-IDF vectorizer
MODEL_SAVE_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 3/mlp_tfidf_model.h5"
MAX_FEATURES = 5000                # Reduce features for memory
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 15
BATCH_SIZE = 32

# ---------------- LOAD DATA ----------------
print("Loading dataset...")
df = pd.read_csv(INPUT_CSV)

# Make sure label column exists
if 'label' not in df.columns:
    raise ValueError("CSV must have a 'label' column")

y = df['label'].values
X = df.drop(columns=['label'])

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Dataset shape: {X.shape}, Labels: {y.shape}")

# ---------------- BUILD MLP ----------------
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------- TRAIN WITH PROGRESS BAR ----------------
es = EarlyStopping(patience=3, restore_best_weights=True)

# Keras shows progress bar by default
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=1
)

# ---------------- SAVE MODEL ----------------
model.save(MODEL_SAVE_PATH)
print(f"✅ MLP model saved as {MODEL_SAVE_PATH}")

# ---------------- EVALUATE ----------------
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------- PLOT GRAPHS ----------------
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()