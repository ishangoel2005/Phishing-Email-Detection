import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D,
    Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = "C:/Users/Dell/Downloads/Research Paper_Literature Review/archive/CLEANED/combined.csv"          # <-- change if needed
MAX_WORDS = 50000
MAX_LEN = 250
EMB_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10

# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(CSV_PATH)

df.fillna("", inplace=True)

# Combine relevant fields
df["text"] = (
    df["sender"] + " " +
    df["receiver"] + " " +
    df["subject"] + " " +
    df["body"]
)

df["text"] = df["text"].apply(clean_text)
labels = df["label"].values

# -------------------------------
# TOKENIZATION
# -------------------------------
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])

sequences = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = labels

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# LOAD GLOVE
# -------------------------------
def load_glove(tokenizer):
    embeddings_index = {}
    with open("glove.6B.100d.txt", encoding="utf8") as f:
        for line in f:
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[1:], dtype="float32")

    embedding_matrix = np.zeros((MAX_WORDS, EMB_DIM))
    for word, i in tokenizer.word_index.items():
        if i < MAX_WORDS and word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]

    return embedding_matrix

embedding_matrix = load_glove(tokenizer)

# -------------------------------
# MODEL: CNN + BiLSTM
# -------------------------------
input_layer = Input(shape=(MAX_LEN,))

embedding = Embedding(
    MAX_WORDS,
    EMB_DIM,
    weights=[embedding_matrix],
    trainable=False
)(input_layer)

conv = Conv1D(128, kernel_size=5, activation="relu")(embedding)
conv = MaxPooling1D(pool_size=2)(conv)

bilstm = Bidirectional(LSTM(64, return_sequences=False))(conv)

dense = Dense(128, activation="relu")(bilstm)
drop = Dropout(0.5)(dense)

output = Dense(1, activation="sigmoid")(drop)

model = Model(input_layer, output)

model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# TRAIN
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=1
)

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("spam_cnn_bilstm_glove.h5")
print("✅ Model saved as spam_cnn_bilstm_glove.h5")

# -------------------------------
# EVALUATION
# -------------------------------
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# PLOTS
# -------------------------------
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.show()