import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from tensorflow.keras.utils import plot_model

MAX_WORDS = 50000
MAX_LEN = 200
EMB_DIM = 100

def load_glove(tokenizer):
    embeddings_index = {}
    with open("C:/Users/Dell/Downloads/Research Paper_Literature Review/glove.6B.100d.txt", encoding="utf8") as f:
        for line in f:
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[1:], dtype="float32")

    embedding_matrix = np.zeros((MAX_WORDS, EMB_DIM))
    for word, i in tokenizer.word_index.items():
        if i < MAX_WORDS and word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]

    return embedding_matrix

def main():
    data = np.load("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 1/processed_data.npz")
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    with open("C:/Users/Dell/Downloads/Research Paper_Literature Review/MODEL 1/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    embedding_matrix = load_glove(tokenizer)

    input_layer = Input(shape=(MAX_LEN,))
    embed = Embedding(
        MAX_WORDS,
        EMB_DIM,
        weights=[embedding_matrix],
        trainable=False
    )(input_layer)

    convs = []
    for k in [3, 4, 5]:
        c = Conv1D(128, k, activation="relu")(embed)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)

    x = Concatenate()(convs)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(input_layer, output)
    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=32,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=1  # progress bar
    )

    model.save("spam_cnn_glove.h5")
    print("✅ Model saved as spam_cnn_glove.h5")

    # Evaluation
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Graphs
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.show()

if __name__ == "__main__":
    main()