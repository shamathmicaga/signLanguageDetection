import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


DATA_PATH = "dataset"
MODEL_PATH = "models/sign_model.keras"

SEQUENCE_LENGTH = 30


# -----------------------------------
# Read sign names automatically
# -----------------------------------

SIGNS = sorted([
    folder
    for folder in os.listdir(DATA_PATH)
    if os.path.isdir(
        os.path.join(DATA_PATH, folder)
    )
])


print("Signs:")

for index, sign in enumerate(SIGNS):
    print(index, sign)


label_map = {
    sign: index
    for index, sign in enumerate(SIGNS)
}


# -----------------------------------
# Load dataset
# -----------------------------------

sequences = []
labels = []


for sign in SIGNS:

    folder = os.path.join(DATA_PATH, sign)

    for filename in os.listdir(folder):

        if not filename.endswith(".npy"):
            continue

        path = os.path.join(
            folder,
            filename
        )

        data = np.load(path)

        if data.shape[0] != SEQUENCE_LENGTH:
            continue

        sequences.append(data)

        labels.append(
            label_map[sign]
        )


X = np.array(sequences)

y = to_categorical(
    labels,
    num_classes=len(SIGNS)
)


print("X:", X.shape)

print("Y:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=np.argmax(y, axis=1)
)


# -----------------------------------
# Model
# -----------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            X.shape[2]
        )
    ),

    tf.keras.layers.LSTM(
        128,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.LSTM(
        128,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.LSTM(
        64
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        len(SIGNS),
        activation="softmax"
    )
])


model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


model.summary()


model.fit(

    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=100,

    batch_size=32
)


os.makedirs(
    "models",
    exist_ok=True
)


model.save(MODEL_PATH)


np.save(
    "models/sign_names.npy",
    np.array(SIGNS)
)


print("Saved:", MODEL_PATH)
