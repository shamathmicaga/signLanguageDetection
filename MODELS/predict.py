import cv2
import numpy as np
import tensorflow as tf
from collections import deque

from extract_landmarks import extract_landmarks


MODEL_PATH = "models/sign_model.keras"

SEQUENCE_LENGTH = 30

CONFIDENCE_THRESHOLD = 0.75


model = tf.keras.models.load_model(
    MODEL_PATH
)

SIGNS = np.load(
    "models/sign_names.npy",
    allow_pickle=True
)


sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction_history = deque(
    maxlen=8
)

sentence = []


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(
        frame,
        1
    )


    landmarks = extract_landmarks(
        frame
    )

    sequence.append(
        landmarks
    )


    predicted_sign = ""

    confidence = 0


    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.expand_dims(
            np.array(sequence),
            axis=0
        )


        prediction = model.predict(
            input_data,
            verbose=0
        )[0]


        index = np.argmax(
            prediction
        )


        confidence = float(
            prediction[index]
        )


        sign = SIGNS[index]


        if confidence >= CONFIDENCE_THRESHOLD:

            prediction_history.append(
                sign
            )


            # Require stable prediction

            if (
                len(prediction_history) == prediction_history.maxlen
                and
                len(set(prediction_history)) == 1
            ):

                predicted_sign = sign


                if (
                    len(sentence) == 0
                    or
                    sentence[-1] != sign
                ):

                    sentence.append(
                        sign
                    )


                prediction_history.clear()


    # Limit displayed sentence

    if len(sentence) > 8:
        sentence = sentence[-8:]


    cv2.putText(
        frame,
        f"Sign: {predicted_sign}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    text = " ".join(sentence)


    cv2.putText(
        frame,
        text,
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Sign Language Translator",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


    if key == ord("c"):
        sentence.clear()


cap.release()

cv2.destroyAllWindows()