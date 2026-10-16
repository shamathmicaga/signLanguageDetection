import cv2
import os
import numpy as np
from extract_landmarks import extract_landmarks


SIGNS = [
    "HELLO",
    "THANK_YOU",
    "PLEASE",
    "SORRY",
    "YES",
    "NO",
    "HELP",
    "WATER",
    "FOOD",
    "GOOD",
    "BAD",
    "LOVE",
    "YOU",
    "I",
    "ME"
]

SEQUENCE_LENGTH = 30
SEQUENCES_PER_SIGN = 100

DATA_PATH = "dataset"


cap = cv2.VideoCapture(0)


for sign in SIGNS:

    sign_path = os.path.join(DATA_PATH, sign)

    os.makedirs(sign_path, exist_ok=True)

    print("\nPreparing:", sign)

    for sequence in range(SEQUENCES_PER_SIGN):

        frames = []

        while len(frames) < SEQUENCE_LENGTH:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"{sign} | Sample {sequence + 1}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Frame {len(frames) + 1}/{SEQUENCE_LENGTH}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Collect Sign Language", frame)

            landmarks = extract_landmarks(frame)

            frames.append(landmarks)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        frames = np.array(frames)

        np.save(
            os.path.join(
                sign_path,
                f"{sequence}.npy"
            ),
            frames
        )

        print(
            sign,
            sequence + 1,
            "/",
            SEQUENCES_PER_SIGN
        )


cap.release()

cv2.destroyAllWindows()