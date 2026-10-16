import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def extract_landmarks(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    features = []

    # -------------------------
    # LEFT + RIGHT HAND
    # -------------------------

    hand_data = []

    if hand_results.multi_hand_landmarks:

        for hand_landmarks in hand_results.multi_hand_landmarks[:2]:

            coords = []

            for lm in hand_landmarks.landmark:

                coords.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            hand_data.append(coords)

    # Each hand = 21 landmarks × 3 = 63

    while len(hand_data) < 2:
        hand_data.append([0.0] * 63)

    features.extend(hand_data[0])
    features.extend(hand_data[1])

    # -------------------------
    # BODY / POSE
    # -------------------------

    if pose_results.pose_landmarks:

        for lm in pose_results.pose_landmarks.landmark:

            features.extend([
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ])

    else:

        features.extend([0.0] * (33 * 4))

    return np.array(features, dtype=np.float32)