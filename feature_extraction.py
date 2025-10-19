import cv2
import mediapipe as mp
import numpy as np

def extract_hand_angles_from_frame(frame):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    all_angles = []

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        joints = np.array([[p.x, p.y, p.z] for p in lm.landmark])

        # Example: angle between wrist, index, middle finger
        v1 = joints[5] - joints[0]
        v2 = joints[9] - joints[0]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
        all_angles.append(angle)

    return np.array(all_angles, dtype=np.float32)
