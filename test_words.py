import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

model = load_model('model_kata.h5')
actions = np.load('label_kata.npy')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.4
)
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(results):
    hand1_coords = [0.0] * 63
    hand2_coords = [0.0] * 63
    num_hands = 0

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)

        for idx, hand_lms in enumerate(sorted_hands):
            wrist = hand_lms.landmark[0]
            scale_dist = np.linalg.norm(np.array([
                hand_lms.landmark[9].x - wrist.x,
                hand_lms.landmark[9].y - wrist.y
            ]))
            if scale_dist == 0:
                scale_dist = 1.0

            coords = []
            for lm in hand_lms.landmark:
                coords.extend([
                    (lm.x - wrist.x) / scale_dist,
                    (lm.y - wrist.y) / scale_dist,
                    (lm.z - wrist.z) / scale_dist
                ])

            if idx == 0:
                hand1_coords = coords
            elif idx == 1:
                hand2_coords = coords

    return np.array(hand1_coords + hand2_coords + [num_hands])

sequence = deque(maxlen=30)
detected_word = "Mendeteksi..."
confidence = 0.0
threshold = 0.80

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        max_idx = np.argmax(res)
        confidence = res[max_idx]

        if confidence > threshold:
            detected_word = actions[max_idx].replace('_', ' ').upper()
        else:
            detected_word = "Mendeteksi..."

    cv2.rectangle(frame, (10, 10), (450, 85), (0, 0, 0), -1)
    cv2.putText(frame, f"Kata: {detected_word}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Deteksi Kata BISINDO - Level 2', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()