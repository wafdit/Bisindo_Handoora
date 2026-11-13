import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.4
)
mp_draw = mp.solutions.drawing_utils

csv_file = 'dataset_abjad_optimal.csv'

# Header: hand1 (63) + hand2 (63) + num_hands (1) + label (1) = total 128 kolom
if not os.path.exists(csv_file):
    header = []
    for h_idx in ['h1', 'h2']:
        for i in range(21):
            header.extend([f'{h_idx}_x_{i}', f'{h_idx}_y_{i}', f'{h_idx}_z_{i}'])
    header.append('num_hands')
    header.append('label')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

cap = cv2.VideoCapture(0)
current_label = 'A'
recording = False
sample_count = 0

print("Tekan 'R' : Mulai / Jeda rekam")
print("Tekan 'S' : Ganti huruf")
print("Tekan 'Q' : Keluar")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand1_coords = [0.0] * 63
    hand2_coords = [0.0] * 63
    num_hands = 0

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        
        # Urutkan tangan berdasarkan posisi X pergelangan di layar (kiri layar duluan)
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)

        for idx, hand_lms in enumerate(sorted_hands):
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            wrist = hand_lms.landmark[0]
            # Normalisasi skala
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

    if recording and num_hands > 0:
        # Gabungkan data: Hand 1 (63) + Hand 2 (63) + Jumlah Tangan (1) + Label
        row_data = hand1_coords + hand2_coords + [num_hands, current_label]
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        sample_count += 1

    status_text = "RECORDING" if recording else "STANDBY"
    color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.putText(frame, f"Label: {current_label} | Sampel: {sample_count} | Hands: {num_hands} | {status_text}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.imshow('Koleksi Dataset BISINDO Optimal', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r') or key == ord('R'):
        recording = not recording
    elif key == ord('s') or key == ord('S'):
        recording = False
        new_label = input("Masukkan huruf target baru: ").strip().upper()
        if new_label:
            current_label = new_label
            sample_count = 0
    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()