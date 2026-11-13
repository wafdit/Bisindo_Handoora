import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.4
)
mp_draw = mp.solutions.drawing_utils

# Konfigurasi folder & daftar kata keseharian
DATA_PATH = 'dataset_kata'
actions = np.array(['halo', 'terima_kasih', 'tolong', 'maaf', 'sama_sama'])
no_sequences = 30      # Jumlah rekaman per kata (30 kali peragaan)
sequence_length = 30   # Durasi tiap gerakan = 30 frame (~1 detik)

for action in actions:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

def extract_keypoints(results):
    """Ekstraksi 127 fitur (konsisten dengan normalisasi optimal Level 1)"""
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

cap = cv2.VideoCapture(0)

print("=== PEREKAMAN DATASET KATA (LEVEL 2) ===")
print("Tekan 'SPASI' untuk mulai merekam tiap kata. Tekan 'Q' untuk keluar.")

for action in actions:
    # Cek apakah sudah ada file sebelumnya agar bisa melanjutkan tanpa menimpa
    existing_files = [f for f in os.listdir(os.path.join(DATA_PATH, action)) if f.endswith('.npy')]
    start_seq = len(existing_files)

    if start_seq >= no_sequences:
        print(f"Kata '{action}' sudah lengkap ({start_seq} sampel). Melewati...")
        continue

    # Layar tunggu sebelum mulai merekam kata tertentu
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (10, 10), (620, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Target Kata: {action.upper()}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Tekan SPASI untuk mulai rekam...", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Koleksi Dataset Kata - Level 2', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 32:  # SPASI
            break
        elif key == ord('q') or key == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Mulai rekam urutan gerakan (sequence)
    for sequence in range(start_seq, no_sequences):
        window_data = []
        for frame_num in range(sequence_length):
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
            window_data.append(keypoints)

            # Indikator visual perekaman
            if frame_num == 0:
                cv2.putText(frame, "MULAI GERAKAN!", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.rectangle(frame, (10, 10), (500, 75), (0, 0, 0), -1)
            cv2.putText(frame, f"Kata: {action.upper()} | Sampel: {sequence+1}/{no_sequences}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_num+1}/{sequence_length}",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow('Koleksi Dataset Kata - Level 2', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Simpan 1 sekuens (30x127) ke file .npy
        npy_path = os.path.join(DATA_PATH, action, f"{sequence}.npy")
        np.save(npy_path, np.array(window_data))

        # Jeda persiapan 1 detik sebelum sampel berikutnya otomatis dimulai
        for pause in range(25):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Tersimpan ({sequence+1}/{no_sequences}) - Siap ulangi...",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.imshow('Koleksi Dataset Kata - Level 2', frame)
            cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()
print("Semua dataset kata selesai direkam!")