import csv
import copy
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import os
import pickle

# --- KONFIGURASI ---
SEQUENCE_LENGTH = 30   # 30 frame (~1 detik per kata)
NUM_FEATURES = 127     # 63 (tangan 1) + 63 (tangan 2) + 1 (jumlah tangan)
DATASET_CSV = 'dataset_kata_level2.csv'
LABEL_CSV = 'kata_label.csv'
MODEL_PATH = 'model_kata_level2.pkl'

# Load daftar label kata
with open(LABEL_CSV, encoding='utf-8-sig') as f:
    labels = [row[0] for row in csv.reader(f) if row]

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.4
)
mp_draw = mp.solutions.drawing_utils

# Load model Scikit-Learn (.pkl)
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"[OK] Model {MODEL_PATH} berhasil dimuat!")
else:
    print(f"[WARNING] File {MODEL_PATH} tidak ditemukan! Jalankan train_level2.py dulu.")

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for lm in landmarks.landmark:
        lm_x = min(int(lm.x * image_width), image_width - 1)
        lm_y = min(int(lm.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [[lm_x, lm_y]], axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def extract_features(results):
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

    return hand1_coords + hand2_coords + [num_hands]

def save_sequence_to_csv(label_id, seq_buffer):
    flattened = np.array(seq_buffer).flatten().tolist()
    with open(DATASET_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([label_id] + flattened)

# --- VARIABEL STATE ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

mode = 0  
recording_active = False
target_label_id = -1

record_buffer = []
inference_buffer = deque(maxlen=SEQUENCE_LENGTH)
# Tambahan: Buffer khusus untuk melacak pergerakan asli ujung telunjuk (tanpa normalisasi)
motion_history = deque(maxlen=SEQUENCE_LENGTH) 

sample_counts = {i: 0 for i in range(len(labels))}
if os.path.exists(DATASET_CSV):
    with open(DATASET_CSV, 'r') as f:
        for row in csv.reader(f):
            if row:
                cid = int(row[0])
                if cid in sample_counts:
                    sample_counts[cid] += 1

prev_tick = cv2.getTickCount()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_tick = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
    prev_tick = curr_tick

    frame = cv2.flip(frame, 1)
    debug_image = copy.deepcopy(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    features = extract_features(results)

    # Lacak ujung telunjuk untuk motion filter
    if results.multi_hand_landmarks:
        # Ambil tangan terakhir (dominan) untuk dilacak gerakannya
        dom_hand = results.multi_hand_landmarks[-1]
        # Simpan koordinat X dan Y asli di layar
        motion_history.append((dom_hand.landmark[8].x, dom_hand.landmark[8].y))
        
        for hand_lms in results.multi_hand_landmarks:
            brect = calc_bounding_rect(debug_image, hand_lms)
            cv2.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 2)
            mp_draw.draw_landmarks(debug_image, hand_lms, mp_hands.HAND_CONNECTIONS)
    else:
        motion_history.clear()

    # --- MODE REKAM (MODE 1) ---
    if recording_active:
        record_buffer.append(features)
        progress = len(record_buffer)
        cv2.putText(debug_image, f"MEREKAM [{labels[target_label_id]}]: {progress}/{SEQUENCE_LENGTH}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if progress == SEQUENCE_LENGTH:
            save_sequence_to_csv(target_label_id, record_buffer)
            sample_counts[target_label_id] += 1
            print(f"Tersimpan! Kata '{labels[target_label_id]}' total: {sample_counts[target_label_id]} sampel")
            recording_active = False
            record_buffer = []

    # --- MODE INFERENCE REAL-TIME (MODE 0) ---
    pred_text = "-"
    conf_val = 0.0
    if mode == 0 and model is not None:
        inference_buffer.append(features)
        
        if len(inference_buffer) == SEQUENCE_LENGTH and len(motion_history) == SEQUENCE_LENGTH:
            # Cek pergerakan hanya pada setengah jendela terakhir (15 frame) agar responsif
            pts = np.array(motion_history)[-15:] 
            gerak_x = np.max(pts[:, 0]) - np.min(pts[:, 0])
            gerak_y = np.max(pts[:, 1]) - np.min(pts[:, 1])
            total_gerak = gerak_x + gerak_y 
            
            # Jika rentang gerak ujung jari sangat kecil (< 8% ukuran layar), anggap diam
            if total_gerak < 0.08:
                pred_text = "Diam / Bersiap..."
                conf_val = 0.0
            else:
                input_data = np.array(inference_buffer).flatten().reshape(1, -1)
                probs = model.predict_proba(input_data)[0]
                max_idx = np.argmax(probs)
                conf_val = probs[max_idx]
                
                if conf_val > 0.70:
                    pred_text = labels[int(model.classes_[max_idx])]
                else:
                    pred_text = "Mendeteksi..."

    # --- TAMPILAN UI ---
    cv2.putText(debug_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    if mode == 0:
        status_model = "" if model is not None else " [MODEL KOSONG]"
        cv2.putText(debug_image, f"Kata: {pred_text} ({conf_val*100:.0f}%){status_model}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    else:
        cv2.putText(debug_image, "MODE: REKAM (Tekan 0-9 untuk mulai 1 gerakan)",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        for idx, name in enumerate(labels):
            cv2.putText(debug_image, f"[{idx}] {name}: {sample_counts[idx]}",
                        (10, 145 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('BISINDO Level 2 - Kata Dinamis', debug_image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('k') or key == ord('K'):
        mode = 1
        recording_active = False
    elif key == ord('n') or key == ord('N'):
        mode = 0
        recording_active = False
        inference_buffer.clear()
        motion_history.clear()
    elif mode == 1 and not recording_active and (48 <= key <= 57):
        num = key - 48
        if num < len(labels):
            target_label_id = num
            record_buffer = []
            motion_history.clear()
            recording_active = True

cap.release()
cv2.destroyAllWindows()