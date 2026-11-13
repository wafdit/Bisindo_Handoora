import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import warnings

# Abaikan warning feature names dari scikit-learn agar terminal bersih
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load model yang sudah dilatih
with open('model_abjad.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.25
)
mp_draw = mp.solutions.drawing_utils

# Buffer untuk menyimpan riwayat jejak ujung jari (25 frame terakhir)
BUFFER_LEN = 25
pinky_points = deque(maxlen=BUFFER_LEN)   # Untuk huruf J (ujung kelingking)
index_points = deque(maxlen=BUFFER_LEN)   # Untuk huruf Z (ujung telunjuk)

def detect_j_gesture(pts):
    """Mendeteksi apakah lintasan membentuk huruf J (turun lalu melengkung)"""
    if len(pts) < 15:
        return False
    y_coords = [p[1] for p in pts]
    x_coords = [p[0] for p in pts]
    
    total_y_movement = max(y_coords) - min(y_coords)
    total_x_movement = max(x_coords) - min(x_coords)
    
    if total_y_movement > 0.12 and total_x_movement > 0.08:
        lowest_point_idx = np.argmax(y_coords)
        if lowest_point_idx < len(pts) - 3:
            return True
    return False

def detect_z_gesture(pts):
    """Mendeteksi apakah lintasan bergerak zigzag membentuk huruf Z"""
    if len(pts) < 20:
        return False
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]
    
    total_movement = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
    return total_movement > 0.25

cap = cv2.VideoCapture(0)

detected_letter = "-"
confidence_score = 0.0

print("=== REAL-TIME DETECTION (HYBRID I/J & Z OPTIMAL) ===")
print("Tekan 'Q' untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # ================= KODE BARU DI SINI =================
    hand1_coords = [0.0] * 63
    hand2_coords = [0.0] * 63
    num_hands = 0
    current_pinky_px = None
    current_index_px = None

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        # Urutkan berdasarkan koordinat X di layar (kiri ke kanan)
        sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)

        for idx, hand_lms in enumerate(sorted_hands):
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            wrist = hand_lms.landmark[0]
            # Normalisasi skala jarak tangan
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

        # Tangan dominan (tangan paling kanan di layar) untuk gesture J dan Z
        dominant_hand = sorted_hands[-1]
        current_pinky_px = (int(dominant_hand.landmark[20].x * w), int(dominant_hand.landmark[20].y * h))
        current_index_px = (int(dominant_hand.landmark[8].x * w), int(dominant_hand.landmark[8].y * h))

        # Bentuk 127 fitur: hand1 (63) + hand2 (63) + num_hands (1)
        features = np.array(hand1_coords + hand2_coords + [num_hands]).reshape(1, -1)
        
        # Prediksi probabilitas model
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            confidence_score = np.max(probs)
            raw_label = model.classes_[np.argmax(probs)]
        else:
            confidence_score = 1.0
            raw_label = model.predict(features)[0]

        # Filter threshold keyakinan (confidence < 70% dianggap sedang transisi)
        if confidence_score < 0.70:
            detected_letter = "Mendeteksi..."
            pinky_points.clear()
            index_points.clear()
        else:
            pred_label = raw_label

            # Logika Motion Buffer (J dan Z)
            if pred_label == 'I':
                if current_pinky_px:
                    pinky_points.append(dominant_hand.landmark[20])
                    if detect_j_gesture([(p.x, p.y) for p in pinky_points]):
                        detected_letter = "J"
                    else:
                        detected_letter = "I"
                index_points.clear()

            elif pred_label in ['D', '1', 'Z']:
                if current_index_px:
                    index_points.append(dominant_hand.landmark[8])
                    if detect_z_gesture([(p.x, p.y) for p in index_points]):
                        detected_letter = "Z"
                    else:
                        detected_letter = pred_label
                pinky_points.clear()

            else:
                detected_letter = pred_label
                pinky_points.clear()
                index_points.clear()

    else:
        pinky_points.clear()
        index_points.clear()
        detected_letter = "-"
        confidence_score = 0.0
    # ================= AKHIR KODE BARU ===================

    # Gambar Jejak Gerakan (Trajectory Trail) di Layar untuk J
    for i in range(1, len(pinky_points)):
        pt1 = (int(pinky_points[i - 1].x * w), int(pinky_points[i - 1].y * h))
        pt2 = (int(pinky_points[i].x * w), int(pinky_points[i].y * h))
        cv2.line(frame, pt1, pt2, (0, 165, 255), 3)

    # Tampilkan hasil deteksi di frame kamera
    cv2.rectangle(frame, (10, 10), (340, 85), (0, 0, 0), -1)
    cv2.putText(frame, f"Huruf: {detected_letter}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Conf: {confidence_score*100:.1f}%", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Deteksi Realtime BISINDO (Hybrid J & Z)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()