import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    # Sıfıra bölünmeyi engellemek için küçük bir epsilon ekleyelim
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# 1. Modelleri Yükle
model = joblib.load("gesture_mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
le = joblib.load("label_encoder.pkl")  # Harf dönüşümü için bunu mutlaka yükle

# 2. MediaPipe Tasks API Setup
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(  # Hata alırsan HandLandmarkerOptions olarak bırak
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

prediction_history = deque(maxlen=10)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    label_to_display = "Tespit ediliyor..."

    if detection_result.hand_landmarks:
        row_data = [0.0] * 96
        # 12 elemanlı boş bir açı listesi oluştur (4 sağ el + 4 sol el + 4 boş)
        angles_buffer = [0.0] * 12

        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            hand_label = detection_result.handedness[idx][0].category_name
            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y

            temp_coords = []
            for lm in hand_landmarks:
                temp_coords.extend([lm.x - wrist_x, lm.y - wrist_y])

            # Sağ el ise ilk 4 slot, Sol el ise 4'ten 8'e kadar olan slotları doldur
            offset = 0 if hand_label == "Right" else 4

            if hand_label == "Right":
                row_data[0:42] = temp_coords
            else:
                row_data[42:84] = temp_coords

            # Açıları hesapla
            for i in range(4):
                base = i * 4 + 1
                a = [hand_landmarks[base].x, hand_landmarks[base].y]
                b = [hand_landmarks[base + 1].x, hand_landmarks[base + 1].y]
                c = [hand_landmarks[base + 2].x, hand_landmarks[base + 2].y]
                angles_buffer[offset + i] = calculate_angle(a, b, c)

        # 96 sütunlu veriyi tamamla (Artık her zaman 12 eleman var, hata vermez)
        row_data[84:96] = angles_buffer

        # Tahmin Et
        input_data = np.array([row_data])
        scaled_data = scaler.transform(input_data)
        final_data = imputer.transform(scaled_data)  # imputer'ı unutma
        prediction_id = model.predict(final_data)[0]

        # Harfe çevir
        predicted_letter = le.inverse_transform([prediction_id])[0]
        prediction_history.append(predicted_letter)
        most_common = max(set(prediction_history), key=prediction_history.count)
        label_to_display = f"Harf: {most_common}"
    else:
        prediction_history.clear()

    cv2.putText(frame, label_to_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow('TID Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()