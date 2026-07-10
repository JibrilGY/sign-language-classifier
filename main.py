import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque  # 🌟 Titremeyi engellemek için kuyruk yapısı

# 1. Modeli ve Scaler'ı Yükle
model = joblib.load("gesture_mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl") # YÜKLE

# 2. MediaPipe Tasks API Setup (Senin sistemin için en doğrusu)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3  # Burayı değiştirebilirsin
)
detector = vision.HandLandmarker.create_from_options(options)

# 3. 🌟 TİTREMEYİ ENGELLEYEN TAMPON
prediction_history = deque(maxlen=10)  # Son 10 kareyi hafızada tut

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe ile tespit
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    label_to_display = "Tespit ediliyor..."

    if detection_result.hand_landmarks:
        # Veriyi çıkar (extract_features.py ile aynı mantık)
        row_data = [0.0] * 84
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            hand_label = detection_result.handedness[idx][0].category_name
            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y

            temp_coords = []
            for lm in hand_landmarks:
                temp_coords.extend([lm.x - wrist_x, lm.y - wrist_y])

            if hand_label == "Right":
                row_data[0:42] = temp_coords
            else:
                row_data[42:84] = temp_coords

        # Tahmin Et
        input_data = np.array([row_data])
        scaled_data = scaler.transform(input_data)
        final_data = imputer.transform(scaled_data)  # DOLDUR
        prediction = model.predict(scaled_data)[0]

        # 🌟 TAMPONA EKLE VE OYLA
        prediction_history.append(prediction)
        most_common = max(set(prediction_history), key=prediction_history.count)
        label_to_display = f"Harf: {most_common}"
    else:
        prediction_history.clear()  # El yoksa geçmişi sıfırla

    # Ekrana yaz
    cv2.putText(frame, label_to_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow('TID Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()