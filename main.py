import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import urllib.request
import os

# 1. Model Dosyasını İndir (Sadece ilk çalışmada indirir)
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Model dosyası indiriliyor...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("İndirme tamamlandı!")

# 2. Hand Landmarker (El İzleyici) Ayarlarını Yapılandır
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. Bağlantı Noktaları (Eski mp_hands_connections yerine manuel liste)
# MediaPipe'in standart 21 noktalı el modeli bağlantı şeması
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Baş parmak
    (0, 5), (5, 6), (6, 7), (7, 8),  # İşaret parmağı
    (5, 9), (9, 10), (10, 11), (11, 12),  # Orta parmak
    (9, 13), (13, 14), (14, 15), (15, 16),  # Yüzük parmağı
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Serçe parmak ve taban
]

# 4. Kamerayı Başlat
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Kamera açılıyor... Çıkmak için 'q' tuşuna basın.")

# 5. Modeli Başlat ve Döngüye Gir
with HandLandmarker.create_from_options(options) as landmarker:
    frame_timestamp_ms = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Kamera okunamadı.")
            continue

        frame_timestamp_ms += int(1000 / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 33)

        # OpenCV BGR'yi MediaPipe'in beklediği RGB formatına çevir
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Image objesi oluştur
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Modeli çalıştır
        hand_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 6. Sonuçları Ekrana Çiz (Tamamen manuel, solutions kullanılmadan)
        if hand_landmarker_result.hand_landmarks:
            for hand_landmarks in hand_landmarker_result.hand_landmarks:
                h, w, _ = frame.shape

                # Önce çizgileri çiz
                for connection in HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]

                    start_point = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                    end_point = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))

                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # Mavi çizgiler

                # Sonra noktaları (eklem yerlerini) çiz
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Yeşil noktalar

        # Görüntüyü göster
        cv2.imshow("MediaPipe Tasks API - El Takibi", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()