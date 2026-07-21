import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv
import glob
import random
import numpy as np  # 🌟 Açı hesabı için numpy lazım

# 1. Configuration
DATASET_PATH = "dataset"
CSV_FILENAME = "tid_dataset_2hands.csv"
MAX_IMAGES_PER_CLASS = 500
MODEL_PATH = "hand_landmarker.task"


# 🌟 AÇI HESAPLAMA FONKSİYONU
def calculate_angle(a, b, c):
    a = np.array(a)  # İlk nokta
    b = np.array(b)  # Orta nokta (eklem)
    c = np.array(c)  # Son nokta

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, min_hand_detection_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Initialize CSV with Headers (84 Koordinat + 24 Açı = 108 Sütun)
with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    headers = ['label']
    for i in range(21): headers.extend([f'rx{i}', f'ry{i}'])  # Right
    for i in range(21): headers.extend([f'lx{i}', f'ly{i}'])  # Left
    # Yeni Açı sütunları
    for i in range(12): headers.append(f'angle_{i}')
    writer.writerow(headers)

print("Feature extraction with Geometry (Angles) started...")
total_processed = 0
classes = sorted(os.listdir(DATASET_PATH))
for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_path): continue
    label = class_name
    all_images = glob.glob(os.path.join(class_path, "*.jpg")) + glob.glob(os.path.join(class_path, "*.png"))
    selected_images = random.sample(all_images, min(len(all_images), MAX_IMAGES_PER_CLASS))

    print(f"Processing '{label}'... ", end="", flush=True)
    class_processed = 0

    for image_path in selected_images:
        try:
            mp_image = mp.Image.create_from_file(image_path)
            results = detector.detect(mp_image)
        except Exception:
            continue

        if not results.hand_landmarks:
            continue

        # 96 elemanlı sabit boyutlu satır (84 koordinat + 12 açı yeri)
        row_data = [0.0] * 96
        angles = []

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            if idx >= 2:  # Güvenlik için en fazla 2 el
                break
            hand_label = results.handedness[idx][0].category_name
            wrist_x, wrist_y = hand_landmarks[0].x, hand_landmarks[0].y

            coords = []
            for lm in hand_landmarks:
                coords.extend([lm.x - wrist_x, lm.y - wrist_y])

            if hand_label == "Right":
                row_data[0:42] = coords
            else:
                row_data[42:84] = coords

            # Parmak açılarını hesapla (İşaret, Orta, Yüzük, Serçe)
            for i in range(4):
                base = i * 4 + 1
                a = [hand_landmarks[base].x, hand_landmarks[base].y]
                b = [hand_landmarks[base + 1].x, hand_landmarks[base + 1].y]
                c = [hand_landmarks[base + 2].x, hand_landmarks[base + 2].y]
                angles.append(calculate_angle(a, b, c))

        # Hesaplanan açıları 84. indexten itibaren yerleştir (fazlası sıfır kalır)
        if angles:
            row_data[84:84 + len(angles)] = angles

        with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([label] + row_data)
        class_processed += 1
        total_processed += 1
    print(f"Success: {class_processed}")