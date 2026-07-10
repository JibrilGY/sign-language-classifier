import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import os
import csv
import glob
import random

# 1. Configuration & Paths
DATASET_PATH = "dataset/train"  # Update this if your path is just "dataset"
CSV_FILENAME = "tid_dataset.csv"
MAX_IMAGES_PER_CLASS = 500
MODEL_PATH = 'hand_landmarker.task'

# 2. MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

# 3. Initialize CSV with Headers
with open(CSV_FILENAME, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    headers = ['label']
    for i in range(21):
        headers.extend([f'x{i}', f'y{i}'])
    writer.writerow(headers)

print(f"Starting feature extraction... Max {MAX_IMAGES_PER_CLASS} images per class.")
total_processed = 0
total_errors = 0

# 4. Processing Loop
with HandLandmarker.create_from_options(options) as landmarker:
    classes = sorted(os.listdir(DATASET_PATH))

    for class_name in classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_path):
            continue

        label = class_name
        all_images = glob.glob(os.path.join(class_path, "*.jpg")) + glob.glob(os.path.join(class_path, "*.png"))

        if len(all_images) > MAX_IMAGES_PER_CLASS:
            selected_images = random.sample(all_images, MAX_IMAGES_PER_CLASS)
        else:
            selected_images = all_images

        print(f"Processing class '{label}'... ({len(selected_images)} images)")

        class_processed = 0
        class_errors = 0

        for image_path in selected_images:
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    wrist_x = hand_landmarks[0].x
                    wrist_y = hand_landmarks[0].y

                    normalized_coords = []
                    for landmark in hand_landmarks:
                        normalized_coords.extend([
                            landmark.x - wrist_x,
                            landmark.y - wrist_y
                        ])

                    with open(CSV_FILENAME, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        row = [label] + normalized_coords
                        writer.writerow(row)

                    class_processed += 1
                    total_processed += 1
            else:
                class_errors += 1
                total_errors += 1

        print(f"  -> Success: {class_processed} | Hand not detected: {class_errors}")

print("-" * 30)
print("Feature Extraction Completed!")
print(f"Total rows written to CSV: {total_processed}")
print(f"Total skipped images: {total_errors}")