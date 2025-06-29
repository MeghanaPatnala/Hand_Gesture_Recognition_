import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import yaml
import os

# Load labels from data.yaml
with open("data.yaml", "r") as f:
    labels = yaml.safe_load(f)

label_names = list(labels.keys())
print("🎯 Available gestures:")
for i, name in enumerate(label_names):
    print(f"  {i}: {name}")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

data = []  # To store [x1, y1, x2, y2, ..., label]

# Open webcam
cap = cv2.VideoCapture(0)
print("📸 Webcam opened. Show your gesture and press keys 0–9 to save. Press 'q' to quit and save collected data.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the webcam feed
    cv2.imshow("Collecting Gesture Data", frame)
    key = cv2.waitKey(1)

    # Quit and save
    if key == ord('q'):
        break

    # Check for digit key (0–9)
    if key != -1:
        try:
            key_char = chr(key)
            if key_char.isdigit():
                label_idx = int(key_char)
                if label_idx < len(label_names) and results.multi_hand_landmarks:
                    landmarks = []
                    for lm in results.multi_hand_landmarks[0].landmark:
                        landmarks.extend([lm.x, lm.y])
                    landmarks.append(label_idx)  # Add gesture label
                    data.append(landmarks)
                    print(f"✅ Collected sample for gesture: {label_names[label_idx]}")
        except:
            pass  # Ignore invalid keys

cap.release()
cv2.destroyAllWindows()

# Save only if data was collected
if len(data) > 0:
    df = pd.DataFrame(data)
    df.to_csv("gesture_data.csv", index=False)
    print(f"\n📁 Saved {len(data)} samples to gesture_data.csv")
else:
    print("⚠️ No gesture samples collected. Nothing to save.")
