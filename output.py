import cv2
import mediapipe as mp
import numpy as np
import pickle
import yaml

# Load gesture labels from data.yaml
with open("data.yaml", "r") as f:
    label_map = yaml.safe_load(f)
    gesture_names = {v: k for k, v in label_map.items()}

# Load the trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print("🖐️ Gesture recognition started. Press ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract (x, y) landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            if len(landmarks) == 42:  # 21 points × 2 (x, y)
                prediction = model.predict([landmarks])[0]
                gesture_name = gesture_names.get(prediction, "Unknown")
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    # ESC key to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
