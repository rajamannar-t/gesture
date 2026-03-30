import cv2
import mediapipe as mp
import joblib
import json
import numpy as np

# Load trained model
model = joblib.load('gesture_model.pkl')

# Load cultural data
with open("cultural_data.json", "r") as f:
    cultural_data = json.load(f)

# Normalize JSON keys
gesture_dict = {
    g["gesture_name"].lower().replace(" ", "_"): g
    for g in cultural_data
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Press 'q' to exit...")

last_prediction = None  # to avoid spam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            if len(landmarks) == 42:
                # ✅ FIX sklearn warning
                prediction = model.predict(np.array([landmarks]))[0]

                # ✅ normalize prediction (IMPORTANT FIX)
                prediction_key = prediction.lower().replace(" ", "_")

                gesture_info = gesture_dict.get(prediction_key)

                if gesture_info:
                    name = gesture_info["gesture_name"]
                    meaning = gesture_info["meaning"]

                    if gesture_info["offensive_in"]:
                        warning = f"Offensive in {', '.join(gesture_info['offensive_in'])}"
                    else:
                        warning = "No major issues"

                    # ✅ avoid terminal spam
                    if last_prediction != name:
                        print(f"\nGesture Detected: {name}")
                        print(f"Meaning: {meaning}")
                        print(f"Warning: {warning}")
                        last_prediction = name

                    # ===== SCREEN OUTPUT =====
                    cv2.putText(frame, f"Gesture Detected: {name}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    cv2.putText(frame, f"Meaning: {meaning}", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                    cv2.putText(frame, f"Warning: {warning}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                else:
                    cv2.putText(frame, prediction, (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Cultural Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()