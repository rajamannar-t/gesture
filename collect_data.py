import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

gesture_name = input("Enter gesture name: ")

file_name = "gesture_data.csv"
file_exists = os.path.isfile(file_name)

cap = cv2.VideoCapture(0)

# ✅ store last valid landmarks
last_landmarks = None

with open(file_name, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    if not file_exists:
        header = []
        for i in range(21):
            header += [f'x{i}', f'y{i}']
        header.append('label')
        writer.writerow(header)

    print("Press 's' to save sample, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmarks = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmarks.append(round(lm.x, 6))
                    landmarks.append(round(lm.y, 6))

                # ✅ update only when valid
                if len(landmarks) == 42:
                    last_landmarks = landmarks.copy()

        # debug print
   

        cv2.imshow("Collect Gesture Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("S pressed")

            if last_landmarks is not None:
                writer.writerow(last_landmarks + [gesture_name])
                f.flush()
                print("Sample Saved ✅")
            else:
                print("Hand not detected properly ❌")

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()