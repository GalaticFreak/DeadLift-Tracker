import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import winsound  # For beep sound (Windows)
from utils import calculate_angle

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

data = []
labels = []
good_count = 0
bad_count = 0

print("➡ Press 'g' for GOOD form, 'b' for BAD form, 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠️ Camera frame not received.")
        continue

    frame = cv2.flip(frame, 1)  # Mirror view
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Key joints
        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        angle_knee = calculate_angle(hip, knee, ankle)
        angle_hip = calculate_angle(shoulder, hip, knee)

        features = [angle_knee, angle_hip]

        # Show live angles
        cv2.putText(frame, f"Hip Angle: {int(angle_hip)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Knee Angle: {int(angle_knee)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Show counts
        cv2.rectangle(frame, (480, 0), (640, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Good: {good_count}", (490, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Bad : {bad_count}", (490, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Labeling logic
        key = cv2.waitKey(10) & 0xFF
        if key == ord('g'):
            labels.append("good")
            data.append(features)
            good_count += 1
            print(f"✔ GOOD saved ({good_count})")
            winsound.Beep(1000, 150)
        elif key == ord('b'):
            labels.append("bad")
            data.append(features)
            bad_count += 1
            print(f"✘ BAD saved ({bad_count})")
            winsound.Beep(600, 150)
        elif key == ord('q'):
            print("🔚 Data collection stopped.")
            break

    cv2.imshow('Collecting Data (Press g/b/q)', frame)

cap.release()
cv2.destroyAllWindows()

# Save data
df = pd.DataFrame(data, columns=['knee_angle', 'hip_angle'])
df['label'] = labels
df.to_csv("form_data.csv", index=False)
print("📁 Data saved to form_data.csv")
