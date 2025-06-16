import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import winsound  # Windows-only; for beeping
from utils import calculate_angle

# Load model
model = tf.keras.models.load_model("dl_model.h5")

# Init MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
counter = 0
stage = None
last_feedback = ""

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror cam

    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Get keypoints
        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        # Angles
        angle_knee = calculate_angle(hip, knee, ankle)
        angle_hip = calculate_angle(shoulder, hip, knee)

        # Rep logic
        if angle_hip > 160:
            stage = "down"
        if angle_hip < 90 and stage == "down":
            stage = "up"
            counter += 1

        # Prediction
        pred = model.predict(np.array([[angle_knee, angle_hip]]))[0][0]
        form = "Good" if pred > 0.5 else "Bad"

        # Feedback logic
        if form == "Bad":
            if angle_hip < 70:
                feedback = "Straighten your back"
            elif angle_knee > 160:
                feedback = "Bend your knees more"
            else:
                feedback = "Fix your form"

            if last_feedback != feedback:
                winsound.Beep(800, 250)  # Beep on bad form
                last_feedback = feedback
        else:
            feedback = "Good form!"
            last_feedback = ""

        # Draw REPS box
        cv2.rectangle(frame, (0, 0), (250, 100), (0, 0, 0), -1)
        cv2.putText(frame, 'REPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, str(counter), (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,255), 4)

        # Form and Feedback with color indicator
        form_color = (0,255,0) if form == "Good" else (0,0,255)
        border_thickness = 4 if form == "Bad" else 1
        cv2.rectangle(frame, (5,5), (frame.shape[1]-5, frame.shape[0]-5), form_color, border_thickness)

        cv2.putText(frame, f"Form: {form}", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, form_color, 2)
        cv2.putText(frame, f"Feedback: {feedback}", (270, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Show live angle values
        cv2.putText(frame, f"Hip Angle: {int(angle_hip)}", (270, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Knee Angle: {int(angle_knee)}", (270, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Deadlift Tracker - Enhanced', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
