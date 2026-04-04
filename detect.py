import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist

# 🔊 Alarm
pygame.mixer.init()
alarm = pygame.mixer.Sound(r"C:\Users\vimal\Downloads\mixkit-classic-alarm-995.wav")

# 🧠 Load CNN model
model = load_model("model/model.h5", compile=False)
IMG_SIZE = 64

# 🎯 Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# 👁️ Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# 📷 Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

# 🔁 Variables
closed_frames = 0
alarm_on = False
last_alarm = 0

THRESHOLD_FRAMES = 12
EAR_THRESHOLD = 0.25

# 🔄 Prediction smoothing
pred_buffer = []
BUFFER_SIZE = 5


# 🧠 EAR function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# 👁️ Crop eye (IMPROVED)
def crop_eye(frame, landmarks, eye_points):
    h, w, _ = frame.shape

    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]

    x_coords = [p[0] for p in coords]
    y_coords = [p[1] for p in coords]

    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)

    pad = 5  # 🔥 smaller padding for cleaner crop
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

    eye = frame[y1:y2, x1:x2]

    if eye is None or eye.size == 0:
        return None

    if eye.shape[0] < 10 or eye.shape[1] < 10:
        return None

    # Normalize
    eye = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
    eye = eye / 255.0
    eye = np.reshape(eye, (1, IMG_SIZE, IMG_SIZE, 3))

    return eye


# 🎥 Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    status = "No Face ❌"

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:

            h, w, _ = frame.shape

            # 👁️ Eye coordinates
            left_eye_pts = [(int(face_landmarks.landmark[i].x * w),
                             int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]

            right_eye_pts = [(int(face_landmarks.landmark[i].x * w),
                              int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            # 🧠 EAR
            left_ear = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)
            ear = (left_ear + right_ear) / 2.0

            # 👁️ CNN eye crop
            left_eye = crop_eye(frame, face_landmarks.landmark, LEFT_EYE)
            right_eye = crop_eye(frame, face_landmarks.landmark, RIGHT_EYE)

            if left_eye is not None and right_eye is not None:

                pred_left = model.predict(left_eye, verbose=0)[0][0]
                pred_right = model.predict(right_eye, verbose=0)[0][0]

                pred = (pred_left + pred_right) / 2

                # 🔄 Smooth prediction
                pred_buffer.append(pred)
                if len(pred_buffer) > BUFFER_SIZE:
                    pred_buffer.pop(0)

                smooth_pred = np.mean(pred_buffer)

                print("EAR:", round(ear, 3), "CNN:", round(smooth_pred, 2))

                # 🔥 HYBRID DECISION
                if ear < EAR_THRESHOLD and smooth_pred < 0.5:
                    closed_frames += 1
                    status = "Eyes Closed 😴"

                    if closed_frames > THRESHOLD_FRAMES:
                        status = "DROWSY! 🚨"

                        if not alarm_on and time.time() - last_alarm > 2:
                            alarm.play(-1)
                            alarm_on = True
                            last_alarm = time.time()

                else:
                    status = "Active 😃"
                    closed_frames = 0

                    if alarm_on:
                        pygame.mixer.stop()
                        alarm_on = False

            # Draw landmarks
            for (x, y) in left_eye_pts + right_eye_pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    else:
        closed_frames = 0
        if alarm_on:
            pygame.mixer.stop()
            alarm_on = False

    # 🎨 Display
    color = (0, 255, 0)
    if "DROWSY" in status:
        color = (0, 0, 255)
    elif "Closed" in status:
        color = (0, 165, 255)

    cv2.putText(frame, status, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Driver Monitoring (Hybrid AI)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()