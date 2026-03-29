import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import tkinter as tk
from PIL import Image, ImageTk

# 🔊 Alarm
pygame.mixer.init()
alarm = pygame.mixer.Sound(r"C:\Users\vimal\Downloads\mixkit-classic-alarm-995.wav")

# 🧠 Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Landmarks
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
NOSE = 1
UPPER_LIP = 13
LOWER_LIP = 14

def dist(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

def EAR(landmarks, eye):
    p1,p2,p3,p4,p5,p6=[landmarks[i] for i in eye]
    return (dist(p2,p6)+dist(p3,p5))/(2.0*dist(p1,p4))

def mouth_open(landmarks):
    return abs(landmarks[UPPER_LIP][1] - landmarks[LOWER_LIP][1])

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

closed_frames = 0
alarm_on = False
last_alarm = 0
fatigue_history = []

EAR_T = 0.25
FRAME_T = 20
MOUTH_T = 25
HEAD_TILT_T = 15

running = False

# GUI
root = tk.Tk()
root.title("AI Driver Monitoring System")
root.geometry("1000x800")
root.configure(bg="#0f172a")

video_label = tk.Label(root, bg="black")
video_label.pack()

status_label = tk.Label(root, text="Status: OFF", font=("Arial",18), bg="#0f172a", fg="white")
status_label.pack()

fatigue_label = tk.Label(root, text="Fatigue: 0%", font=("Arial",16), bg="#0f172a", fg="yellow")
fatigue_label.pack()

score_label = tk.Label(root, text="Driver Score: 100", font=("Arial",16), bg="#0f172a", fg="cyan")
score_label.pack()

def start():
    global running
    running = True
    update_frame()

def stop():
    global running
    running = False
    pygame.mixer.stop()
    status_label.config(text="Status: STOPPED", fg="white")

def update_frame():
    global closed_frames, alarm_on, last_alarm

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h,w,_ = frame.shape
            landmarks = [(int(p.x*w), int(p.y*h)) for p in face_landmarks.landmark]

            ear = (EAR(landmarks, LEFT_EYE)+EAR(landmarks, RIGHT_EYE))/2
            mouth = mouth_open(landmarks)

            # HEAD TILT
            nose_y = landmarks[NOSE][1]
            tilt = abs(nose_y - h//2)

            # EYE LOGIC
            if ear < EAR_T:
                closed_frames += 1
            else:
                closed_frames = 0
                alarm_on = False
                pygame.mixer.stop()
                status_label.config(text="Status: ACTIVE 😃", fg="green")

            fatigue = min(100, closed_frames*3)
            fatigue_history.append(fatigue)

            score = max(0, 100-fatigue)

            # STATES
            if fatigue > 70:
                status_label.config(text="CRITICAL 😡", fg="red")
            elif fatigue > 40:
                status_label.config(text="WARNING ⚠️", fg="orange")

            if closed_frames > FRAME_T:
                if not alarm_on and time.time()-last_alarm > 3:
                    alarm.play(-1)
                    alarm_on = True
                    last_alarm = time.time()

            # YAWN
            if mouth > MOUTH_T:
                cv2.putText(frame,"YAWNING!",(50,200),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

            # HEAD
            if tilt > HEAD_TILT_T:
                cv2.putText(frame,"HEAD DOWN!",(50,250),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

            # DRAW
            for i in LEFT_EYE+RIGHT_EYE:
                cv2.circle(frame, landmarks[i],2,(0,255,0),-1)

    else:
        status_label.config(text="No Face ❌", fg="orange")

    fatigue = min(100, closed_frames*3)
    score = max(0, 100-fatigue)

    fatigue_label.config(text=f"Fatigue: {fatigue}%")
    score_label.config(text=f"Driver Score: {score}")

    # UI PANEL
    cv2.rectangle(frame,(0,0),(320,150),(0,0,0),-1)
    cv2.putText(frame,f"Fatigue: {fatigue}%",(10,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(frame,f"Score: {score}",(10,80),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# Buttons
frame_btn = tk.Frame(root, bg="#0f172a")
frame_btn.pack(pady=10)

tk.Button(frame_btn,text="START",command=start,bg="green",fg="white",width=12).grid(row=0,column=0,padx=10)
tk.Button(frame_btn,text="STOP",command=stop,bg="red",fg="white",width=12).grid(row=0,column=1,padx=10)

root.mainloop()

cap.release()
cv2.destroyAllWindows()