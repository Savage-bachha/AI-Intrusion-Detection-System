import cv2
import face_recognition
import numpy as np
import os
import time
import sqlite3
import threading
import datetime
import torch
from ultralytics import YOLO
from twilio.rest import Client
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Twilio Setup
client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
TWILIO_PHONE = os.getenv("TWILIO_PHONE")
USER_PHONE = os.getenv("USER_PHONE")

# ✅ SQLite Database Setup
conn = sqlite3.connect("intruders.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS intruders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                timestamp TEXT,
                image_path TEXT)''')
conn.commit()

# ✅ YOLOv8 Model for Object Detection
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO('yolov8n.pt').to(device)

# ✅ Motion Detector (Background Subtraction)
motion_detector = cv2.createBackgroundSubtractorMOG2()

# ✅ Load Known Faces
known_faces, known_names = [], []
faces_dir = "known_faces"
os.makedirs(faces_dir, exist_ok=True)

for filename in os.listdir(faces_dir):
    if filename.endswith((".jpg", ".png")):
        img_path = os.path.join(faces_dir, filename)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(filename.split(".")[0])

# ✅ Function to Save Intruder Image
def save_intruder_image(frame):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("intruders", exist_ok=True)
    filename = f"intruders/intruder_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename

# ✅ Function to Log Intruder in Database
def log_intruder(name, image_path):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        c.execute("INSERT INTO intruders (name, timestamp, image_path) VALUES (?, ?, ?)", 
                  (name, timestamp, image_path))
        conn.commit()
        print(f"✅ Logged Intruder: {name} at {timestamp}")
    except Exception as e:
        print(f"⚠️ Database Error: {e}")

# ✅ Function to Send SMS Alert After a 15-Second Countdown
ALERT_DELAY = 15  
LAST_ALERT_TIME = 0  

def send_alert(name, image_path):
    global LAST_ALERT_TIME
    current_time = time.time()

    if current_time - LAST_ALERT_TIME > ALERT_DELAY:
        def sms_thread():
            print(f"⏳ Waiting {ALERT_DELAY} seconds before sending SMS...")
            time.sleep(ALERT_DELAY)

            message = client.messages.create(
                body=f"🚨 Intruder Alert: {name} detected!\nCheck image: {image_path}",
                from_=TWILIO_PHONE,
                to=USER_PHONE,
                provide_feedback=True
            )
            print(f"✅ SMS Sent: {message.sid}")

        threading.Thread(target=sms_thread).start()
        LAST_ALERT_TIME = current_time

# ✅ Open Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

motion_detected = False
motion_frames = 0  
frame_skip = 5  
frame_count = 0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Frame Skipping for Performance
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  

    # ✅ Motion Detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_mask = motion_detector.apply(gray_frame)
    motion_score = np.sum(motion_mask > 200)  

    if motion_score > 5000:
        motion_frames += 1
    else:
        motion_frames = max(0, motion_frames - 1)

    if motion_frames > 3:  
        results = model(frame, stream=True)

        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]

                if label == "person":
                    face_frame = frame[y1:y2, x1:x2]
                    if face_frame.size == 0:
                        continue

                    rgb_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(rgb_frame)
                    name = "Unknown"

                    if face_encodings:
                        face_encoding = face_encodings[0]
                        if known_faces:
                            face_distances = face_recognition.face_distance(known_faces, face_encoding)
                            if len(face_distances) > 0:
                                best_match = np.argmin(face_distances)
                                if face_distances[best_match] < 0.5:
                                    name = known_names[best_match]

                    if name == "Unknown":  
                        image_path = save_intruder_image(frame)
                        log_intruder(name, image_path)
                        send_alert(name, image_path)

                    text_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), text_color, 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    cv2.imshow("Intruder Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
