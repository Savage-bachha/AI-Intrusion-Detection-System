from flask import Flask, render_template, Response, jsonify, request
import cv2
import face_recognition
import numpy as np
import os
import time
import sqlite3
import threading
from ultralytics import YOLO
from twilio.rest import Client
from dotenv import load_dotenv

app = Flask(__name__)

# ✅ Load environment variables
load_dotenv()

# ✅ Twilio Setup
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")
USER_PHONE = os.getenv("USER_PHONE")

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

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
model = YOLO('yolov8n.pt')

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

# ✅ Camera Setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ✅ Motion Detection
motion_detector = cv2.createBackgroundSubtractorMOG2()
prev_frame = None
motion_detected = False
motion_threshold = 5000

# ✅ Alert & Logging Variables
detected_name = "No Detection"
intruder_detected = False
ALERT_DELAY = 15
last_alert_time = 0
countdown_value = 15

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
    c.execute("INSERT INTO intruders (name, timestamp, image_path) VALUES (?, ?, ?)", 
              (name, timestamp, image_path))
    conn.commit()
    print(f"🔴 Logged Intruder: {name} at {timestamp}")

# ✅ Function to Send SMS Alert
def send_alert(name, image_path):
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time >= ALERT_DELAY:
        def sms_thread():
            print(f"📡 Sending SMS Alert for: {name}")
            message = client.messages.create(
                body=f"🚨 Intruder Alert: {name} detected!\nCheck image: {image_path}",
                from_=TWILIO_PHONE,
                to=USER_PHONE
            )
            print(f"✅ SMS Sent: {message.sid}")

        threading.Thread(target=sms_thread).start()
        last_alert_time = current_time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global detected_name, intruder_detected, prev_frame, motion_detected

        while True:
            success, frame = cap.read()
            if not success:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # ✅ Motion Detection
            if prev_frame is not None:
                frame_diff = cv2.absdiff(prev_frame, gray_frame)
                motion_mask = motion_detector.apply(gray_frame)
                motion_score = np.sum(frame_diff > 50) + np.sum(motion_mask > 200)

                motion_detected = motion_score > motion_threshold
            prev_frame = gray_frame.copy()

            if motion_detected:
                results = model(frame, stream=True)
                intruder_detected = False  

                for result in results:
                    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                        x1, y1, x2, y2 = map(int, box)
                        label = model.names[int(cls)]

                        if label == "person":
                            face_frame = frame[y1:y2, x1:x2]
                            detected_name = "Unknown"

                            if face_frame.size > 0:
                                rgb_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                                face_encodings = face_recognition.face_encodings(rgb_frame)

                                if face_encodings:
                                    face_encoding = face_encodings[0]
                                    if known_faces:
                                        face_distances = face_recognition.face_distance(known_faces, face_encoding)
                                        if len(face_distances) > 0:
                                            best_match = np.argmin(face_distances)
                                            if face_distances[best_match] < 0.5:
                                                detected_name = known_names[best_match]

                            intruder_detected = detected_name == "Unknown"

                            if intruder_detected:
                                image_path = save_intruder_image(frame)
                                log_intruder(detected_name, image_path)
                                send_alert(detected_name, image_path)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if intruder_detected else (0, 255, 0), 2)
                            cv2.putText(frame, detected_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    global countdown_value
    return jsonify({
        "detected": detected_name,
        "intruder": intruder_detected,
        "countdown": countdown_value if intruder_detected else None
    })

@app.route('/trigger_alert', methods=['POST'])
def trigger_alert():
    send_alert("Manual Alert", "N/A")
    return jsonify({"message": "Alert triggered successfully!"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')