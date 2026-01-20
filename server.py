from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)
CORS(app)

mp_hands = mp.solutions.hands.Hands()
ALERT_DATA = {
    "status": "SAFE",
    "lat": "",
    "lon": "",
    "time": ""
}

@app.route("/upload", methods=["POST"])
def upload():
    global ALERT_DATA

    file = request.files["frame"]
    lat = request.form["lat"]
    lon = request.form["lon"]

    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if result.multi_hand_landmarks:
        ALERT_DATA = {
            "status": "FIST_DETECTED",
            "lat": lat,
            "lon": lon,
            "time": time.ctime()
        }
        cv2.imwrite("alert.jpg", frame)

    return "OK"

@app.route("/alert")
def alert():
    return jsonify(ALERT_DATA)

@app.route("/image")
def image():
    return send_file("alert.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
