from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)

# Load CNN model
model_path = "heartbeat_cnn_model.h5"
cnn_model = load_model(model_path)

user_data = {}

def preprocess_frame_for_cnn(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0)

def calculate_heartbeat_cnn(frames):
    predictions = []
    for frame in frames:
        processed_frame = preprocess_frame_for_cnn(frame)
        prediction = cnn_model.predict(processed_frame)
        predictions.append(prediction[0][0])
    return round(np.mean(predictions) * 60)  # Convert to BPM

@app.route('/', methods=['GET', 'POST'])
def landing_page():
    if request.method == 'POST':
        global user_data
        user_data = {
            "name": request.form['name'],
            "age": int(request.form['age']),
            "height": int(request.form['height']),
            "weight": int(request.form['weight']),
        }
        return redirect(url_for('heartbeat_page'))
    return render_template('landing.html')

@app.route('/heartbeat', methods=['GET', 'POST'])
def heartbeat_page():
    if request.method == 'POST':
        frames = request.files.getlist('video_frames')
        processed_frames = [cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) for f in frames]
        bpm = calculate_heartbeat_cnn(processed_frames)
        age = user_data.get("age")

        # Define normal range based on age
        if age <= 18:
            normal_range = (70, 100)
        elif 19 <= age <= 40:
            normal_range = (60, 100)
        else:
            normal_range = (60, 90)

        if normal_range[0] <= bpm <= normal_range[1]:
            message = f"Hi {user_data['name']}, your heartbeat is normal. Keep up your healthy lifestyle!"
        elif bpm < normal_range[0]:
            message = f"Hi {user_data['name']}, your heartbeat is low. Consider consulting a doctor and ensuring adequate hydration and nutrition."
        else:
            message = f"Hi {user_data['name']}, your heartbeat is high. Relax, breathe deeply, and consider consulting a doctor if it persists."

        return jsonify({"bpm": bpm, "message": message})
    return render_template('heartbeat.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
