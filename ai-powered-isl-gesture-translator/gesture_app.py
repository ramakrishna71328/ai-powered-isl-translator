from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from gtts import gTTS
import tempfile
import threading
import pygame

app = Flask(__name__)

# Global variable to store current gesture
current_gesture = "None"

# File and Directory Paths
GESTURE_FILE = "gesture_encodings.pkl"
os.makedirs("gesture_images", exist_ok=True)

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.75)

# Load or initialize gesture encodings
try:
    with open(GESTURE_FILE, "rb") as f:
        gesture_encodings = pickle.load(f)
except FileNotFoundError:
    gesture_encodings = {}

# Functions
def extract_landmark_vector(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        coords = []
        for lm in landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return normalize([coords])[0]
    return None

def validate_and_add_gesture(name, samples):
    samples = np.array(samples)
    if name in gesture_encodings:
        existing = np.array(gesture_encodings[name])
        if existing.shape[1] == samples.shape[1]:
            gesture_encodings[name] = np.vstack([existing, samples])
        else:
            gesture_encodings[name] = samples
    else:
        gesture_encodings[name] = samples
    with open(GESTURE_FILE, "wb") as f:
        pickle.dump(gesture_encodings, f)

def capture_hand_gesture():
    cap = cv2.VideoCapture(0)
    samples = []
    while len(samples) < 5:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        box_start = (w // 3, h // 3)
        box_end = (2 * w // 3, 2 * h // 3)

        roi = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]
        vector = extract_landmark_vector(roi)
        if vector is not None:
            cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, box_start, box_end, (0, 0, 255), 2)

        cv2.putText(frame, f"Sample {len(samples)+1}/5 | Press 'q' to capture | 'c' to cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Capture Gesture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and vector is not None:
            samples.append(vector)
        elif key == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            return None
    cap.release()
    cv2.destroyAllWindows()
    return samples if len(samples) == 5 else None

def recognize_gesture(vector, encodings, threshold=0.3):
    min_distance = float("inf")
    recognized_name = None
    vector = np.array(vector).reshape(1, -1)

    for name, vectors in encodings.items():
        vectors = np.array(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != vector.shape[1]:
            continue
        distance = euclidean_distances(vector, vectors).min()
        if distance < min_distance and distance < threshold:
            min_distance = distance
            recognized_name = name

    confidence = 1 - (min_distance / threshold) if recognized_name else 0
    return recognized_name, min_distance, confidence

def speak(text):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts = gTTS(text=text)
        tts.save(fp.name + ".mp3")
        pygame.mixer.init()
        pygame.mixer.music.load(fp.name + ".mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

# Webcam Recognition Function
def webcam_recognition():
    global current_gesture
    cap = cv2.VideoCapture(0)
    pygame.mixer.init()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        box_start = (w // 3, h // 3)
        box_end = (2 * w // 3, 2 * h // 3)

        roi = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]
        vector = extract_landmark_vector(roi)
        
        if vector is not None:
            name, _, conf = recognize_gesture(vector, gesture_encodings)
            recognized_name = name if name else "Unknown"
            current_gesture = recognized_name
            display = f"{recognized_name} ({conf:.2%})"
            cv2.putText(frame, display, (box_start[0], box_start[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)
        else:
            current_gesture = "None"
            cv2.putText(frame, "No Hand", (box_start[0], box_start[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, box_start, box_end, (0, 0, 255), 2)

        cv2.imshow("Real-Time Gesture Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if current_gesture and current_gesture not in ["None", "Unknown"]:
                speak(current_gesture)

    cap.release()
    cv2.destroyAllWindows()

# Start the recognition thread only once
recognition_thread = None

def start_recognition_thread():
    global recognition_thread
    if recognition_thread is None or not recognition_thread.is_alive():
        recognition_thread = threading.Thread(target=webcam_recognition)
        recognition_thread.start()

# Flask Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/add_gesture', methods=['GET', 'POST'])
def add_gesture():
    if request.method == 'POST':
        name = request.form['gesture_name']
        samples = capture_hand_gesture()
        if samples:
            validate_and_add_gesture(name, samples)
            return redirect(url_for('home'))
        else:
            return "Gesture capture cancelled."
    return render_template('add_gesture.html')

@app.route('/recognize')
def recognize():
    start_recognition_thread()
    return render_template('recognize.html', gesture=current_gesture)

@app.route('/speak_gesture')
def speak_gesture():
    global current_gesture
    if current_gesture and current_gesture != "None" and current_gesture != "Unknown":
        speak(current_gesture)
    return redirect(url_for('recognize'))

@app.route('/update_gesture', methods=['GET', 'POST'])
def update_gesture():
    if request.method == 'POST':
        name = request.form['gesture_name']
        if name in gesture_encodings:
            samples = capture_hand_gesture()
            if samples:
                validate_and_add_gesture(name, samples)
                return redirect(url_for('home'))
            else:
                return "Gesture capture cancelled."
        else:
            return "Gesture not found."
    return render_template('update_gesture.html')

@app.route('/list_gestures')
def list_gestures():
    return render_template('list_gestures.html', gestures=list(gesture_encodings.keys()))

@app.route('/delete_gesture', methods=['GET', 'POST'])
def delete_gesture():
    if request.method == 'POST':
        name = request.form['gesture_name']
        if name in gesture_encodings:
            del gesture_encodings[name]
            with open(GESTURE_FILE, "wb") as f:
                pickle.dump(gesture_encodings, f)
            return redirect(url_for('home'))
        else:
            return "Gesture not found."
    return render_template('delete_gesture.html')

# ✅✅✅ --- ADD THIS BELOW ---
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

# ✅✅✅ --- PORT BINDING FIX DONE ---
