import cv2
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import os
import mediapipe as mp
from gtts import gTTS
import tempfile
import pygame

# File and Directory Paths
GESTURE_FILE = "gesture_encodings.pkl"
IMAGE_DIR = "gesture_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.75)
mp_drawing = mp.solutions.drawing_utils

# Load or initialize gesture encodings
try:
    with open(GESTURE_FILE, "rb") as f:
        gesture_encodings = pickle.load(f)
except FileNotFoundError:
    gesture_encodings = {}

# Extract MediaPipe hand landmarks vector (63D)
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

# Gesture recognition
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

# Convert gesture name to MP3 and play
def speak(text):
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts = gTTS(text=text)
        tts.save(fp.name + ".mp3")
        pygame.mixer.init()
        pygame.mixer.music.load(fp.name + ".mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

# Gesture capture using webcam
def capture_hand_gesture():
    print("Place your hand in the blue box. Press 'q' to capture, or 'c' to cancel.")
    cap = cv2.VideoCapture(0)
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
            cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, box_start, box_end, (0, 0, 255), 2)

        cv2.putText(frame, "Press 'q' to capture | 'c' to cancel", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Capture Gesture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and vector is not None:
            cap.release()
            cv2.destroyAllWindows()
            return vector
        elif key == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
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

# Real-time recognition with speech on button press
def real_time_gesture_recognition():
    cap = cv2.VideoCapture(0)
    recognized_name = "None"
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
            display = f"{recognized_name} ({conf:.2%})"
            cv2.putText(frame, display, (box_start[0], box_start[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Hand", (box_start[0], box_start[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, box_start, box_end, (0, 0, 255), 2)

        cv2.putText(frame, "Press 's' to speak gesture", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Real-Time Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and recognized_name and recognized_name != "Unknown":
            speak(recognized_name)

    cap.release()
    cv2.destroyAllWindows()

# Menu
while True:
    print("\nMenu:")
    print("1. Add Gesture")
    print("2. Update Gesture")
    print("3. Delete Gesture")
    print("4. List Gestures")
    print("5. Real-Time Recognition")
    print("6. Exit")

    choice = input("Choose an option (1-6): ")

    if choice == "1":
        name = input("Enter gesture name: ")
        samples = []
        print("Capture 5 samples:")
        for i in range(5):
            print(f"Sample {i+1}")
            vec = capture_hand_gesture()
            if vec is not None:
                samples.append(vec)
            else:
                print("Capture cancelled.")
                break
        if len(samples) == 5:
            validate_and_add_gesture(name, samples)
            with open(GESTURE_FILE, "wb") as f:
                pickle.dump(gesture_encodings, f)
            print(f"Gesture '{name}' added.")
        else:
            print("Gesture not saved. Not enough samples.")

    elif choice == "2":
        name = input("Enter gesture name to update: ")
        if name in gesture_encodings:
            print("Capture 5 new samples:")
            new_samples = []
            for i in range(5):
                vec = capture_hand_gesture()
                if vec is not None:
                    new_samples.append(vec)
            if new_samples:
                validate_and_add_gesture(name, new_samples)
                with open(GESTURE_FILE, "wb") as f:
                    pickle.dump(gesture_encodings, f)
                print(f"Gesture '{name}' updated.")
        else:
            print(f"Gesture '{name}' not found.")

    elif choice == "3":
        name = input("Enter gesture name to delete: ")
        if name in gesture_encodings:
            del gesture_encodings[name]
            with open(GESTURE_FILE, "wb") as f:
                pickle.dump(gesture_encodings, f)
            print(f"Gesture '{name}' deleted.")
        else:
            print(f"Gesture '{name}' not found.")

    elif choice == "4":
        print("Gestures:", list(gesture_encodings.keys()))

    elif choice == "5":
        real_time_gesture_recognition()

    elif choice == "6":
        with open(GESTURE_FILE, "wb") as f:
            pickle.dump(gesture_encodings, f)
        print("Exiting...")
        break

    else:
        print("Invalid option. Try again.")
