import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False)
mp_drawing = mp.solutions.drawing_utils
REFERENCE_OBJECT_WIDTH = 52
FOCAL_LENGTH = 600

def calculate_measurements(pose_landmarks):
    L_shoulder = mp_pose.PoseLandmark.LEFT_SHOULDER
    R_shoulder = mp_pose.PoseLandmark.RIGHT_SHOULDER
    L_hip = mp_pose.PoseLandmark.LEFT_HIP
    R_hip = mp_pose.PoseLandmark.RIGHT_HIP

    shoulder_left = np.array([pose_landmarks[L_shoulder].x, pose_landmarks[L_shoulder].y])
    shoulder_right = np.array([pose_landmarks[R_shoulder].x, pose_landmarks[R_shoulder].y])
    hip_left = np.array([pose_landmarks[L_hip].x, pose_landmarks[L_hip].y])
    hip_right = np.array([pose_landmarks[R_hip].x, pose_landmarks[R_hip].y])

    shoulder_width = np.linalg.norm(shoulder_left - shoulder_right)
    torso_length = (np.linalg.norm(shoulder_left - hip_left) + np.linalg.norm(shoulder_right - hip_right)) / 2
    waist_width = np.linalg.norm(hip_left - hip_right)

    return {
        "shoulder_width": shoulder_width,
        "torso_length": torso_length,
        "waist_width": waist_width,
    }
def conversion_size(measurements):
    shoulder_width = measurements["shoulder_width"]
    torso_length = measurements["torso_length"]
    waist_width = measurements["waist_width"]

    if shoulder_width < 0.10 and torso_length < 0.5 and waist_width < 0.07:
        return "XS"
    elif 0.10 <= shoulder_width < 0.15 and 0.2 <= torso_length < 0.4 and 0.04 <= waist_width < 0.08:
        return "S"
    elif 0.25 <= shoulder_width < 0.29 and 0.6 <= torso_length < 0.7 and 0.12 <= waist_width < 0.17:
        return "M"
    elif 0.29 <= shoulder_width < 0.32 and 0.7 <= torso_length < 0.8 and 0.17 <= waist_width < 0.22:
        return "L"
    elif shoulder_width >= 0.32 and torso_length >= 0.8 and waist_width >= 0.22:
        return "XL"
    else:
        return "N/A"

def estimate_distance(detected_width, actual_width, focal_length):
    distance = (actual_width * focal_length) / detected_width
    return distance

# I used these numbers with accordance of my measurements to convert pixels into inches( ±2 error rate)
def store_measurements(measurements):
    measurements["waist_width"] *= 240
    measurements["torso_length"] *= 30
    measurements["shoulder_width"] *= 66.5
    df = pd.DataFrame([measurements])
    df = df.round(2)
    df.to_csv('TMG_project_measurement.csv', mode='a', header=True, index=False)
    st.write("Measurements saved to TMG_project_measurements.csv")

st.title('TMG Body Size Estimator')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera_button = st.button('Turn on your camera')
stop_button = st.button('Turn off your camera')
size_button = st.button("Know your measurements in inches")
st.write("Please stand at approximately 80 cm from the camera lens")
stframe = st.empty()
clothing_size = "N/A"
distance = 0

if camera_button:
    st.session_state['capturing'] = True
elif stop_button:
    st.session_state['capturing'] = False

if 'capturing' not in st.session_state:
    st.session_state['capturing'] = False

if st.session_state['capturing']:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            measurements = calculate_measurements(pose_landmarks)
            clothing_size = conversion_size(measurements)

            detected_width_pixels = measurements["shoulder_width"] * frame.shape[1]
            distance = estimate_distance(detected_width_pixels, REFERENCE_OBJECT_WIDTH, FOCAL_LENGTH)

            if size_button:  # I used time module to hold the capturing process so that the user can adjust.
                st.write("Wait for 2 seconds.. while we scan your body")
                time.sleep(2)
                store_measurements(measurements)
                break
        else:
            cv2.putText(frame, "No body parts detected", (240, 540), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        stframe.image(image, caption=f"Clothing Size: {clothing_size} | Distance: {distance:.2f} cm", use_column_width=True)
        if not st.session_state['capturing']:
            break
cap.release()
cv2.destroyAllWindows()