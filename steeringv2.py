import cv2
import mediapipe as mp
import pydirectinput as pyput
import time
import numpy as np
import tensorflow as tf
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.8,
                        min_tracking_confidence=0.5)

screen_width = 1920  # Replace with your screen width

cap = cv2.VideoCapture(0)

ANGLE_RESOLUTION = 0.1 # Finer resolution = more accurate, larger table
NUM_ANGLES = int(math.pi / 2 / ANGLE_RESOLUTION) + 1 

# Create the Lookup Table
angle_lut = [math.acos(np.clip(i * ANGLE_RESOLUTION, -1.0, 1.0)) for i in range(NUM_ANGLES)]

def calculate_steering(landmarks, screen_width):
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # Define the vectors
    vec1 = np.array([pinky_tip.x - wrist.x, pinky_tip.y - wrist.y])  # From wrist to pinky tip
    vec2 = np.array([middle_tip.x - wrist.x, middle_tip.y - wrist.y])  # From wrist to middle finger tip
    
    print("vec1:", vec1)
    print("vec2:", vec2)
    print("norm(vec1):", np.linalg.norm(vec1))
    print("norm(vec2):", np.linalg.norm(vec2))

     # Calculate the angle
    angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.clip(angle, -1.0, 1.0)  # Ensure angle is within valid range
    angle_index = int(min(max(0, angle / ANGLE_RESOLUTION), NUM_ANGLES - 1))
    angle = angle_lut[angle_index]
    
    # Determine tilt direction 
    if pinky_tip.x < wrist.x:  
        angle = -angle

    # Calculate absolute tilt angle 
    absolute_angle = abs(angle)
    
    print("Angle:", angle)

    # Steering control based on tilt
    steering_threshold = 0.4  # Adjust this based on tilt sensitivity
    steering_value = 0.0

    if absolute_angle > steering_threshold:
        steering_value = min(1.0, abs(angle) / np.pi * 1.5) * np.sign(angle)
        
        print("Absolute Angle:", absolute_angle)
        print("Steering Value:", steering_value)
        
    return steering_value


def control_steering(steering_value):
    if steering_value > 0.3:  # Threshold for right turn
        pyput.keyDown("d")
    elif steering_value < -0.3:  # Threshold for left turn
        pyput.keyDown('a')
    else:  
        # Release keys if approximately centered
        pyput.keyUp('a') 
        pyput.keyUp('d') 



OPTIMIZE_DETECTION = True
OPTIMIZE_DRAWING = False 
RESIZE_WIDTH = 640  # Adjust as needed

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break

    # Optimization (Resize if needed)
    if OPTIMIZE_DETECTION:
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(RESIZE_WIDTH * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)
        
    frame.flags.writeable = False    

    # Measure before detection 
    detect_start_time = time.time() 
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
    detect_end_time = time.time()
    
    frame.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:  
            landmarks = hand_landmarks 

            calc_start_time = time.time()  
            steering_value = calculate_steering(landmarks, screen_width)
            calc_end_time = time.time()

            control_start_time = time.time()
            control_steering(steering_value)
            control_end_time = time.time()

            print(f"Calculation Time: {calc_end_time - calc_start_time:.4f}s")
            print(f"Control Time: {control_end_time - control_start_time:.4f}s")

            # Optimization: Conditional Drawing 
            if OPTIMIZE_DRAWING:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)


    draw_end_time = time.time() 

    # Print profiling results
    print(f"Detection Time: {detect_end_time - detect_start_time:.4f}s")
    print(f"Drawing Time: {draw_end_time - detect_start_time:.4f}s") 

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Steering', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
