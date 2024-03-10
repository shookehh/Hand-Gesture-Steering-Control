import cv2
import mediapipe as mp
import pyautogui
import pydirectinput as pyput
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) 

cap = cv2.VideoCapture(0)

def calculate_steering(landmarks):

    # Get wrist and pinky fingertip landmarks (CORRECTED)
    wrist_landmark = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    wrist_y = wrist_landmark.y
    pinky_tip_landmark = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_tip_y = pinky_tip_landmark.y

    # Calculate the tilt angle
    tilt_angle = (wrist_y - pinky_tip_y) / wrist_y  

    # Map angle to steering value (-1 left, 1 right)
    steering_value = tilt_angle * 2 
    return steering_value

def control_steering(steering_value):
    if steering_value > 0.5:  # Threshold for right turn
        pyput.keyDown("d")
    elif steering_value < -0.5:  # Threshold for left turn
        pyput.keyDown('a')
    else:  
        # Release keys if approximately centered
        pyput.keyUp('a') 
        pyput.keyUp('d') 

# Optimization Parameters 
OPTIMIZE_DETECTION = True
OPTIMIZE_DRAWING = False 
RESIZE_WIDTH = 640  # Adjust as needed

while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()
    if not success:
        break  

    # Flip the image 
    frame = cv2.flip(frame, 1)

    # Optimization: Resize if needed 
    if OPTIMIZE_DETECTION:
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(RESIZE_WIDTH * frame.shape[0] / frame.shape[1])), interpolation=cv2.INTER_AREA)

    # Measure before hand detection 
    detect_start_time = time.time() 
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
    detect_end_time = time.time()

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]  
        steering_value = calculate_steering(landmarks)
        control_steering(steering_value)

        # Optimization: Conditional Drawing 
        if OPTIMIZE_DRAWING:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Measure drawing time 
    draw_end_time = time.time() 

    # Print profiling results
    print(f"Detection Time: {detect_end_time - detect_start_time:.4f}s")
    print(f"Drawing Time: {draw_end_time - detect_start_time:.4f}s") 

    cv2.imshow('Hand Steering', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()