import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# --- Setup ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # Set to 0 for maximum speed

# Initialize MediaPipe Hand Model
# model_complexity=0 is much faster on Raspberry Pi
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Start Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables for FPS and Click logic
pTime = 0
click_cooldown = 0.3  
last_click_time = time.time()

# To smooth the mouse movement
prev_x, prev_y = 0, 0
smoothing = 5 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get Index Finger Tip (Landmark 8)
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            ix, iy = int(index_finger.x * frame_w), int(index_finger.y * frame_h)

            # Map coordinates to screen size
            # We use a slightly smaller range [50, frame_w-50] to make it easier to reach screen edges
            screen_x = np.interp(ix, [50, frame_w - 50], [0, screen_w])
            screen_y = np.interp(iy, [50, frame_h - 50], [0, screen_h])

            # Smooth movement logic
            curr_x = prev_x + (screen_x - prev_x) / smoothing
            curr_y = prev_y + (screen_y - prev_y) / smoothing
            
            # Direct move (removed threading to prevent X11 crashes)
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Click Gesture (Thumb Tip & Index Tip distance)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            tx, ty = int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h)
            
            # Calculate distance between thumb and index
            distance = np.hypot(ix - tx, iy - ty)
            
            # Visualize the "pinch"
            cv2.line(frame, (ix, iy), (tx, ty), (255, 0, 255), 2)

            if distance < 30: 
                current_time = time.time()
                if current_time - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = current_time
                    cv2.putText(frame, "CLICK!", (ix, iy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- Calculate and Display FPS ---
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("AirCursor - Raspberry Pi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()