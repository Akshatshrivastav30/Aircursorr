import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# --- Setup ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Variables ---
pTime = 0
prev_x, prev_y = 0, 0
prev_x_raw, prev_y_raw = 0, 0
smoothing = 7 

# Dwell Click Variables
dwell_start_time = None
dwell_duration = 2.0      # Time in seconds to hold still
movement_threshold = 20    # Max pixels allowed to move during dwell
click_cooldown = 1.0       # Prevent rapid double-clicks
last_click_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 1. Get Index Finger Tip
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            ix, iy = int(index_finger.x * frame_w), int(index_finger.y * frame_h)

            # 2. Smooth Mouse Movement
            screen_x = np.interp(ix, [50, frame_w - 50], [0, screen_w])
            screen_y = np.interp(iy, [50, frame_h - 50], [0, screen_h])
            curr_x = prev_x + (screen_x - prev_x) / smoothing
            curr_y = prev_y + (screen_y - prev_y) / smoothing
            
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # 3. Dwell Logic (Hold for 2 Seconds)
            # Calculate how much the finger moved since the last frame
            dist_moved = np.hypot(ix - prev_x_raw, iy - prev_y_raw)

            if dist_moved < movement_threshold:
                if dwell_start_time is None:
                    dwell_start_time = time.time()
                
                elapsed = time.time() - dwell_start_time
                
                # Visual Feedback: Progress Circle
                # Drawing a circle that grows as you get closer to 2 seconds
                angle = int((elapsed / dwell_duration) * 360)
                cv2.ellipse(frame, (ix, iy), (30, 30), 0, 0, angle, (0, 255, 255), 3)
                cv2.putText(frame, f"{elapsed:.1f}s", (ix + 40, iy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if elapsed >= dwell_duration:
                    if time.time() - last_click_time > click_cooldown:
                        pyautogui.click()
                        last_click_time = time.time()
                        dwell_start_time = None # Reset
            else:
                # If moved too much, reset the timer
                dwell_start_time = None

            # Update raw tracking for next frame's movement check
            prev_x_raw, prev_y_raw = ix, iy

    # FPS Display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("AirCursor - Dwell Click", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()