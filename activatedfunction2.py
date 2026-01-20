import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# --- Configuration ---
IDLE_VIDEO_PATH = 'file:///home/adverto/Desktop/aircursorr/wave_your_hands_illustration.mp4'
ACTIVE_URL = 'https://kappa.hyperstate.tech/viewer/d97f0f51-dacc-4261-a014-020969a5f863/8fce99ec-1e7d-47b6-a90b-e7eb00e9fc45?autoplay=true'  

# --- Browser Setup ---
chrome_options = Options()
chrome_options.add_argument("--kiosk") 
chrome_options.add_argument("--no-sandbox") 
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

service = Service('/usr/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.get(IDLE_VIDEO_PATH)

# --- MediaPipe & PyAutoGUI Setup ---
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

# --- Tracking & FPS Variables ---
pTime = 0
prev_x, prev_y = 0, 0
prev_x_raw, prev_y_raw = 0, 0
smoothing = 7 

# Dwell Click Variables
dwell_start_time = None
dwell_duration = 2.0     
movement_threshold = 10    
click_cooldown = 1.0       
last_click_time = 0

# State Management
active_mode = False
wave_history = []
last_hand_seen_time = time.time()

def check_for_wave(palm_x):
    global wave_history
    current_time = time.time()
    wave_history.append((palm_x, current_time))
    wave_history = [h for h in wave_history if current_time - h[1] < 1.0]
    
    if len(wave_history) > 15:
        x_coords = [h[0] for h in wave_history]
        return (max(x_coords) - min(x_coords)) > 100
    return False

# --- Main Logic Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # --- FPS Calculation ---
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    if not active_mode:
        # IDLE STATE
        if result.multi_hand_landmarks:
            palm = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
            if check_for_wave(int(palm.x * frame_w)):
                print("Wave detected! Opening project URL...")
                driver.get(ACTIVE_URL)
                time.sleep(1) # Brief pause for browser transition
                active_mode = True
                last_hand_seen_time = time.time()
    else:
        # ACTIVE STATE
        if result.multi_hand_landmarks:
            last_hand_seen_time = time.time()
            
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                ix, iy = int(index_finger.x * frame_w), int(index_finger.y * frame_h)

                screen_x = np.interp(ix, [50, frame_w - 50], [0, screen_w])
                screen_y = np.interp(iy, [50, frame_h - 50], [0, screen_h])
                curr_x = prev_x + (screen_x - prev_x) / smoothing
                curr_y = prev_y + (screen_y - prev_y) / smoothing
                
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Dwell Logic
                dist_moved = np.hypot(ix - prev_x_raw, iy - prev_y_raw)
                if dist_moved < movement_threshold:
                    if dwell_start_time is None: dwell_start_time = time.time()
                    elapsed = time.time() - dwell_start_time
                    
                    # Progress circle feedback
                    angle = int((elapsed / dwell_duration) * 360)
                    cv2.ellipse(frame, (ix, iy), (30, 30), 0, 0, angle, (0, 255, 255), 3)

                    if elapsed >= dwell_duration:
                        if time.time() - last_click_time > click_cooldown:
                            pyautogui.click()
                            last_click_time = time.time()
                            dwell_start_time = None 
                else:
                    dwell_start_time = None
                
                prev_x_raw, prev_y_raw = ix, iy

        if time.time() - last_hand_seen_time > 8:
            driver.get(IDLE_VIDEO_PATH)
            active_mode = False

    # Display FPS and Mode on the Debug window
    mode_text = "ACTIVE" if active_mode else "IDLE"
    cv2.putText(frame, f'FPS: {int(fps)} | MODE: {mode_text}', (20, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Kiosk Debug Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
driver.quit()
cv2.destroyAllWindows()