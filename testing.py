import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# --- Configuration ---
IDLE_VIDEO_PATH = 'file:///home/adverto/Desktop/aircursorr/index.html'
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

def inject_cursor_js(driver):
    script = """
    if (!document.getElementById('cursorProgressCanvas')) {
        var canvas = document.createElement('canvas');
        canvas.id = 'cursorProgressCanvas';
        canvas.width = 100;
        canvas.height = 100;
        canvas.style.position = 'fixed';
        canvas.style.pointerEvents = 'none';
        canvas.style.zIndex = '10000';
        document.body.appendChild(canvas);
        
        window.drawProgress = function(x, y, progress) {
            requestAnimationFrame(() => {
                var ctx = canvas.getContext('2d');
                canvas.style.left = (x - 50) + 'px';
                canvas.style.top = (y - 50) + 'px';
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (progress > 0) {
                    ctx.beginPath();
                    ctx.arc(50, 50, 25, 0, 2 * Math.PI * progress);
                    ctx.strokeStyle = '#00FFFF';
                    ctx.lineWidth = 6;
                    ctx.stroke();
                }
            });
        };
    }
    """
    try: driver.execute_script(script)
    except: pass

# --- MediaPipe Setup ---
pyautogui.FAILSAFE, pyautogui.PAUSE = False, 0 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.8,
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
smoothing = 5 
dwell_start_time = None
dwell_duration = 2.0     
movement_threshold = 7   
click_cooldown, last_click_time = 1.0, 0
active_mode = False
wave_history = []
last_hand_seen_time = time.time()
ui_injected = False
last_sent_progress = -1

# --- Optimization Variable ---
frame_counter = 0

def check_for_wave(palm_x):
    global wave_history
    curr_time = time.time()
    wave_history.append((palm_x, curr_time))
    wave_history = [h for h in wave_history if curr_time - h[1] < 1.0]
    
    if len(wave_history) > 10:
        x_coords = [h[0] for h in wave_history]
        total_range = max(x_coords) - min(x_coords)
        return total_range > 120 
    return False

# --- Main Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_counter += 1
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Only process AI and Browser UI every 2nd frame to save CPU
    if frame_counter % 2 == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if not active_mode:
            ui_injected = False 
            if result.multi_hand_landmarks:
                palm_x = int(result.multi_hand_landmarks[0].landmark[0].x * frame_w)
                if check_for_wave(palm_x):
                    driver.get(ACTIVE_URL)
                    active_mode = True
                    last_hand_seen_time = time.time()
                    wave_history = [] 
        else:
            if not ui_injected:
                time.sleep(1) 
                inject_cursor_js(driver)
                ui_injected = True

            if result.multi_hand_landmarks:
                last_hand_seen_time = time.time()
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Cursor movement
                index_finger = hand_landmarks.landmark[8]
                ix, iy = int(index_finger.x * frame_w), int(index_finger.y * frame_h)
                curr_x = prev_x + (np.interp(ix, [50, frame_w-50], [0, screen_w]) - prev_x) / smoothing
                curr_y = prev_y + (np.interp(iy, [50, frame_h-50], [0, screen_h]) - prev_y) / smoothing
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Dwell Logic
                dist_moved = np.hypot(ix - prev_x_raw, iy - prev_y_raw)
                current_progress = 0
                if dist_moved < movement_threshold:
                    if dwell_start_time is None: dwell_start_time = time.time()
                    current_progress = min((time.time() - dwell_start_time) / dwell_duration, 1.0)
                    if current_progress >= 1.0 and (time.time() - last_click_time > click_cooldown):
                        pyautogui.click()
                        last_click_time, dwell_start_time = time.time(), None 
                else:
                    dwell_start_time = None
                
                # Update UI only if changed significantly
                if abs(current_progress - last_sent_progress) > 0.05:
                    try: driver.execute_script(f"if(window.drawProgress) window.drawProgress({curr_x}, {curr_y}, {current_progress});")
                    except: pass
                    last_sent_progress = current_progress
                    
                prev_x_raw, prev_y_raw = ix, iy
            else:
                if last_sent_progress != 0:
                    try: driver.execute_script("if(window.drawProgress) window.drawProgress(0, 0, 0);")
                    except: pass
                    last_sent_progress = 0

            if time.time() - last_hand_seen_time > 10:
                driver.get(IDLE_VIDEO_PATH)
                active_mode = False

    # --- FPS CALCULATION ---
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # --- DISPLAY FPS AND STATUS ---
    mode_text = "ACTIVE" if active_mode else "IDLE"
    cv2.putText(frame, f'FPS: {int(fps)} | MODE: {mode_text}', (20, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Kiosk Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
driver.quit()
cv2.destroyAllWindows()