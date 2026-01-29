import cv2
import mediapipe as mp
import numpy as np
import time
from pynput.mouse import Button, Controller
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# --- Configuration ---
IDLE_VIDEO_PATH = 'file:///home/adverto/Desktop/aircursorr/index.html'
ACTIVE_URL = 'https://kappa.hyperstate.tech/viewer/d97f0f51-dacc-4261-a014-020969a5f863/8fce99ec-1e7d-47b6-a90b-e7eb00e9fc45?autoplay=true'  

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
            var ctx = canvas.getContext('2d');
            canvas.style.left = (x - 50) + 'px';
            canvas.style.top = (y - 50) + 'px';
            
            // 1. Detect element under cursor
            var el = document.elementFromPoint(x, y);
            var isInteractable = false;
            
            if (el) {
                // Check the element or its parent (useful for icons inside buttons)
                var check = (node) => {
                    if(!node) return false;
                    var s = window.getComputedStyle(node);
                    return (node.tagName === 'BUTTON' || node.tagName === 'A' || 
                            node.tagName === 'INPUT' || s.cursor === 'pointer');
                };
                isInteractable = check(el) || check(el.parentElement);
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // 2. Always draw a small static circle so we know it's working
            ctx.beginPath();
            ctx.arc(50, 50, 5, 0, 2 * Math.PI);
            ctx.fillStyle = isInteractable ? '#00FFFF' : 'rgba(255, 255, 255, 0.3)';
            ctx.fill();

            // 3. Only draw the loading progress if interactable
            if (isInteractable && progress > 0) {
                ctx.beginPath();
                ctx.arc(50, 50, 25, 0, 2 * Math.PI * progress);
                ctx.strokeStyle = '#00FFFF';
                ctx.lineWidth = 6;
                ctx.stroke();
            }
            return isInteractable; 
        };
    }
    """
    try: driver.execute_script(script)
    except: pass

# --- Controls Setup ---
mouse = Controller()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.8)

# Get Screen Resolution automatically
import pyautogui
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Variables ---
pTime, prev_x, prev_y, prev_raw_x, prev_raw_y = 0, 0, 0, 0, 0
alpha = 0.25 
dwell_start_time = None
dwell_duration = 2.0    
movement_threshold = 8   
click_cooldown, last_click_time = 1.0, 0
active_mode = False
wave_history = []
last_hand_seen_time = time.time()
on_clickable = False

def check_for_wave(palm_x):
    global wave_history
    curr_time = time.time()
    wave_history.append((palm_x, curr_time))
    wave_history = [h for h in wave_history if curr_time - h[1] < 1.0]
    if len(wave_history) > 10:
        x_coords = [h[0] for h in wave_history]
        return (max(x_coords) - min(x_coords)) > 120 
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if not active_mode:
        if result.multi_hand_landmarks:
            palm_x = int(result.multi_hand_landmarks[0].landmark[0].x * frame_w)
            if check_for_wave(palm_x):
                driver.get(ACTIVE_URL)
                active_mode = True
                last_hand_seen_time = time.time()
    else:
        inject_cursor_js(driver)
        if result.multi_hand_landmarks:
            last_hand_seen_time = time.time()
            index_finger = result.multi_hand_landmarks[0].landmark[8]
            ix, iy = index_finger.x * frame_w, index_finger.y * frame_h
            
            # MOVEMENT
            target_x = np.interp(ix, [50, frame_w-50], [0, screen_w])
            target_y = np.interp(iy, [50, frame_h-50], [0, screen_h])
            curr_x = (target_x * alpha) + (prev_x * (1 - alpha))
            curr_y = (target_y * alpha) + (prev_y * (1 - alpha))
            mouse.position = (int(curr_x), int(curr_y))
            
            # DWELL
            dist_moved = np.hypot(ix - prev_raw_x, iy - prev_raw_y)
            current_progress = 0
            
            if dist_moved < movement_threshold and on_clickable:
                if dwell_start_time is None: dwell_start_time = time.time()
                current_progress = min((time.time() - dwell_start_time) / dwell_duration, 1.0)
                if current_progress >= 1.0 and (time.time() - last_click_time > click_cooldown):
                    mouse.click(Button.left, 1)
                    last_click_time, dwell_start_time = time.time(), None 
            else:
                dwell_start_time = None

            # JS SYNC
            try: 
                # We pass int coordinates to ensure JS doesn't break
                on_clickable = driver.execute_script(
                    f"return window.drawProgress ? window.drawProgress({int(curr_x)}, {int(curr_y)}, {current_progress}) : false;"
                )
            except: 
                on_clickable = False
                
            prev_x, prev_y = curr_x, curr_y
            prev_raw_x, prev_raw_y = ix, iy
        else:
            try: driver.execute_script("if(window.drawProgress) window.drawProgress(0, 0, 0);")
            except: pass

        if time.time() - last_hand_seen_time > 15:
            driver.get(IDLE_VIDEO_PATH)
            active_mode = False

    cv2.imshow("Kiosk Debug", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
driver.quit()
cv2.destroyAllWindows()