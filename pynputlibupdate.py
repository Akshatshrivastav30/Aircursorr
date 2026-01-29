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

# --- Browser Setup ---
chrome_options = Options()
chrome_options.add_argument("--kiosk") 
chrome_options.add_argument("--no-sandbox") 
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

# Ensure the path to your chromedriver is correct
service = Service('/usr/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.get(IDLE_VIDEO_PATH)

def inject_cursor_js(driver):
    """Injects the visual feedback circle into the browser DOM."""
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

# --- Controls Setup ---
mouse = Controller()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

# Replace with your actual kiosk screen resolution
screen_w, screen_h = 1920, 1080 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Variables ---
pTime = 0
prev_x, prev_y = 0, 0
prev_raw_x, prev_raw_y = 0, 0

# EMA Alpha: 0.2 is smooth but responsive. Lower = more lag/more smooth.
alpha = 0.2 

dwell_start_time = None
dwell_duration = 2    
movement_threshold = 8   
click_cooldown, last_click_time = 1.2, 0
active_mode = False
wave_history = []
last_hand_seen_time = time.time()
last_sent_progress = -1

def check_for_wave(palm_x):
    """Detects a side-to-side waving motion to 'wake up' the kiosk."""
    global wave_history
    curr_time = time.time()
    wave_history.append((palm_x, curr_time))
    wave_history = [h for h in wave_history if curr_time - h[1] < 1.0]
    if len(wave_history) > 10:
        x_coords = [h[0] for h in wave_history]
        return (max(x_coords) - min(x_coords)) > 120 
    return False

# --- Main Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    if not active_mode:
        if result.multi_hand_landmarks:
            palm_x = int(result.multi_hand_landmarks[0].landmark[0].x * frame_w)
            if check_for_wave(palm_x):
                driver.get(ACTIVE_URL)
                active_mode = True
                last_hand_seen_time = time.time()
    else:
        # Maintenance: Keep JS injected in case of page navigation
        inject_cursor_js(driver)

        if result.multi_hand_landmarks:
            last_hand_seen_time = time.time()
            hand_landmarks = result.multi_hand_landmarks[0]
            index_finger = hand_landmarks.landmark[8]
            
            # 1. Capture Raw Coordinates
            ix, iy = index_finger.x * frame_w, index_finger.y * frame_h
            
            # 2. Dwell Timer Logic
            dist_moved = np.hypot(ix - prev_raw_x, iy - prev_raw_y)
            current_progress = 0
            
            if dist_moved < movement_threshold:
                if dwell_start_time is None: dwell_start_time = time.time()
                current_progress = min((time.time() - dwell_start_time) / dwell_duration, 1.0)
                
                # Execute Click
                if current_progress >= 1.0 and (time.time() - last_click_time > click_cooldown):
                    mouse.click(Button.left, 1)
                    last_click_time, dwell_start_time = time.time(), None 
            else:
                dwell_start_time = None

            # 3. Movement with EMA Smoothing (No Freeze)
            target_x = np.interp(ix, [50, frame_w-50], [0, screen_w])
            target_y = np.interp(iy, [50, frame_h-50], [0, screen_h])
            
            # Calculation: (New * Alpha) + (Old * (1 - Alpha))
            curr_x = (target_x * alpha) + (prev_x * (1 - alpha))
            curr_y = (target_y * alpha) + (prev_y * (1 - alpha))
            
            mouse.position = (curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            
            # 4. Sync Browser Feedback
            if abs(current_progress - last_sent_progress) > 0.02:
                try: 
                    driver.execute_script(f"if(window.drawProgress) window.drawProgress({prev_x}, {prev_y}, {current_progress});")
                except: 
                    pass
                last_sent_progress = current_progress
                
            prev_raw_x, prev_raw_y = ix, iy
        else:
            # If hand is lost, reset cursor progress bar
            if last_sent_progress != 0:
                try: driver.execute_script("if(window.drawProgress) window.drawProgress(0, 0, 0);")
                except: pass
                last_sent_progress = 0

        # Return to IDLE mode after 10 seconds of no hand detection
        if time.time() - last_hand_seen_time > 8:
            driver.get(IDLE_VIDEO_PATH)
            active_mode = False

    # Visual Feedback for Debugging
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Kiosk Debug Window", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
driver.quit()
cv2.destroyAllWindows()