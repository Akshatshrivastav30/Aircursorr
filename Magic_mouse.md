# Magic mouse: Gesture-Controlled Web Navigation

An AI-powered  system designed for Raspberry Pi that allows users to interact with web content using touchless hand gestures.

## 

---

### Important Note :"To ensure dependency stability and prevent version conflicts, the system is deployed within a dedicated **Python 3.11** virtual environment. Prior to execution, initialize the `airmouse` environment and then launch the main application script."

---

## 🛠 Workflow & Architecture

### 1. The Idle State (Attract Mode)

The magic mouse starts by playing a local `.mp4` video in full-screen (Kiosk Mode). This video acts as a visual prompt for users to "Wave to Start." In the background, OpenCV and MediaPipe monitor for a specific horizontal waving pattern.

### 2. The Active State (AirCursor Mode)

Once a wave is detected, the browser navigates to the target URL.

- **Hand Tracking:** Uses MediaPipe to track the index finger and wrist.

- **Smoothing:** Implements a low-pass filter (Smoothing factor: 7) to remove jitter from the cursor.

- **Dwell-Clicking:** A 2-second stability timer ($threshold = 10px$) triggers a hardware click via PyAutoGUI.

- **Visual Feedback:** A cyan progress ring is dynamically drawn around the browser's cursor via injected JavaScript.

### 3. Automatic Timeout

If no hand is detected for 10 seconds, the system automatically navigates back to the idle video, ensuring the kiosk is always ready for the next user.

---

## Technical Stack

- **Core:** Python 3.x

- **Computer Vision:** MediaPipe (Hand Landmarking), OpenCV

- **Automation:** Selenium WebDriver (Chromium), PyAutoGUI

- **UI/UX:** JavaScript (Canvas API), CSS

- **OS:** Linux (Raspberry Pi OS / Debian Trixie)

---

## Setup & Installation

### 1. Prerequisites

Ensure you are using a Virtual Environment (e.g., `airmouse`):

Bash

```
python -m venv airmouse
source airmouse/bin/activate
```

### 2. Install Dependencies

Bash

```
pip install mediapipe opencv-python pyautogui selenium numpy
sudo apt update
sudo apt install chromium-driver -y
```

### 3. File Structure

Ensure your local video is in the correct directory:

Plaintext

```
/home/adverto/Desktop/aircursorr/
├── activatedfunction2.py
└── wave_your_hands_illustration.mp4
|__index.html
```

### 4. Running the Kiosk

Bash

```
python activatedfunction2.py
```

## issue :Unlike conventional methods that rely on CPU-intensive transparent window layers—which often bottleneck video processing—this implementation achieves visual feedback through **client-side DOM manipulation**. This architectural choice ensures that the AI tracking loop remains responsive and jitter-free

## solution:## Selenium UI Injection

Standard gesture-control systems often use a **transparent overlay window** or a **virtual screen** to show feedback. These methods are heavy and often cause significant lag on low-power hardware like the Raspberry Pi.

**AirCursor** solves this by using **Selenium-specific UI Injection**:

- **No Virtual Screens:** We avoid the overhead of managing secondary transparent windows.

- **Direct DOM Injection:** We inject a lightweight JavaScript/CSS `<canvas>` directly into the webpage's Document Object Model.

- **GPU Acceleration:** By drawing the progress ring inside the browser, we leverage Chromium’s hardware acceleration, keeping the Python loop fast and responsive.  


