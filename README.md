# 🖐️ Python OpenCV Hand-Tracking Portal Filters

A real-time computer vision project built with Python, OpenCV, and MediaPipe to create interactive visual "portals" controlled by hand gestures. Pinch your fingers to dynamically cycle between live camera filters!

---

## ⚡ Features

* **Gesture Control:** Detects finger pinch gestures in real time using MediaPipe.
* **Interactive Portal Filters:** Pixelate, Thermal, Glitch, and Matrix.
* **Dynamic Overlay:** Smoothly resizes and updates portal coordinates based on hand positions.
* **Keybind Actions:** Hotkeys for quick filter toggling, recording status, and exiting.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`opencv-python`)
* **Hand Tracking:** MediaPipe (`mediapipe`)
* **Array Math:** NumPy (`numpy`)

---

## 📦 Getting Started

Run the following commands in your terminal to clone the repository, install dependencies, and start the application:

```bash
git clone [https://github.com/YOUR-USERNAME/hand-tracking-portal.git](https://github.com/YOUR-USERNAME/hand-tracking-portal.git)
cd hand-tracking-portal
pip install opencv-python mediapipe numpy
python main.py

🎮 Controls

    Pinch (Thumb + Index): Cycle Active Filter
    r: Toggle Recording Status
    q: Quit Application

📄 License
Distributed under the MIT License. See LICENSE for more information.
