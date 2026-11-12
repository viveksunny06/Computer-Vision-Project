# Intelligent Real-Time AI Surveillance & Threat Detection System

A computer vision system for real-time live surveillance featuring interactive virtual tripwires, multi-pose face authorization, immediate intrusion alerting, and fall/distress detection.

## Features

- **Live Camera Stream** — Streamlined, dedicated live camera processing (Device 0) with zero aspect ratio distortion or stretching.
- **Interactive Tripwire Zone (ROI)** — Draw and customize your active surveillance area directly on screen by clicking and dragging with your mouse.
- **Multi-Pose Face Authorization** — DeepFace + Facenet facial matching against `known_faces/<Person_Name>/`.
  - **Authorized Personnel**: Permanently highlighted in **Emerald Green** (`AUTHORIZED: <Name>`).
  - **Unauthorized Intruders**: Highlighted in **Crimson Red** immediately upon entering the tripwire zone (`UNAUTHORIZED INTRUDER (Xs)`).
- **Behavioral & Threat Alerts**:
  - **Immediate Breach Alert**: Logs breach incidents and archives timestamped snapshots to `outputs/` the second an unauthorized target enters the tripwire.
  - **Prolonged Loitering**: Escalates alert if an intruder lingers beyond the threshold (>4.0s).
  - **Fall / Distress Detection**: Computes bounding box aspect ratio anomaly ($W/H > 1.25$) for rapid safety alerts.
- **Modern Slate Interface** — Clean dark slate design with real-time HUD, live FPS counter, active target counter, and incident logging.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Register Authorized Personnel (Face Studio)

Launch the enrollment interface to add authorized people:

```bash
python register_face.py
```
- Enter the person's name (case-sensitive folder persistence).
- Click **📸 Capture Photo** as many times as you like at your own pace across different poses/angles.
- Existing names will seamlessly append new photos to their folder.
- Click **✕ Exit Studio** when done.

### 3. Run Live Surveillance

Launch the surveillance dashboard:

```bash
python surveillance.py
```
- **Draw Tripwire**: Click and drag your mouse across the live video canvas to define your active tripwire zone.
- **Face Studio Button**: Launch `register_face.py` directly from the dashboard if you need to enroll new faces.
- Click **▶ Start Surveillance** to begin live monitoring.

---

## Project Structure

```text
Computer-Vision-Project/
├── known_faces/               # Authorized personnel database (folders per person)
├── outputs/                   # Auto-saved breach alert snapshots
├── register_face.py           # Face registration GUI studio
├── surveillance.py            # Master live surveillance application
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```
