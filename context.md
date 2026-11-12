# Intelligent Real-Time AI Surveillance & Behavioral Threat Detection System
## System Architecture, Current Implementation Status & Roadmap

---

## 1. Project Overview & Core Vision

The **Intelligent Real-Time AI Surveillance & Behavioral Threat Detection System** is an autonomous, vision-based security solution designed to convert commodity webcams and CCTV hardware into proactive threat-monitoring systems. 

Unlike traditional passive recording systems that require manual human review after a security incident has occurred, this system autonomously:
1. **Tracks Human Targets**: Detects and tracks individuals in real time using YOLOv8 multi-object tracking.
2. **Enforces Virtual Tripwire Boundaries (ROI)**: Allows security operators to interactively draw and adjust dynamic tripwire zones on screen using mouse gestures.
3. **Verifies Biometric Authorization**: Employs deep facial landmark detection (YuNet) and deep facial feature embeddings (Facenet 512-D) to differentiate verified staff from unauthorized intruders and obscured targets (e.g., individuals with covered faces, masks, or turned heads).
4. **Detects Behavioral & Safety Anomalies**:
   - **Intrusion Breaches**: Instant flagging and snapshot capture for unauthorized entry.
   - **Suspicious Loitering**: Dwell-time monitoring with automated escalation when unauthorized individuals linger in restricted zones.
   - **Fall & Collapse Detection**: Bounding box aspect-ratio deformation and vertical displacement analysis for rapid safety/medical alerts.
5. **Maintains Incident Records**: Automatically captures timestamped, full-resolution security snapshots to `./outputs/` and maintains an interactive, real-time security incident log.

---

## 2. System Architecture & Pipeline Flow

```
                     +----------------------------------------+
                     |         Live Camera Feed (0)           |
                     +----------------------------------------+
                                         |
                                         v
                     +----------------------------------------+
                     |       YOLOv8 Multi-Object Tracker      |
                     |         (Persist ID, Class 0)          |
                     +----------------------------------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
                       v                                   v
             [Target Outside ROI]                 [Target Inside ROI]
                       |                                   |
                       |                     +-------------+-------------+
                       |                     | Face Verification Worker  |
                       |                     |   (Background Async)      |
                       |                     +-------------+-------------+
                       |                                   |
                       |                        [YuNet Landmark Check]
                       |                                   |
                       |                      +------------+------------+
                       |                      |                         |
                       |                 [Face Found]             [No Face / Covered]
                       |                      |                         |
                       |             [Facenet Embedding]          [Mark OBSCURED]
                       |              (Cosine <= 0.40)                  |
                       |                      |                         |
                       |            +---------+---------+               |
                       |            |                   |               |
                       |      [Match Auth]       [No Match]             |
                       |            |                   |               |
                       v            v                   v               v
                [Outside Zone]  [AUTHORIZED]       [INTRUDER]       [INTRUDER]
                   (Yellow)       (Green)            (Red)            (Red)
                                    |                   |               |
                                    v                   +-------+-------+
                            [Fall Anomaly?]                     |
                                    |                           v
                          +---------+---------+        [Instant Breach Snapshot]
                          |                   |        [Loitering Timer Active]
                     [Normal Staff]    [Medical Fall]
                        (Green)           (Amber)
```

---

## 3. Present Components & File Organization

```text
Computer-Vision-Project/
│
├── surveillance.py            # Master surveillance application & real-time GUI
├── register_face.py           # Personnel face enrollment studio GUI
├── context.md                 # System architecture, state & roadmap documentation
├── README.md                  # Quick-start instructions and user guide
├── requirements.txt           # Project dependencies
├── yolov8n.pt                 # Ultralytics YOLOv8 nano model weights
│
├── known_faces/               # Authorized personnel biometric database
│   ├── vivek/                 # Multi-angle verified samples of Vivek
│   ├── Susan/                 # Multi-angle verified samples of Susan
│   └── ds_model_facenet_...   # Pre-computed YuNet + Facenet biometric datastore (.npz)
│
└── outputs/                   # Auto-saved breach alert snapshots and incident records
    ├── ALERT_INTRUDER_...     # Intrusion and face-obscured event snapshots
    └── ALERT_FALL_...         # Medical/fall anomaly incident snapshots
```

---

## 4. Current Working Condition & Capabilities

### A. Live Surveillance & Threat Monitor (`surveillance.py`)
- **Operating Mode**: 100% dedicated live camera processing (Device 0). Pre-recorded video options and mock media fallbacks have been removed for an uncluttered, real-time security workflow.
- **Display Pipeline**:
  - Proportional letterboxing dynamically scales the feed to the canvas while strictly preserving native 4:3 or 16:9 aspect ratios.
  - Drag-and-drop interactive tripwire tool maps canvas mouse coordinates directly to native camera pixels.
  - Zero-lag Tkinter threading using `queue.Queue` delivers 50–60 FPS video throughput without UI freezes on Python 3.14 / Windows.
  - Dedicated **Standby Screen** clears the frozen last frame when monitoring is stopped or before camera initialization.
- **Biometric Recognition Engine**:
  - Uses **YuNet** (`face_detection_yunet_2023mar.onnx`) for human facial landmark extraction and alignment.
  - Uses **Facenet** (512-dimensional feature space) for cosine similarity matching against `known_faces/`.
  - **Asynchronous worker thread**: Recognizes personnel in the background without dropping stream framerate.
  - **Covered Face Rejection**: If an individual covers their face with a T-shirt, mask, or hand, YuNet detects zero facial landmarks, immediately tagging them as `FACE OBSCURED / INTRUDER` in red.
  - **Different Person Discrimination**: Different faces yield cosine distances $> 0.40$ and are locked as `UNAUTHORIZED INTRUDER`.
- **Behavioral & Safety Anomaly Classification**:
  - **Authorized Staff**: Displayed in permanent **Emerald Green** (`AUTHORIZED: <Name>`). They are never falsely penalized for normal loitering.
  - **Unauthorized Intruders**: Displayed in **Crimson Red** immediately upon entering the tripwire zone with live dwell time displayed on-screen (`UNAUTHORIZED INTRUDER (X.Xs)`).
  - **Immediate Breach Snapshot**: Captures full-resolution security snapshots to `outputs/` the instant an unauthorized or obscured person breaches the tripwire.
  - **Loitering Escalation**: Flags `SUSPICIOUS LOITERING` if an intruder stays beyond `LOITERING_THRESHOLD_SEC` (4.0s).
  - **Fall / Distress Detection**: Calibrated with `FALL_ASPECT_RATIO_THRESHOLD = 1.65`, vertical grounding filter (`by2 > 0.65 * frame_h`), and minimum bounding area filter ($>8000\text{ px}$) to prevent false fall alerts when users sit close to their desk webcam.

### B. Personnel Face Enrollment Studio (`register_face.py`)
- **Interface**: Unified Slate 900/800 design system matching `surveillance.py`.
- **Manual Shutter Capture**: Users can capture photos at their own pace across multiple angles, tilts, and expressions.
- **Case-Sensitive Folder Persistence**: Appends new photos to existing personnel folders (e.g., `known_faces/vivek/vivek_1.jpg`, `vivek_2.jpg`, etc.).
- **Enrollment Validation**: Integrated YuNet landmark validation checks each frame before saving. It refuses to enroll blank frames or covered faces.
- **Database Management**: Live directory listing, photo counters, deletion, and auto-invalidation of stale representation caches (`.pkl` and `.npz`).
- **Safe Exit**: Includes a dedicated `✕ Exit Studio` button to cleanly release the camera handle.

---

## 5. What Was Expected vs. What Was Discovered (Lessons Learned)

| Issue Discovered | Root Cause Identified | Resolution Implemented |
| :--- | :--- | :--- |
| **OpenCV `CascadeClassifier` Crash** | OpenCV 5.0.0 on Python 3.14 omits Haar Cascades, causing default DeepFace calls to fail silently. | Upgraded detector backend to **YuNet ONNX**, completely eliminating cascade dependencies. |
| **Covered Face Tagged as "Vivek"** | 1. `detector_backend="skip"` passed raw room pixels to Facenet without face detection.<br>2. 3 non-face photos (`vivek_6.jpg`, `vivek_38.jpg`, `vivek_39.jpg`) were enrolled in `known_faces/vivek/`. | 1. Switched to YuNet with `enforce_detection=True`.<br>2. Purged non-face photos from `known_faces/`.<br>3. Added face verification check to `register_face.py`. |
| **Sitting at Desk Flagged as "Fall"** | Close-up webcam perspective produces an upper-body aspect ratio of $\sim 1.33$, exceeding the previous $1.25$ threshold. | Raised threshold to $1.65$, added vertical grounding filter ($by_2 > 0.65 \times H$), and minimum bounding area check. |
| **Camera Feed Aspect Ratio Distortion** | Canvas previously resized all video streams to a fixed $760 \times 480$, squishing 4:3 and 16:9 webcam inputs. | Implemented proportional scaling and dynamic letterboxing with coordinate remapping. |
| **Thread Crashes on Stop & MSMF Spam** | Calling `cap.release()` while worker thread was inside `cap.read()` caused MSMF driver warnings and `NoneType.set()` errors. | Added `video_thread.join(timeout=0.3)` and removed legacy video seek commands. |

---

## 6. What Is Yet to Be Done (Roadmap & Next Steps)

### Phase 1: Near-Term Enhancements (Model & Detection Robustness)
- [ ] **Multi-Camera Support**: Add support for switching between multiple USB/IP cameras or RTSP video streams.
- [ ] **Temporal Face Verification Smoothing**: Run face verification across 3 consecutive frames with a voting buffer (e.g., 2 out of 3 matches required) to eliminate single-frame lighting or motion blur anomalies.
- [ ] **Tripwire Exclusion Zones**: Allow operators to define "Ignore Zones" (e.g., windows, ceiling fans, or computer monitors) to avoid tracking irrelevant background motion.
- [ ] **Pose-Based Fall Detection**: Upgrade from bounding box aspect-ratio heuristics to **YOLOv8-Pose** (keypoint detection: head, shoulders, hips, knees, ankles) for medical-grade fall classification regardless of camera angle.

### Phase 2: Notification & Remote Alerting
- [ ] **Audio/Siren Alarm**: Optional audible beep or audio alert when an unauthorized intrusion occurs.
- [ ] **Telegram / Email / Webhook Alerts**: Automatically dispatch instant breach snapshots and incident descriptions to security personnel via Telegram Bot or SMTP email.
- [ ] **Headless Edge Server Mode**: Provide a CLI-only daemon runner (e.g., `python surveillance_headless.py --camera 0 --alert-webhook <url>`) for deployment on headless Linux edge servers or NVIDIA Jetson devices.

### Phase 3: Analytics & Database Scaling
- [ ] **Vector Database Indexing (FAISS / ChromaDB)**: If personnel database grows beyond 500+ individuals, integrate FAISS for sub-millisecond vector similarity search.
- [ ] **Searchable Incident Log**: Add date/time and category filters (Intruder, Loitering, Fall) to the GUI incident log.
- [ ] **Snapshot Auto-Pruning**: Implement a retention policy to automatically archive or delete snapshots older than $N$ days to manage disk space.

---

## 7. Key Technology Stack

- **Language & Runtime**: Python 3.14 (AMD64)
- **Computer Vision & Processing**: OpenCV (`opencv-python 5.0.0.93`), NumPy, Pillow (PIL)
- **Object Detection & Tracking**: Ultralytics YOLOv8 (`yolov8n.pt`, PyTorch)
- **Biometric Landmark Detection**: OpenCV YuNet ONNX (`face_detection_yunet_2023mar.onnx`)
- **Facial Feature Representation**: DeepFace with Facenet (512-dimensional biometric embeddings)
- **Graphical User Interface**: Python Tkinter with custom ttk styling and multi-threaded `queue.Queue` event dispatchers

