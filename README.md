# FaceWire: Real-Time Identity-Aware Zone Intrusion and Fall Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8--Pose-orange.svg)](https://github.com/ultralytics/ultralytics)
[![DeepFace](https://img.shields.io/badge/DeepFace-Facenet512-red.svg)](https://github.com/serengil/deepface)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![Hardware](https://img.shields.io/badge/Execution-CPU%20Only-informational.svg)]()

**FaceWire** is a surveillance application that runs fully on a local computer, with no GPU and no cloud. It combines three things in one window: a virtual tripwire you draw on the video, face verification to tell authorized people from strangers, and skeleton-based fall detection.

![FaceWire dashboard showing two authorized people and one flagged intruder](assets/screenshot1.png)

*The live dashboard with three people in the restricted zone. Tracks 7 and 8 are enrolled and shown as authorized (green). Track 9 is not authorized and is shown in red with a loitering alert. The right panel shows FPS, people in the zone, breach count, enrolled personnel, and the incident feed.*

---

## What It Does

* **Detection and tracking**: YOLOv8-Pose nano finds each person and their 17 body keypoints in one pass. ByteTrack gives every person a stable ID across frames.
* **Drawable tripwire zone**: Drag on the video to draw a zone with any number of corners. A person is "in the zone" if either their feet point or the center of their box is inside it.
* **Face verification**: Faces are found with OpenCV YuNet and turned into a 512-value signature by Facenet512. A person is authorized only after a majority vote over several frames, so one lucky frame cannot let a stranger through.
* **Fall detection without training**: The angle of the shoulder-to-hip line is measured against vertical. A tilt above 60 degrees that lasts one second raises a fall alert.
* **Behavior alerts**: Loitering, running, overcrowding, and optional after-hours presence.
* **Automatic incident records**: Every alert is written to a CSV file and saved as an annotated JPEG, without slowing down the video.
* **CPU-friendly**: 14.8 FPS in the benchmark run on an Intel Core i7. Speed drops when several people are in the zone at once (the screenshot above shows 5.8 FPS with three people).

---

## How It Works

```
                 VIDEO INPUT (webcam 0 or video file)
                               |
                               v
              Resize to 640 x 640, keeping aspect ratio
                               |
                               v
                YOLOv8-Pose nano (one pass per frame)
                  - person boxes (confidence 0.40+)
                  - 17 body keypoints
                               |
                               v
                ByteTrack: stable ID for each person
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
        Zone check                      Posture and motion
  (feet point or box center       - torso tilt from vertical
   inside the drawn polygon?)     - sustained tilt for 1 second
              |                   - box-shape backup rule
              v                   - speed in body heights/second
     Face verification                          |
  - upper 55% of the box                        |
  - YuNet finds face + landmarks                |
  - Facenet512 face signature                   |
  - compare with enrolled faces                 |
  - vote: 2 of the last 3 samples               |
              |                                 |
      +-------+-------+                         |
      v               v                         |
 AUTHORIZED        INTRUDER                     |
      |               |                         |
      +---------------+-------------------------+
                      |
                      v
            Alert manager (background thread)
              - incidents.csv log
              - annotated JPEG snapshot
              - live incident panel in the GUI
```

Each person in the zone goes through these states: **Outside**, **Scanning** (2.5-second grace period), then **Authorized**, **Intruder**, or **Obscured** (face could not be found). Authorized people are re-checked every 10 seconds.

---

## Project Structure

```
Computer-Vision-Project/
├── main.py              # Whole application: surveillance engine, GUI, enrollment studio
├── yolov8n-pose.pt      # YOLOv8-Pose nano weights
├── known_faces/         # Enrolled people (one sub-folder per person)
├── cache/               # Saved face signatures for fast startup
│   ├── embeddings.npz
│   └── manifest.json
├── outputs/             # Incident log and snapshots
│   ├── incidents.csv
│   └── alert_*.jpg
├── assets/
│   └── screenshot1.png  # Dashboard screenshot used in this README and the report
├── _archive/            # Earlier separate modules from development
├── requirements.txt
├── report.tex           # Project report
└── README.md
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/vivekvardhan28/Computer-Vision-Project.git
cd Computer-Vision-Project

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
python main.py
```

* **Start surveillance**: starts the default camera.
* **Draw the zone**: click and drag on the video. **Reset zone** restores the default zone.
* **Send test**: creates a test alert so you can check logging and snapshots.
* **Open CSV log / Open snapshots**: opens the files in the `outputs/` folder.
* **Clear**: empties the incident panel on screen.

### 3. Enroll people

Use **Enroll faces** in the dashboard, or open the enrollment window directly:

```bash
python main.py --enroll
```

1. Enter the person's name.
2. Follow the on-screen poses: front, slightly left, slightly right, slightly up, slightly down, natural smile, and a neutral face (with or without glasses).
3. Each photo is checked automatically and rejected if it is not good enough:
   * face confidence of at least 0.90
   * face width of at least 80 pixels
   * sharpness score of at least 60
   * not a near-copy of a photo already taken for that person
4. Face signatures are saved to `cache/embeddings.npz`, so the next start is quick. Use **Reload database** if you add photos by hand.

---

## Settings

All settings are at the top of `main.py`.

| Setting | Default | Meaning |
| :--- | :--- | :--- |
| `YOLO_CONF` | 0.40 | Minimum confidence to accept a person detection |
| `ROI_MODE` | `"any"` | Zone test uses the feet point, the box center, or either one |
| `GRACE_PERIOD_SEC` | 2.5 s | Time given to identify a person before they are marked as an intruder |
| `VOTE_WINDOW` / `VOTES_REQUIRED` | 3 / 2 | A person needs 2 matches among the last 3 face samples |
| `FACE_DISTANCE_THRESH` | 0.35 | Largest face distance still counted as a match (smaller means more similar) |
| `FACE_MATCH_MARGIN` | 0.05 | How much better the best match must be than the second best |
| `FACE_TOPK` | 3 | Number of closest stored faces averaged per person |
| `MIN_FACE_PX` | 48 | Smallest face width accepted during surveillance |
| `REAUTH_INTERVAL_SEC` | 10 s | How often authorized people are checked again |
| `FALL_TORSO_ANGLE_DEG` | 60 degrees | Torso tilt from vertical that counts as a fall posture |
| `FALL_SUSTAIN_SEC` | 1.0 s | How long the fall posture must last before an alert |
| `FALL_ASPECT_RATIO_THRESHOLD` | 1.65 | Backup rule: box width divided by height when keypoints are missing |
| `RUN_SPEED_BODYHEIGHTS_PER_S` | 1.5 | Speed that counts as running |
| `LOITERING_THRESHOLD_SEC` | 4.0 s | Time an unauthorized person can stay in the zone before a loitering alert |
| `MAX_PEOPLE_IN_ROI` | 3 | People in the zone before an overcrowding alert |
| `ALERT_COOLDOWN_SEC` | 10 s | Minimum gap between repeats of the same alert for the same person |
| `SNAPSHOT_RETENTION_DAYS` | 30 | Snapshots older than this are deleted automatically |

You can also override settings in a `config.json` file next to `main.py`.

---

## Output Records

Every alert is saved in the `outputs/` folder.

1. **`incidents.csv`** has these columns: `timestamp_iso`, `event_type`, `track_id`, `name`, `state`, `distance`, `dwell_sec`, `message`, `snapshot_path`.
2. **`alert_*.jpg`** snapshots show the full frame with the zone outline, the person's box, and a status label.

---

## System Requirements

* **OS**: Windows 10/11, Ubuntu 20.04 or newer, or macOS.
* **Processor**: Multi-core Intel Core i5/i7 (8th generation or newer) or AMD Ryzen 5/7.
* **Memory**: 8 GB minimum, 16 GB recommended.
* **GPU**: Not needed.
* **Speed**: 14.8 FPS in the benchmark run on an Intel Core i7 at 640 x 640 input. Expect lower speed with more people in the zone.

---

## Known Limitations

* A photo or screen shown to the camera can fool the face check, because there is no liveness test yet.
* The face check can mark a visible face as "Face covered" when it cannot be read clearly (for example, track 9 in the screenshot above).
* When keypoints are hidden, fall detection uses a less precise rule based on the box shape.
* Speed drops when several people are in the zone at once.

---

## Team

| Member | Roll No. | Area | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Guna** | `CS24B2043` | Face verification | Upper-body face crop, YuNet landmarks, Facenet512 signatures, best-three-match scoring, and the voting rules. |
| **Susan** | `CS23B2026` | Motion and posture | Center-point tracking over time, speed in body heights per second, torso tilt from the 17 keypoints, and the fall duration rule. |
| **Vivek** | `CS23B1043` | Tracking and zone geometry | Aspect-preserving frame mapping, YOLOv8-Pose detection, ByteTrack association, and the polygon zone test. |
