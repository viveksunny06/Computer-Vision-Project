# AI Surveillance & Behavioral Threat Detection System

Autonomous Edge Vision Platform — integrates real-time human tracking, interactive tripwire boundary enforcement, 512-dimensional facial biometrics, and training-free behavioral threat detection.

---

## Project Structure

```
Computer-Vision-Project/
├── main.py                  # Core unified application: dashboard, tracking, biometrics & enrollment studio
├── known_faces/             # Biometric gallery containing enrolled personnel face crops
├── cache/                   # Incremental biometric embeddings cache (instant zero-delay startup)
│   ├── embeddings.npz       # Pre-computed 512-D unit embedding matrices
│   └── manifest.json        # Cache validation manifest with file hashes & timestamps
├── outputs/                 # Incident audit records & annotated threat snapshots
│   ├── incidents.csv        # Structured CSV security audit log
│   └── alert_*.jpg          # Full-frame annotated incident captures with tripwire overlay
├── yolov8n-pose.pt          # Pretrained YOLOv8-Pose weights (17-keypoint human skeleton model)
├── requirements.txt         # Pinned Python dependencies
├── context.md               # Technical report, model benchmark analysis & state machine design
└── README.md                # System documentation and operational manual
```

**Architecture:** The entire platform executes as a **single consolidated process** (`main.py`) powered by dedicated daemon worker threads. The video acquisition thread captures live frames at native sensor rates with aspect-ratio letterboxing. Person detections and 17 COCO skeletal keypoints are tracked across frames via ByteTrack. Bounded asynchronous workers handle biometric identity verification using `Facenet512` and OpenCV `YuNet` against an in-memory normalized vector gallery. Thread-safe queues stream annotated visual overlays to a custom GitHub-dark themed Tkinter GUI while logging incidents to disk.

---

## Quick Start

### 1. Prerequisites & Installation

Clone the repository and install runtime dependencies within a Python virtual environment:

```bash
# Clone repository
git clone https://github.com/vivekvardhan28/Computer-Vision-Project.git
cd Computer-Vision-Project

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Launch Live Surveillance

Start the main surveillance workstation:

```bash
python main.py
```

* **Activate Camera:** Click **`▶ Start surveillance`** to initiate live camera capture (Device 0).
* **Adjust Tripwire ROI:** Click and drag anywhere on the live video canvas to define a custom monitoring boundary. Click **`↺ Reset tripwire`** to revert to the default centered corridor.
* **Test Incident Auditing:** Click **`Send Test Alert`** to instantly verify disk logging and snapshot capture.
* **Inspect Security Logs:** Click **`Open Log`** or **`Open Snapshots`** to review `outputs/incidents.csv` and visual evidence files.

### 3. Personnel Face Enrollment

Enroll authorized individuals directly from the running dashboard by clicking **`Enroll personnel`** (which safely pauses surveillance to release the camera), or launch the enrollment studio directly via CLI:

```bash
python main.py --enroll
```

1. Enter the person's identifier or name (e.g., `Vivek`).
2. Follow the guided pose prompts (*Front*, *Slightly left*, *Slightly right*, *Slightly up*, *Smile*, etc.).
3. Click **`Capture Photo`** to acquire samples. The background biometric gate automatically verifies that:
   * Exactly 1 face is present with YuNet detection confidence $\ge 0.90$.
   * Face width is at least 80 pixels.
   * Image sharpness (Laplacian variance) meets the clarity threshold ($\ge 60.0$).
   * Cosine similarity against recent captures is $\le 0.98$ (prevents redundant duplicate angles).
4. Closing the studio window automatically re-indexes `cache/embeddings.npz` and resumes system readiness without restarting.

---

## Tech Stack

| Layer | Technology | Details |
| :--- | :--- | :--- |
| **Object Detection & Pose** | Ultralytics YOLOv8-Pose (`yolov8n-pose.pt`) | 17 COCO human skeletal keypoints at 640px input resolution |
| **Object Tracking** | ByteTrack (`bytetrack.yaml`) | Two-stage association using high- and low-confidence detection boxes |
| **Face Detection & Landmarks** | OpenCV YuNet (`face_detection_yunet`) | 5-point facial landmark detector with strict face presence enforcement |
| **Biometric Embeddings** | DeepFace (`Facenet512`) | 512-dimensional unit-normalized facial embeddings |
| **Biometric Database** | In-Memory `FaceDB` + NumPy Matrix Ops | Top-3 cosine similarity aggregation with runner-up margin thresholding |
| **User Interface** | Python Tkinter + TTK + Pillow | GitHub Primer Dark palette (`#0d1117`), anti-aliased HUD badges, custom widgets |
| **Incident Logging** | Asynchronous `AlertManager` | Automated CSV audit trail (`outputs/incidents.csv`) and JPEG snapshots |

---

## System Operational Specifications

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `YOLO_CONF` | `0.40` | Person detection confidence floor |
| `ROI_MODE` | `"any"` | Tripwire intersection mode (evaluates bounding-box feet and center points) |
| `GRACE_PERIOD_SEC` | `2.5s` | Identification window before unverified targets are flagged as intruders |
| `VOTES_REQUIRED` | `2 of 3` | Multi-frame voting quorum required for biometric authorization |
| `FACE_DISTANCE_THRESH` | `0.35` | Maximum cosine distance threshold for genuine match ($EER = 0.00\%$) |
| `FACE_MATCH_MARGIN` | `0.05` | Minimum score separation required between top candidate and runner-up |
| `FALL_TORSO_ANGLE_DEG` | `60.0°` | Shoulder-to-hip torso tilt angle from vertical triggering collapse alert |
| `FALL_SUSTAIN_SEC` | `1.0s` | Required temporal duration of horizontal posture before fall dispatch |
| `LOITERING_THRESHOLD_SEC`| `4.0s` | Dwell duration triggering loitering escalation for unauthorized persons |
| `REAUTH_INTERVAL_SEC` | `10.0s` | Periodic re-verification timer to detect ID switches or substitutions |

---

## Team

| Name | Primary Responsibility |
| :--- | :--- |
| **Guna** | YuNet face landmark detection, FaceNet512 embedding engine, Top-K matching and decision rules |
| **Susan** | Behavioral anomaly detection (YOLOv8-Pose fall tilt, running speed), audit logging and documentation |
| **Vivek** | Person detection, ByteTrack tracker pipeline, Tripwire ROI geometry and UI dashboard engineering |
