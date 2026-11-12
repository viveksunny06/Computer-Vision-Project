# Intelligent Real-Time AI Surveillance & Behavioral Threat Detection System

A real-time live camera surveillance and threat-monitoring application featuring interactive tripwire zones, biometric facial verification, automated incident logging (CSV audit trail & annotated JPEG snapshots), and training-free pose/motion anomaly detection.

---

## Features

- **Dedicated Live Camera Pipeline** — Real-time capture (Device 0) with aspect-ratio-preserving letterboxing, clean Standby screen when paused, and thread-safe Queue rendering.
- **Biometric Facial Verification (`Facenet512` + `YuNet`)** —
  - **Landmark Verification**: Requires genuine human facial landmarks before matching; covered or obscured faces are rejected (`FACE OBSCURED / INTRUDER`).
  - **Top-K + Margin Matching**: Compares 512-D embeddings against enrolled personnel using Top-3 mean cosine similarity plus a runner-up margin check (`FACE_DISTANCE_THRESH = 0.35`, `FACE_MATCH_MARGIN = 0.05`).
  - **Grace Period & Voting State Machine**: Tracks progress through `OUTSIDE → SCANNING → AUTHORIZED | INTRUDER | OBSCURED` using `2-of-3` vote quorum over a 2.5s grace window, plus 10s periodic re-authentication.
- **Training-Free Behavioural & Safety Anomalies**:
  - **Pose-Based Fall Detection**: Uses `yolov8n-pose.pt` torso tilt angle (`> 60°` sustained `>= 1.0s`), with calibrated bounding-box fallback.
  - **Running / Rapid Movement**: Scale-invariant body-heights/sec tracking (`> 1.5 bh/s` sustained `>= 0.5s`).
  - **Loitering Escalation**: Escalates unauthorized intrusion if dwell time exceeds `4.0s`.
  - **Crowding & After-Hours Rules**: Configurable `MAX_PEOPLE_IN_ROI` and `RESTRICTED_HOURS`.
- **Automated Incident Logging (`AlertManager`)**:
  - Structured **`outputs/incidents.csv`** log recording ISO timestamps, track ID, name, status, dwell time, and message.
  - Full-frame annotated JPEG snapshots saved automatically to `outputs/` with tripwire boundaries.
  - Auto-pruning retention policy (default 30 days).

---

## Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Single-File Execution

The entire system is self-contained in **`main.py`**:

```powershell
# Run the complete AI Surveillance System dashboard
python main.py
```

*Note: You can also open the Face Enrollment Studio directly from inside the dashboard by clicking **`👤 Face Enrollment Studio`**, or run it directly via CLI:*

```powershell
python main.py --enroll
```

### 3. Quick Operation Guide

1. **Dashboard Controls**:
   - Draw a custom **Tripwire Zone** directly by clicking and dragging on the video canvas (or use default).
   - Click **`▶ Start Surveillance`** to activate live webcam threat monitoring.
   - Click **`👤 Face Enrollment Studio`** to enroll new authorized personnel with guided poses.
   - Click **`Send Test Alert`** to test CSV incident logging and snapshot generation.
2. **Review Incidents**:
   - Click **`📂 Open Snapshots Folder`** to inspect captured breach snapshots.
   - Click **`📄 Open CSV Log`** to review `outputs/incidents.csv`.

---

## Submission Package Inventory

For academic/course evaluation requiring a single self-contained `.py` file:

| Item | Status | Purpose |
| :--- | :--- | :--- |
| **`main.py`** | **KEEP (Core)** | **Single consolidated executable** containing GUI, YOLOv8 tracker, Facenet512 engine, AlertManager, and Face Enrollment Studio. |
| **`requirements.txt`** | **KEEP** | Package dependencies (`ultralytics`, `deepface`, `opencv-python`, etc.). |
| **`known_faces/`** | **KEEP** | Enrolled face dataset (`vivek/`, `Susan/`) for live verification. |
| **`yolov8n-pose.pt`** | **KEEP** | Local pose model weights (enables offline evaluation without downloading). |
| **`yolov8n.pt`** | **KEEP** | Fallback detection model weights. |
| **`cache/`** | **KEEP** | Pre-computed embeddings (`embeddings.npz`) for instant zero-delay startup. |
| **`outputs/`** | **KEEP** | Sample incident logs (`incidents.csv`) and breach snapshots. |
| **`README.md` / `context.md`** | **KEEP** | Documentation and technical project report. |
| `_archive/` | *EXCLUDE / BACKUP* | Archived legacy modular files (`config.py`, `face_db.py`, etc., now merged into `main.py`). |

---

## Evaluation Scenario Matrix

Test and record performance across:
1. Authorized person frontal & multi-pose
2. Authorized person entering sideways, then turning toward camera
3. Stranger entering alone
4. Authorized person + stranger in the frame together (verify independent track states)
5. Face covered (T-shirt, mask, hand) → `FACE OBSCURED`
6. Head turned completely away during grace window
7. Replay attack check (printed/phone photo — expected known limitation)
8. Low-light entrance
9. Two people crossing paths (verify 10s re-auth catches ID swaps)
10. Real collapse/fall vs. sitting close to desk webcam
11. Unauthorized loitering `> 4.0s`
12. Leaving and re-entering the ROI (dwell timer reset + alert cooldown)

---

## Limitations & Responsible Use

- **Anti-Spoofing**: This prototype does not perform 3D depth or liveness verification; a high-resolution printed photo or screen replay of an enrolled person can match.
- **Lighting & Pose**: Extreme backlighting, low resolution (`< 48 px` face width), or severe motion blur degrades detector confidence and delays verification.
- **Demographic Variance**: Pretrained facial recognition models can exhibit accuracy disparities across skin tones, ages, and eyewear. Always calibrate thresholds (`tools/calibrate_threshold.py`) on representative target users.
- **Privacy & Data Storage**: Enrolled face images in `known_faces/` and cached embeddings in `cache/` are stored unencrypted locally. Only enroll individuals with explicit informed consent. Incident snapshots in `outputs/` are automatically pruned after `SNAPSHOT_RETENTION_DAYS = 30` days.
- **Prototype Notice**: This software is an educational/research prototype and is **not** certified for life-safety, medical fall monitoring, or regulated physical security deployments.
