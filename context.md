# Intelligent Real-Time AI Surveillance & Behavioral Threat Detection System
## System Architecture, Current Implementation Status, Decisions & Roadmap

---

## 1. Project Overview & Core Vision

The **Intelligent Real-Time AI Surveillance & Behavioral Threat Detection System** is an autonomous, vision-based security platform that converts commodity webcams into proactive threat-monitoring systems.

Unlike passive recording, this system autonomously:
1. **Tracks Human Targets & Body Pose**: Uses YOLOv8-Pose (`yolov8n-pose.pt`, with `yolov8n.pt` fallback) and ByteTrack (`bytetrack.yaml`) to track individuals and extract 17 COCO body keypoints in real time.
2. **Enforces Virtual Tripwire Boundaries (ROI)**: Operators interactively draw and adjust rectangular tripwire zones on the canvas; ROI membership checks both feet and bounding-box center (`ROI_MODE = "any"`).
3. **Verifies Biometric Authorization**: Uses OpenCV **YuNet** facial landmark detection + **Facenet512** (512-dimensional embeddings) in an in-memory `FaceDB` engine with Top-3 mean cosine similarity, runner-up margin validation, grace-period multi-frame voting (`2 of 3`), and periodic re-authentication (`10s`).
4. **Detects Behavioural & Safety Anomalies (Training-Free)**:
   - **Intrusion / Face Obscured Breaches**: Instant flagging after the 2.5s scanning grace period when unrecognized or face-covered targets remain in the zone.
   - **Suspicious Loitering**: Escalates unauthorized intrusion when dwell time exceeds `4.0s`.
   - **Pose-Based Fall / Collapse Detection**: Computes torso tilt angle from shoulders/hips (`> 60°` sustained `>= 1.0s`), with calibrated bounding-box ratio fallback.
   - **Running / Rapid Movement**: Scale-invariant movement speed (`> 1.5 body-heights/sec` sustained `>= 0.5s`).
   - **Zone Crowding & Restricted Hours**: Optional configurable `MAX_PEOPLE_IN_ROI` and `RESTRICTED_HOURS`.
5. **Incident Logging & Auditing (`AlertManager`)**:
   - Saves full-frame annotated JPEG snapshots (with all bounding boxes + ROI) to `outputs/`.
   - Appends structured audit records to `outputs/incidents.csv` and updates the live GUI log.

---

## 2. System Architecture & Pipeline Flow

```
                     +----------------------------------------+
                     |       Live Camera Feed (Device 0)      |
                     +----------------------------------------+
                                         |
                                         v
                     +----------------------------------------+
                     |    YOLOv8-Pose + ByteTrack Tracker     |
                     |    (BBox + 17 COCO Keypoints, Cls 0)   |
                     +----------------------------------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
                       v                                   v
             [Outside ROI (any)]                  [Inside ROI (feet or center)]
                       |                                   |
                       |                     +-------------+-------------+
                       |                     |  Bounded Verify Worker    |
                       |                     |  (1 Thread, Queue max=8)  |
                       |                     +-------------+-------------+
                       |                                   |
                       |                     [FaceDB: YuNet + Facenet512]
                       |                     [Top-3 Mean + 0.05 Margin  ]
                       |                                   |
                       |             +---------------------+---------------------+
                       |             |                     |                     |
                       |      [2/3 MATCH Votes]     [Grace >= 2.5s]      [LOW_QUALITY / ERROR]
                       |             |                     |                     |
                       |             v              +------+------+              |
                       |       [AUTHORIZED]         |             |              |
                       |        (Re-auth 10s)   [>=50% NO_FACE]  [Else]         (No Vote)
                       |             |              |             |
                       |             |              v             v
                       v             v         [OBSCURED]    [INTRUDER]
                [Outside Zone]  [AUTHORIZED]     (Red)          (Red)
                   (Yellow)       (Green)          |              |
                                     |             +------+-------+
                                     v                    v
                         [Pose Fall / Running]   [AlertManager Worker Thread]
                                                         |
                                    +--------------------+--------------------+
                                    |                    |                    |
                                    v                    v                    v
                           [JPEG + incidents.csv]   [Siren WAV]       [Telegram Photo]
```

---

## 3. Single-File Consolidation & Submission Structure

To satisfy academic/course evaluation constraints requiring **one single `.py` file**, the entire modular architecture (`config.py`, `theme.py`, `face_db.py`, `alerts.py`, `anomaly.py`, `register_face.py`, and `surveillance.py`) has been unified into a single self-contained executable: **`main.py`**.

```text
Computer-Vision-Project/
│
├── main.py                      # [SINGLE FILE] Complete self-contained AI surveillance application
│                                # (GUI, Face Enrollment Studio, YOLOv8 tracker, Facenet512, AlertManager)
├── requirements.txt             # Pinned package dependencies
├── README.md                    # Quickstart, setup, and submission inventory
├── context.md                   # Comprehensive system report and technical architecture
├── yolov8n-pose.pt              # Local YOLOv8-Pose nano weights (~6.8 MB)
├── yolov8n.pt                   # Local YOLOv8 fallback detection weights (~6.5 MB)
│
├── cache/                       # embeddings.npz + manifest.json (instant startup cache)
├── known_faces/                 # Authorized personnel biometric folders (e.g., vivek/, Susan/)
├── outputs/                     # Incidents log (incidents.csv) & alert snapshots
│
└── _archive/                    # Archived legacy modular files (kept as backup, exclude from submission)
    ├── alerts.py
    ├── anomaly.py
    ├── config.py
    ├── face_db.py
    ├── theme.py
    ├── register_face.py
    └── create_test_assets.py
```

---

## 4. Current Working Condition & Capabilities

### A. Live Surveillance & Threat Monitor (`main.py`)
- **Per-Track State Machine**: Eliminates single-frame false positives via `SCANNING` (2.5s grace period), `2-of-3` vote quorum (`VOTE_WINDOW = 3`, `VOTES_REQUIRED = 2`), and 10s periodic re-authentication (`REAUTH_INTERVAL_SEC`) to catch ID swaps. Tracks unseen for `30s` are purged automatically.
- **Explicit Failure Separation**: `FaceDB` cleanly distinguishes `MATCH`, `NO_MATCH`, `NO_FACE`, `LOW_QUALITY` (`face_px < 48`), and `ERROR`. System/runtime errors never masquerade as `FACE OBSCURED`.
- **Single Verification Worker**: One long-lived daemon thread processes verification requests via an in-memory bounded `queue.Queue(maxsize=8)` with zero temp JPEGs on disk.
- **ROI & Display**: Proportional letterboxing preserves 4:3 / 16:9 webcam aspect ratios; ROI membership (`ROI_MODE = "any"`) checks both bounding-box center and feet so seated desk subjects trigger accurately. Clean Standby screen clears the last frame on stop.
- **HUD & Snapshots**: ASCII-only OpenCV badge labels clamped inside frame bounds; snapshots are captured after all bounding boxes are drawn and include the ROI overlay. Auto-reloads `FaceDB` within ~2s when `known_faces/` changes without resetting breach flags.

### B. Personnel Face Enrollment Studio (`register_face.py`)
- **Thread-Safe Queue Preview**: Camera thread communicates strictly via `frame_queue`, `status_queue`, and `capture_result_queue`, drained on the Tk main thread (`_poll_queues`), with safe `thread.join()` hardware shutdown.
- **Background Quality Validation**: Rejects frames with 0 or >1 faces, confidence `< 0.90`, face width `< 80 px`, Laplacian sharpness `< 60.0`, or near-duplicates (`cosine sim > 0.98`).
- **Guided Capture & Clean Crops**: Cycles through 7 pose prompts (`Front`, tilts, `Smile`, etc.) with `n / 10 recommended` guidance, saves 40%-padded face crops verified with `FaceDB.embed`, pre-warms `cache/embeddings.npz`, and handles NTFS case-insensitivity plus reserved Windows filenames.

### C. Biometric Calibration Results (`tools/calibrate_threshold.py`)
Run against the enrolled database (`40` genuine leave-one-out pairs + `40` cross-person impostor pairs + Puneeth probe):
- **Genuine Distances (`Facenet512` + `YuNet`, Top-3)**: `[0.0753, 0.3466]`, mean = `0.1671`.
- **Impostor Distances (cross-person)**: `[0.4797, 0.9815]`, mean = `0.6763` (Puneeth vs. Vivek = `0.5376`).
- **EER**: `0.00%` at threshold **`0.35`** (`FAR = 0.00%`, `FRR = 0.00%`).

---

## 5. Decisions Recorded

| Setting / Choice | Decision Made | Rationale |
| :--- | :--- | :--- |
| **Face Recognition Model** | `Facenet512` (512-D) + `yunet` | 128-D `Facenet` collapsed Puneeth and Vivek to cosine dist `0.1513`; `Facenet512` separated them cleanly (`0.1429` vs `0.5376`). |
| **Distance Threshold & Margin** | `FACE_DISTANCE_THRESH = 0.35`, `FACE_MATCH_MARGIN = 0.05`, `FACE_TOPK = 3` | Calibrated on `outputs/calibration/roc.csv` yielding `0.00% EER`; Top-3 + margin stops single outlier photos from falsely authorizing strangers. |
| **ROI Membership Mode** | `ROI_MODE = "any"` (feet or center) | Seated/close-up webcam subjects often have bottom-edge `by2` outside the tripwire rectangle while torso center is clearly inside. |
| **Pose Model & Fall Rule** | `yolov8n-pose.pt` torso angle `> 60°` sustained `>= 1.0s` | Eliminates false fall alerts caused by seated upper-body bounding boxes (`W/H ~ 1.33`). |
| **Embedding Cache Location** | `cache/embeddings.npz` + `cache/manifest.json` | Keeps `known_faces/` clean of DeepFace `.pkl/.npz` artifacts and supports fast incremental delta updates. |

---

## 6. Lessons Learned (Resolved Issues)

1. **128-D `Facenet` + Outlier Matching False Positives**: Switching to `Facenet512` (512-D), Top-3 gallery averaging, runner-up margin (`0.05`), and `2-of-3` multi-frame voting eliminated cross-person false positives (Puneeth tagged as Vivek).
2. **Skipped Detector & Non-Face Enrollment Leakage**: Enforcing `yunet` facial landmark detection (`enforce_detection=True`) in both verification and enrollment ensures covered/obscured faces (`FACE OBSCURED`) can never match background or clothing embeddings.
3. **Silent Verification Stalls & Single-Shot Lock-In**: Explicit `VerifyResult` error types (`NO_FACE`, `LOW_QUALITY`, `ERROR`) and a `SCANNING` grace state prevent momentary motion blur from permanently locking an authorized person as an intruder.
4. **Thread-Safety & Camera Lifecycle**: Queue-only Tk UI updates across both GUIs and `join()` before `cap.release()` eliminated Windows MSMF errors and `RuntimeError: main thread is not in main loop`.

---

## 7. Roadmap (Completed & Remaining Optional Work)

### Completed (Phases 1–5 Core)
- [x] Centralized `config.py` (with `config.json` persistence) & shared `theme.py`
- [x] `FaceDB` (`Facenet512` + `YuNet`, Top-K + margin, incremental `cache/`)
- [x] Per-track verification state machine (`OUTSIDE | SCANNING | AUTHORIZED | INTRUDER | OBSCURED`), grace period, `2/3` voting, `10s` re-auth, `30s` track purge
- [x] `AlertManager` (Windows siren WAV, `incidents.csv`, annotated JPEG snapshots, Telegram photo alerts, 30-day pruning, GUI toggles & Test Alert)
- [x] Thread-safe `register_face.py` with background quality validation, guided poses, 40% crops & NTFS name safety
- [x] Training-free `yolov8n-pose` torso-angle fall detection, scale-invariant running detection, crowding & after-hours checks
- [x] Quantitative threshold calibration (`tools/calibrate_threshold.py`) + pinned `requirements.txt`, `.gitignore`, and `README.md`

### Optional Future Enhancements
- [ ] **Whole-frame YuNet Face Association (`B3`)**: Run YuNet periodically on the full frame and match faces to person boxes by containment.
- [ ] **Liveness / Anti-Spoofing (`B4`)**: Add depth/texture liveness checks to defeat printed or phone-screen photo replays.
- [ ] **Abandoned Object Detector (`D3`)**: Second COCO detector pass for unattended backpacks/suitcases.
- [ ] **Line-Crossing Tripwire & Multi-Point Polygon ROI (`D5`, `D6`)**: Directional segment crossing mode and arbitrary polygon exclusion zones.
- [ ] **Tk-Free Pipeline Split & Offline Clip Evaluator (`G1`, `F2`)**: Extract `pipeline.py` for headless CLI deployment and automated MP4 benchmark evaluation.
