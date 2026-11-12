#!/usr/bin/env python3
"""
main.py
-------
AI Surveillance & Behavioral Threat Detection System
Computer Vision Project - Single Consolidated Submission File

Features & Architecture:
  1. Centralized Configuration & Design System (Slate 900/800 aesthetic).
  2. Training-free Behavioral Threat Detection:
     - YOLOv8-Pose torso-angle fall detection (>60 deg from vertical) with aspect-ratio fallback.
     - Scale-invariant running / sudden movement detection (body-heights / sec).
     - Overcrowding and restricted-hours occupancy detection.
     - Interactive drag-to-set Tripwire ROI with aspect-ratio preserved coordinate mapping.
  3. Multi-Channel Alert & Logging Engine (AlertManager):
     - Threaded worker with cooldown deduplication per track.
     - Auto-saved JPEG incident snapshots with tripwire boundary overlay (30-day retention).
     - Persistent CSV incident logging (`outputs/incidents.csv`).
     - Scrolling live GUI incident log.
  4. High-Precision Biometric Face Recognition Engine (FaceDB):
     - DeepFace Facenet512 (512-D) + OpenCV YuNet landmark detector.
     - Multi-template Top-K cosine similarity aggregation + runner-up margin check.
     - Incremental disk caching (`cache/embeddings.npz` and `cache/manifest.json`).
     - Strict face presence enforcement (prevents false positives on clothing or textures).
  5. Built-in Personnel Face Enrollment Studio (FaceEnrollmentDialog):
     - Multi-pose guided enrollment workflow (7 poses, progress bar).
     - Real-time quality verification (Confidence >= 0.90, Size >= 80px, Laplacian Sharpness >= 60.0).
     - Near-duplicate cosine guard (<= 0.98) and automatic 40%-padded face cropping.
     - Safe camera handover (pauses surveillance during enrollment to avoid camera resource locks).
  6. Main Surveillance Dashboard (SurveillanceGUIApp):
     - Zero-lag multithreaded queue-driven rendering.
     - Multi-frame state machine: OUTSIDE -> SCANNING -> AUTHORIZED / INTRUDER / OBSCURED.
     - Clean standby screen when stopped (no frozen frames).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import wave
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk

# ===========================================================================
# 1. CENTRALIZED CONFIGURATION & SYSTEM CONSTANTS
# ===========================================================================
BASE_DIR: Path = Path(__file__).resolve().parent
KNOWN_FACES_DIR: Path = BASE_DIR / "known_faces"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
CACHE_DIR: Path = BASE_DIR / "cache"
ASSETS_DIR: Path = BASE_DIR / "assets"
CONFIG_OVERRIDE_PATH: Path = BASE_DIR / "config.json"

CAMERA_INDEX: int = 0

# Detection & Tracking
PERSON_MODEL_PATH: str = "yolov8n-pose.pt"
PERSON_FALLBACK_MODEL_PATH: str = "yolov8n.pt"
YOLO_CONF: float = 0.40
TRACKER_CFG: str = "bytetrack.yaml"
ROI_MODE: str = "any"  # "feet" | "center" | "any"

# Verification State Machine
GRACE_PERIOD_SEC: float = 2.5
REVERIFY_INTERVAL_SEC: float = 1.0
VOTE_WINDOW: int = 3
VOTES_REQUIRED: int = 2
REAUTH_INTERVAL_SEC: float = 10.0
ALERT_COOLDOWN_SEC: float = 10.0
TRACK_STALE_SEC: float = 30.0

# Biometric Engine (Facenet512 + YuNet)
FACE_MODEL: str = "Facenet512"
FACE_DETECTOR: str = "yunet"
FACE_DISTANCE_THRESH: float = 0.35
FACE_MATCH_MARGIN: float = 0.05
FACE_TOPK: int = 3
MIN_FACE_PX: int = 48

# Enrollment Studio Validation
ENROLL_MIN_CONFIDENCE: float = 0.90
ENROLL_MIN_FACE_PX: int = 80
ENROLL_MIN_SHARPNESS: float = 60.0
ENROLL_MAX_SIMILARITY: float = 0.98
ENROLL_RECOMMENDED_COUNT: int = 10

# Behavioral & Safety Anomalies
ENABLE_POSE_FALL: bool = True
ENABLE_RUNNING: bool = True
ENABLE_OVERCROWDING: bool = True
ENABLE_RESTRICTED_HOURS: bool = True

LOITERING_THRESHOLD_SEC: float = 4.0
FALL_ASPECT_RATIO_THRESHOLD: float = 1.65
FALL_TORSO_ANGLE_DEG: float = 60.0
FALL_SUSTAIN_SEC: float = 1.0
RUN_SPEED_BODYHEIGHTS_PER_S: float = 1.5
RUN_SUSTAIN_SEC: float = 0.5
MAX_PEOPLE_IN_ROI: int = 3
RESTRICTED_HOURS: Optional[Tuple[str, str]] = None  # e.g., ("22:00", "06:00")

# Alerts & Retention
SNAPSHOT_RETENTION_DAYS: int = 30

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}

POSE_PROMPTS = [
    "Front (look straight at camera)",
    "Slightly left",
    "Slightly right",
    "Slightly up",
    "Slightly down",
    "Natural smile",
    "Neutral expression (with/without glasses if applicable)",
]

_PATH_KEYS = {"KNOWN_FACES_DIR", "OUTPUTS_DIR", "CACHE_DIR", "ASSETS_DIR"}


def _apply_config_overrides() -> None:
    """Load user settings from config.json if present."""
    if not CONFIG_OVERRIDE_PATH.exists():
        return
    try:
        data = json.loads(CONFIG_OVERRIDE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        g = globals()
        for k, v in data.items():
            if k in g and k.isupper():
                g[k] = Path(v) if k in _PATH_KEYS else v
    except Exception as exc:
        print(f"[CONFIG] Failed to load config overrides: {exc}")


def save_config_overrides(updates: Dict[str, Any]) -> None:
    """Persist runtime config overrides to config.json."""
    existing: Dict[str, Any] = {}
    if CONFIG_OVERRIDE_PATH.exists():
        try:
            loaded = json.loads(CONFIG_OVERRIDE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    g = globals()
    for k, v in updates.items():
        if k in g and k.isupper():
            g[k] = Path(v) if k in _PATH_KEYS else v
            existing[k] = str(v) if isinstance(v, Path) else v

    try:
        CONFIG_OVERRIDE_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[CONFIG] Failed to persist config.json: {exc}")


_apply_config_overrides()


# ===========================================================================
# 2. DESIGN SYSTEM & COLOR THEME (Slate 900 / 800)
# ===========================================================================
THEME = {
    "bg_main": "#0f172a",        # Slate 900
    "bg_surface": "#1e293b",     # Slate 800
    "bg_card": "#334155",        # Slate 700
    "bg_input": "#0f172a",       # Slate 900
    "border": "#475569",         # Slate 600
    "border_light": "#64748b",   # Slate 500
    "text_primary": "#f8fafc",   # Slate 50
    "text_secondary": "#94a3b8", # Slate 400
    "text_muted": "#64748b",     # Slate 500
    "primary": "#2563eb",        # Royal Blue 600
    "primary_hover": "#1d4ed8",  # Royal Blue 700
    "success": "#16a34a",        # Emerald 600
    "success_hover": "#15803d",  # Emerald 700
    "danger": "#dc2626",         # Crimson 600
    "danger_hover": "#b91c1c",   # Crimson 700
    "warning": "#d97706",        # Amber 600
    "neutral_btn": "#475569",    # Slate 600
    "neutral_hover": "#334155",  # Slate 700
}


# ===========================================================================
# 3. BEHAVIORAL & SAFETY ANOMALY HELPERS
# ===========================================================================
def torso_angle_deg(
    kp: Optional[np.ndarray],
    conf: Optional[np.ndarray],
    min_conf: float = 0.40,
) -> Optional[float]:
    """
    Compute torso tilt angle from vertical (degrees: 0 = upright, 90 = horizontal).
    Uses COCO keypoints for shoulders (5, 6) and hips (11, 12).
    """
    if kp is None or conf is None or len(kp) < 13 or len(conf) < 13:
        return None
    for idx in (5, 6, 11, 12):
        if float(conf[idx]) < min_conf:
            return None
    sh = (kp[5] + kp[6]) * 0.5
    hip = (kp[11] + kp[12]) * 0.5
    dx = float(hip[0] - sh[0])
    dy = float(hip[1] - sh[1])
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def is_fall_candidate(
    bbox: Tuple[int, int, int, int],
    frame_h: int,
    kp: Optional[np.ndarray] = None,
    kp_conf: Optional[np.ndarray] = None,
) -> Tuple[bool, Optional[float]]:
    """
    Evaluate whether person is in a fallen/horizontal posture.
    Returns (is_candidate, torso_angle_or_None).
    """
    if ENABLE_POSE_FALL:
        angle = torso_angle_deg(kp, kp_conf)
        if angle is not None:
            return (angle > FALL_TORSO_ANGLE_DEG), angle

    bx1, by1, bx2, by2 = bbox
    w = max(bx2 - bx1, 1)
    h = max(by2 - by1, 1)
    aspect_ratio = w / h
    fallback_fall = (
        (aspect_ratio > FALL_ASPECT_RATIO_THRESHOLD)
        and (by2 > int(frame_h * 0.65))
        and (w * h > 8000)
    )
    return fallback_fall, None


def compute_body_speed(
    trail: Deque[Tuple[float, float, float]],
    bbox_h: int,
) -> float:
    """Compute centroid path speed in body-heights per second."""
    if len(trail) < 3 or bbox_h <= 10:
        return 0.0
    dt = trail[-1][0] - trail[0][0]
    if dt < 0.25:
        return 0.0

    dist_px = 0.0
    for i in range(1, len(trail)):
        dx = trail[i][1] - trail[i - 1][1]
        dy = trail[i][2] - trail[i - 1][2]
        dist_px += math.hypot(dx, dy)
    return (dist_px / dt) / float(bbox_h)


def is_within_restricted_hours(
    window: Optional[Tuple[str, str]],
    now_dt: Optional[datetime] = None,
) -> bool:
    """Return True if current local time falls within (start_HHMM, end_HHMM)."""
    if not window or len(window) != 2:
        return False
    try:
        sh, sm = map(int, window[0].split(":"))
        eh, em = map(int, window[1].split(":"))
    except Exception:
        return False

    dt = now_dt or datetime.now()
    cur = dt.hour * 60 + dt.minute
    start = sh * 60 + sm
    end = eh * 60 + em

    if start <= end:
        return start <= cur < end
    # Overnight window, e.g. 22:00 -> 06:00
    return cur >= start or cur < end


def is_in_roi(
    bbox: Tuple[int, int, int, int],
    roi_poly: np.ndarray,
    mode: str = ROI_MODE,
) -> bool:
    """Return True if target reference point(s) lie inside or on the ROI polygon."""
    x1, y1, x2, y2 = bbox
    feet = ((x1 + x2) / 2.0, float(y2))
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    pts = {"feet": [feet], "center": [center], "any": [feet, center]}.get(mode, [feet, center])
    return any(cv2.pointPolygonTest(roi_poly, (float(px), float(py)), False) >= 0 for px, py in pts)


def open_path_in_os(target: Path) -> None:
    """Cross-platform helper to open a folder or file in the OS explorer."""
    resolved = str(target.resolve())
    if sys.platform == "win32":
        os.startfile(resolved)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", resolved])
    else:
        subprocess.Popen(["xdg-open", resolved])


# ===========================================================================
# 4. INCIDENT LOGGING & ALERT MANAGER
# ===========================================================================
CSV_HEADER = [
    "timestamp_iso",
    "event_type",
    "track_id",
    "name",
    "state",
    "distance",
    "dwell_sec",
    "message",
    "snapshot_path",
]


class AlertManager:
    """Asynchronous incident logger and snapshot dispatcher with cooldown deduplication."""

    def __init__(
        self,
        outputs_dir: Path = OUTPUTS_DIR,
        log_queue: Optional[queue.Queue] = None,
    ):
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.log_queue = log_queue

        self._csv_path = self.outputs_dir / "incidents.csv"
        self._prune_old_snapshots()

        self._queue: queue.Queue = queue.Queue(maxsize=64)
        self._cooldowns: Dict[Tuple[str, int], float] = {}
        self._stop_event = threading.Event()

        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def _prune_old_snapshots(self) -> None:
        """Enforce snapshot retention policy (default 30 days)."""
        retention_days = SNAPSHOT_RETENTION_DAYS
        if retention_days <= 0 or not self.outputs_dir.exists():
            return
        cutoff_sec = time.time() - (retention_days * 86400.0)
        try:
            for item in self.outputs_dir.iterdir():
                if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    try:
                        if item.stat().st_mtime < cutoff_sec:
                            item.unlink()
                    except Exception:
                        pass
        except Exception as exc:
            print(f"[ALERTS] Snapshot retention prune error: {exc}")

    def emit(
        self,
        event_type: str,
        track_id: int,
        message: str,
        snapshot_bgr: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Enqueue alert with per-(event_type, track_id) cooldown check."""
        now = time.time()
        cd_key = (event_type, track_id)
        last_t = self._cooldowns.get(cd_key, 0.0)
        if (now - last_t) < ALERT_COOLDOWN_SEC:
            return False

        self._cooldowns[cd_key] = now
        item = {
            "time": now,
            "event_type": event_type,
            "track_id": track_id,
            "message": message,
            "snapshot": snapshot_bgr.copy() if snapshot_bgr is not None else None,
            "extra": dict(extra or {}),
        }
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            print(f"[ALERTS] Queue full, dropped alert: {event_type} | {message}")
            return False

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if item is None:
                break

            try:
                self._handle_alert(item)
            except Exception as exc:
                print(f"[ALERTS] Dispatch error: {exc}\n{traceback.format_exc()}")
            finally:
                self._queue.task_done()

    def _handle_alert(self, item: Dict[str, Any]) -> None:
        event_type = item["event_type"]
        track_id = item["track_id"]
        msg = item["message"]
        snapshot_bgr = item.get("snapshot")

        # 1. Forward to GUI log queue
        if self.log_queue is not None:
            ts_str = time.strftime("%H:%M:%S", time.localtime(item["time"]))
            self.log_queue.put(f"[{ts_str}] {msg}")

        # 2. Save JPEG snapshot
        snapshot_rel_path = ""
        if snapshot_bgr is not None:
            dt_tag = datetime.fromtimestamp(item["time"]).strftime("%Y%m%d_%H%M%S_%f")[:19]
            safe_type = "".join(c if c.isalnum() else "_" for c in event_type.lower())
            filename = f"alert_{safe_type}_id{track_id}_{dt_tag}.jpg"
            snap_path = self.outputs_dir / filename
            try:
                cv2.imwrite(str(snap_path), snapshot_bgr)
                snapshot_rel_path = str(snap_path)
            except Exception as exc:
                print(f"[ALERTS] Failed to save snapshot: {exc}")

        # 3. Append to CSV incident log
        self._append_csv(item, snapshot_rel_path)

    def _append_csv(self, item: Dict[str, Any], snapshot_path: str) -> None:
        try:
            write_header = not self._csv_path.exists()
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(CSV_HEADER)
                dt_iso = datetime.fromtimestamp(item["time"]).isoformat(timespec="seconds")
                extra = item.get("extra", {})
                dist = extra.get("distance")
                dist_str = f"{dist:.3f}" if isinstance(dist, (float, int)) else ""
                dwell = extra.get("dwell_sec")
                dwell_str = f"{dwell:.1f}" if isinstance(dwell, (float, int)) else ""
                writer.writerow([
                    dt_iso,
                    item["event_type"],
                    item["track_id"],
                    extra.get("name", ""),
                    extra.get("state", ""),
                    dist_str,
                    dwell_str,
                    item["message"],
                    snapshot_path,
                ])
        except Exception as exc:
            print(f"[ALERTS] CSV append error: {exc}")

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        except Exception:
            pass


# ===========================================================================
# 5. BIOMETRIC FACE EMBEDDING & VERIFICATION ENGINE (FaceDB)
# ===========================================================================
class NoFaceError(Exception):
    """Raised when the face detector confirms no human face is visible."""


@dataclass
class VerifyResult:
    kind: str  # "MATCH" | "NO_MATCH" | "NO_FACE" | "LOW_QUALITY" | "ERROR" | "RELOAD"
    name: Optional[str] = None
    distance: Optional[float] = None
    margin: Optional[float] = None
    face_px: Optional[int] = None
    error: Optional[str] = None


class FaceDB:
    """In-memory 512-D biometric embedding database with incremental caching."""

    def __init__(
        self,
        known_dir: Path = KNOWN_FACES_DIR,
        cache_dir: Path = CACHE_DIR,
        model: str = FACE_MODEL,
        detector: str = FACE_DETECTOR,
    ):
        self.known_dir = Path(known_dir)
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.detector = detector

        self.known_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._manifest_path = self.cache_dir / "manifest.json"
        self._npz_path = self.cache_dir / "embeddings.npz"

        self._swap_lock = threading.Lock()
        self._model_lock = threading.Lock()

        self._E: Optional[np.ndarray] = None  # (N, 512) unit vectors
        self._labels: List[str] = []
        self._idx: Dict[str, np.ndarray] = {}

        self._cleanup_legacy_caches()

    def _cleanup_legacy_caches(self) -> None:
        """Remove legacy DeepFace .pkl files from known_faces/."""
        try:
            for item in self.known_dir.iterdir():
                if item.is_file() and (item.suffix in {".pkl", ".npz"} or item.name.startswith("ds_model_")):
                    try:
                        item.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    def _scan_files(self) -> List[Tuple[str, str, Path, int, int]]:
        entries: List[Tuple[str, str, Path, int, int]] = []
        if not self.known_dir.exists():
            return entries

        for item in sorted(self.known_dir.iterdir(), key=lambda p: p.name.lower()):
            if item.is_dir():
                person_key = item.name
                for f in sorted(item.iterdir(), key=lambda p: p.name.lower()):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        st = f.stat()
                        rel = f"{person_key}/{f.name}"
                        entries.append((rel, person_key, f, st.st_size, st.st_mtime_ns))
            elif item.is_file() and item.suffix.lower() in IMAGE_EXTS:
                st = item.stat()
                person_key = item.stem
                entries.append((item.name, person_key, item, st.st_size, st.st_mtime_ns))
        return entries

    def _load_cache_manifest(self) -> Dict[str, Any]:
        if not self._manifest_path.exists() or not self._npz_path.exists():
            return {}
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            if data.get("model") != self.model or data.get("detector") != self.detector:
                return {}
            return data
        except Exception as exc:
            print(f"[FACE_DB] Manifest read error: {exc}")
            return {}

    def _read_npz(self) -> Dict[str, np.ndarray]:
        if not self._npz_path.exists():
            return {}
        try:
            with np.load(str(self._npz_path)) as data:
                return {k: data[k] for k in data.files}
        except Exception as exc:
            print(f"[FACE_DB] NPZ read error: {exc}")
            return {}

    def reload(self) -> Tuple[int, int]:
        """Incrementally sync disk directory with cache and load vectors into memory."""
        disk_files = self._scan_files()
        manifest = self._load_cache_manifest()
        cached_entries: Dict[str, Dict[str, Any]] = manifest.get("entries", {})
        cached_npz = self._read_npz()

        new_entries: Dict[str, Dict[str, Any]] = {}
        new_npz: Dict[str, np.ndarray] = {}
        to_embed: List[Tuple[str, str, Path, int, int]] = []

        disk_rel_set = set()
        for rel, person_key, path, sz, mtime in disk_files:
            disk_rel_set.add(rel)
            c = cached_entries.get(rel)
            if c is not None and c.get("size") == sz and c.get("mtime_ns") == mtime and rel in cached_npz:
                new_entries[rel] = c
                new_npz[rel] = cached_npz[rel]
            else:
                to_embed.append((rel, person_key, path, sz, mtime))

        if to_embed:
            print(f"[FACE_DB] Incrementally embedding {len(to_embed)} photo(s) with {self.model}+{self.detector}...")

        from deepface import DeepFace

        for rel, person_key, path, sz, mtime in to_embed:
            try:
                with self._model_lock:
                    reps = DeepFace.represent(
                        img_path=str(path),
                        model_name=self.model,
                        detector_backend=self.detector,
                        enforce_detection=True,
                    )
                if not reps or "embedding" not in reps[0]:
                    print(f"[FACE_DB] Warning: No face detected in {rel}, skipping.")
                    continue
                vec = np.array(reps[0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
                new_npz[rel] = vec
                new_entries[rel] = {
                    "person": person_key,
                    "size": sz,
                    "mtime_ns": mtime,
                }
            except Exception as exc:
                print(f"[FACE_DB] Skipped {rel} (detection/embed error: {exc})")

        # Persist updated cache
        try:
            manifest_data = {
                "model": self.model,
                "detector": self.detector,
                "version": 1,
                "entries": new_entries,
            }
            tmp_manifest = self._manifest_path.with_suffix(".tmp")
            tmp_manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
            tmp_manifest.replace(self._manifest_path)

            tmp_npz = self._npz_path.with_suffix(".tmp.npz")
            np.savez_compressed(str(tmp_npz), **new_npz)
            if self._npz_path.exists():
                try:
                    self._npz_path.unlink()
                except Exception:
                    pass
            tmp_npz.replace(self._npz_path)
        except Exception as exc:
            print(f"[FACE_DB] Failed to persist cache: {exc}")

        # Assemble in-memory unit matrix
        items = sorted(new_entries.items(), key=lambda kv: kv[0])
        labels = [kv[1]["person"] for kv in items if kv[0] in new_npz]
        vecs = [new_npz[kv[0]] for kv in items if kv[0] in new_npz]

        with self._swap_lock:
            if vecs:
                self._E = np.vstack(vecs).astype(np.float32)
                self._labels = labels
                unique_labels = sorted(set(labels))
                self._idx = {lbl: np.where(np.array(labels) == lbl)[0] for lbl in unique_labels}
            else:
                self._E = None
                self._labels = []
                self._idx = {}

        n_people = len(self._idx)
        n_photos = len(self._labels)
        print(f"[FACE_DB] Ready. {n_people} personnel registered ({n_photos} photos indexed).")
        return n_people, n_photos

    def warmup(self) -> None:
        """Pre-warm model weights on dummy canvas to prevent runtime stalls."""
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            from deepface import DeepFace
            with self._model_lock:
                DeepFace.represent(
                    img_path=dummy,
                    model_name=self.model,
                    detector_backend="skip",
                    enforce_detection=False,
                )
            print(f"[FACE_DB] Warmup complete for {self.model}.")
        except Exception as exc:
            print(f"[FACE_DB] Warmup notice: {exc}")

    def embed(self, face_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract face embedding and bounding box using YuNet detector."""
        from deepface import DeepFace
        with self._model_lock:
            try:
                reps = DeepFace.represent(
                    img_path=face_bgr,
                    model_name=self.model,
                    detector_backend=self.detector,
                    enforce_detection=True,
                )
            except Exception as exc:
                raise NoFaceError(f"No face detected: {exc}")

        if not reps:
            raise NoFaceError("No face detected in crop")

        rep = reps[0]
        vec = np.array(rep["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        area = rep.get("facial_area", {})
        fx = int(area.get("x", 0))
        fy = int(area.get("y", 0))
        fw = int(area.get("w", 0))
        fh = int(area.get("h", 0))
        return vec, (fx, fy, fw, fh)

    def verify(self, crop_bgr: np.ndarray) -> VerifyResult:
        """Verify face crop against in-memory database using Top-K cosine similarity."""
        if crop_bgr is None or crop_bgr.size == 0:
            return VerifyResult(kind="NO_FACE", error="Empty crop")

        try:
            q, (fx, fy, fw, fh) = self.embed(crop_bgr)
        except NoFaceError:
            return VerifyResult(kind="NO_FACE")
        except Exception as exc:
            return VerifyResult(kind="ERROR", error=str(exc))

        face_px = max(fw, fh)
        if face_px < MIN_FACE_PX:
            return VerifyResult(kind="LOW_QUALITY", face_px=face_px)

        with self._swap_lock:
            E = self._E
            idx_map = dict(self._idx)

        if E is None or len(E) == 0 or not idx_map:
            return VerifyResult(kind="NO_MATCH", distance=1.0, margin=1.0, face_px=face_px)

        # Cosine distance: 1.0 - (E @ q)
        sims = np.dot(E, q)
        dists = 1.0 - sims

        person_scores: List[Tuple[float, str]] = []
        for person, row_indices in idx_map.items():
            p_dists = dists[row_indices]
            p_dists.sort()
            k = min(len(p_dists), FACE_TOPK)
            best_k = p_dists[:k]
            score = float(np.mean(best_k))
            person_scores.append((score, person))

        person_scores.sort(key=lambda t: t[0])
        best_dist, best_person = person_scores[0]
        runner_up_dist = person_scores[1][0] if len(person_scores) > 1 else 1.0
        margin = runner_up_dist - best_dist

        if best_dist < FACE_DISTANCE_THRESH:
            if len(person_scores) > 1 and margin < FACE_MATCH_MARGIN:
                print(f"[FACE_DB] Margin reject: {best_person} dist={best_dist:.3f} vs runner-up={runner_up_dist:.3f} (margin={margin:.3f})")
                return VerifyResult(kind="NO_MATCH", distance=best_dist, margin=margin, face_px=face_px)
            return VerifyResult(kind="MATCH", name=best_person, distance=best_dist, margin=margin, face_px=face_px)

        return VerifyResult(kind="NO_MATCH", distance=best_dist, margin=margin, face_px=face_px)

    def signature(self) -> Tuple[int, int]:
        """Fast fingerprint for folder mtime + file count."""
        try:
            if not self.known_dir.exists():
                return 0, 0
            cnt = 0
            latest = 0
            for item in self.known_dir.iterdir():
                if item.is_dir():
                    for f in item.iterdir():
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                            cnt += 1
                            latest = max(latest, f.stat().st_mtime_ns)
                elif item.is_file() and item.suffix.lower() in IMAGE_EXTS:
                    cnt += 1
                    latest = max(latest, item.stat().st_mtime_ns)
            return cnt, latest
        except Exception:
            return 0, 0


# ===========================================================================
# 6. BUILT-IN PERSONNEL FACE ENROLLMENT STUDIO (FaceEnrollmentDialog)
# ===========================================================================
class FaceEnrollmentDialog:
    """Integrated Face Enrollment Studio modal with biometric validation."""

    def __init__(
        self,
        master: Optional[tk.Widget] = None,
        face_db: Optional[FaceDB] = None,
        on_close: Optional[Callable[[], None]] = None,
    ):
        self.master = master
        self.face_db = face_db or FaceDB(known_dir=KNOWN_FACES_DIR, cache_dir=CACHE_DIR)
        self.on_close_cb = on_close

        if master is None:
            self.window = tk.Tk()
            self._is_toplevel = False
        else:
            self.window = tk.Toplevel(master)
            self._is_toplevel = True

        self.window.title("Personnel Enrollment Studio")
        self.window.geometry("1120x740")
        self.window.minsize(980, 660)
        self.window.configure(bg=THEME["bg_main"])
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running: bool = False
        self.cam_thread: Optional[threading.Thread] = None
        self.current_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.status_queue: queue.Queue = queue.Queue()
        self.capture_result_queue: queue.Queue = queue.Queue()
        self.current_tk_image = None
        self.flash_alpha: float = 0.0

        self.pose_idx: int = 0
        self.last_saved_vec_by_person: Dict[str, np.ndarray] = {}

        self._configure_styles()
        self._build_ui()
        self._poll_queues()
        self._start_camera()
        self._refresh_enrolled_list()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Enroll.TEntry",
            fieldbackground=THEME["bg_input"],
            foreground=THEME["text_primary"],
            insertcolor=THEME["text_primary"],
            borderwidth=1,
            relief="solid",
        )

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self.window, bg=THEME["bg_surface"], height=64, padx=24)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        header_left = tk.Frame(header, bg=THEME["bg_surface"])
        header_left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            header_left,
            text="Personnel Enrollment Studio",
            font=("Segoe UI", 15, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_surface"],
        ).pack(side=tk.LEFT, pady=16)

        tk.Label(
            header_left,
            text=" |  Authorized Face Database Management",
            font=("Segoe UI", 11),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
        ).pack(side=tk.LEFT, pady=16)

        btn_exit = tk.Button(
            header,
            text="✕  Exit Studio",
            font=("Segoe UI", 10, "bold"),
            bg=THEME["neutral_btn"],
            fg=THEME["text_primary"],
            activebackground=THEME["danger"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=6,
            command=self.close,
        )
        btn_exit.pack(side=tk.RIGHT, pady=14)

        # Content view
        content = tk.Frame(self.window, bg=THEME["bg_main"], padx=24, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        # Left Panel (Camera Preview)
        left_panel = tk.Frame(content, bg=THEME["bg_surface"], padx=16, pady=16)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        cam_hdr = tk.Frame(left_panel, bg=THEME["bg_surface"])
        cam_hdr.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            cam_hdr,
            text="Live Camera Preview",
            font=("Segoe UI", 12, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_surface"],
        ).pack(side=tk.LEFT)

        self.lbl_cam_indicator = tk.Label(
            cam_hdr,
            text="● CONNECTING...",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
        )
        self.lbl_cam_indicator.pack(side=tk.RIGHT)

        self.video_container = tk.Frame(
            left_panel,
            bg="#020617",
            highlightbackground=THEME["border"],
            highlightthickness=1,
        )
        self.video_container.pack(fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(
            self.video_container,
            bg="#020617",
            highlightthickness=0,
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        self.lbl_pose_prompt = tk.Label(
            left_panel,
            text=f"Suggested Pose: {POSE_PROMPTS[0]}",
            font=("Segoe UI", 10, "bold"),
            fg="#38bdf8",
            bg=THEME["bg_surface"],
            pady=8,
        )
        self.lbl_pose_prompt.pack(fill=tk.X)

        # Right Panel (Controls & Directory)
        right_panel = tk.Frame(content, bg=THEME["bg_main"], width=390)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)

        capture_card = tk.Frame(right_panel, bg=THEME["bg_surface"], padx=18, pady=18)
        capture_card.pack(fill=tk.X, pady=(0, 16))

        tk.Label(
            capture_card,
            text="PERSONNEL DETAILS",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
        ).pack(anchor=tk.W)

        tk.Label(
            capture_card,
            text="Full Name / Identifier:",
            font=("Segoe UI", 10),
            fg=THEME["text_primary"],
            bg=THEME["bg_surface"],
        ).pack(anchor=tk.W, pady=(8, 4))

        self.name_entry = ttk.Entry(
            capture_card,
            font=("Segoe UI", 11),
            style="Enroll.TEntry",
        )
        self.name_entry.pack(fill=tk.X, pady=(0, 8))
        self.name_entry.focus()
        self.name_entry.bind("<KeyRelease>", self._on_name_changed)

        self.lbl_person_stats = tk.Label(
            capture_card,
            text=f"Enter name to see progress (0 / {ENROLL_RECOMMENDED_COUNT} recommended).",
            font=("Segoe UI", 9),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
            anchor=tk.W,
            wraplength=345,
            justify=tk.LEFT,
        )
        self.lbl_person_stats.pack(fill=tk.X, pady=(0, 12))

        self.btn_capture = tk.Button(
            capture_card,
            text="📸  Capture Photo",
            font=("Segoe UI", 11, "bold"),
            bg=THEME["primary"],
            fg=THEME["text_primary"],
            activebackground=THEME["primary_hover"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            pady=10,
            command=self.capture_single_photo,
        )
        self.btn_capture.pack(fill=tk.X)

        self.lbl_capture_feedback = tk.Label(
            capture_card,
            text="",
            font=("Segoe UI", 9),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
            wraplength=345,
            justify=tk.LEFT,
        )
        self.lbl_capture_feedback.pack(fill=tk.X, pady=(10, 0))

        # Enrolled Directory List
        list_card = tk.Frame(right_panel, bg=THEME["bg_surface"], padx=18, pady=18)
        list_card.pack(fill=tk.BOTH, expand=True)

        list_header = tk.Frame(list_card, bg=THEME["bg_surface"])
        list_header.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            list_header,
            text="ENROLLED DIRECTORY",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
        ).pack(side=tk.LEFT)

        self.lbl_total_enrolled = tk.Label(
            list_header,
            text="0 People",
            font=("Segoe UI", 9),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
        )
        self.lbl_total_enrolled.pack(side=tk.RIGHT)

        list_container = tk.Frame(
            list_card,
            bg=THEME["bg_main"],
            highlightbackground=THEME["border"],
            highlightthickness=1,
        )
        list_container.pack(fill=tk.BOTH, expand=True)

        self.personnel_listbox = tk.Listbox(
            list_container,
            font=("Segoe UI", 10),
            bg=THEME["bg_main"],
            fg=THEME["text_primary"],
            selectbackground=THEME["bg_card"],
            selectforeground=THEME["text_primary"],
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
        )
        self.personnel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.personnel_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        scrollbar = ttk.Scrollbar(
            list_container,
            orient=tk.VERTICAL,
            command=self.personnel_listbox.yview,
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.personnel_listbox.config(yscrollcommand=scrollbar.set)

        list_actions = tk.Frame(list_card, bg=THEME["bg_surface"])
        list_actions.pack(fill=tk.X, pady=(12, 0))

        tk.Button(
            list_actions,
            text="Delete Selected",
            font=("Segoe UI", 9),
            bg=THEME["danger"],
            fg=THEME["text_primary"],
            activebackground=THEME["danger_hover"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            pady=6,
            command=self.delete_selected_person,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        tk.Button(
            list_actions,
            text="Refresh List",
            font=("Segoe UI", 9),
            bg=THEME["neutral_btn"],
            fg=THEME["text_primary"],
            activebackground=THEME["neutral_hover"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            pady=6,
            command=self._refresh_enrolled_list,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 0))

    def _poll_queues(self) -> None:
        try:
            while not self.status_queue.empty():
                text, color = self.status_queue.get_nowait()
                self.lbl_cam_indicator.config(text=text, fg=color)
        except Exception:
            pass

        try:
            while not self.capture_result_queue.empty():
                ok, msg, saved_name = self.capture_result_queue.get_nowait()
                self.btn_capture.config(state=tk.NORMAL, text="📸  Capture Photo")
                if ok:
                    self.flash_alpha = 0.70
                    self.pose_idx = (self.pose_idx + 1) % len(POSE_PROMPTS)
                    self.lbl_pose_prompt.config(text=f"Suggested Pose: {POSE_PROMPTS[self.pose_idx]}")
                    self.lbl_capture_feedback.config(text=msg, fg=THEME["success"])
                    self._on_name_changed()
                    self._refresh_enrolled_list()
                else:
                    self.lbl_capture_feedback.config(text=msg, fg=THEME["danger"])
        except Exception:
            pass

        try:
            latest = None
            while not self.frame_queue.empty():
                latest = self.frame_queue.get_nowait()

            if latest is not None and self.video_canvas.winfo_exists():
                display_frame = latest.copy()
                if self.flash_alpha > 0.0:
                    white = np.full_like(display_frame, 255)
                    cv2.addWeighted(white, self.flash_alpha, display_frame, 1.0 - self.flash_alpha, 0, display_frame)
                    self.flash_alpha = max(0.0, self.flash_alpha - 0.25)

                cw = max(self.video_canvas.winfo_width(), 320)
                ch = max(self.video_canvas.winfo_height(), 240)
                fh, fw = display_frame.shape[:2]
                scale = min(cw / fw, ch / fh)
                dw = max(int(fw * scale), 1)
                dh = max(int(fh * scale), 1)
                ox = (cw - dw) // 2
                oy = (ch - dh) // 2

                rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)
                canvas_img = np.full((ch, cw, 3), 2, dtype=np.uint8)
                canvas_img[oy : oy + dh, ox : ox + dw] = resized

                pil_img = Image.fromarray(canvas_img)
                self.current_tk_image = ImageTk.PhotoImage(image=pil_img, master=self.video_canvas)
                self.video_canvas.delete("all")
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_tk_image)
        except Exception:
            pass

        if self.window.winfo_exists():
            self.window.after(30, self._poll_queues)

    def _start_camera(self) -> None:
        self.is_camera_running = True
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

    def _camera_loop(self) -> None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap = cap
        if not cap.isOpened():
            self.status_queue.put(("● CAMERA OFFLINE", THEME["danger"]))
            return

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.status_queue.put(("● LIVE CAMERA", THEME["success"]))

        while self.is_camera_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret or not self.is_camera_running or frame is None:
                time.sleep(0.03)
                continue

            with self.lock:
                self.current_frame = frame.copy()

            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass

    def _sanitize_name(self, raw_name: str) -> str:
        trimmed = raw_name.strip()
        safe = "".join(c for c in trimmed if c.isalnum() or c in (" ", "_", "-")).strip()
        safe = re.sub(r"\s+", "_", safe)
        return safe

    def _resolve_person_dir_name(self, safe_name: str) -> str:
        if not safe_name:
            return ""
        if KNOWN_FACES_DIR.exists():
            lower_target = safe_name.lower()
            for p in KNOWN_FACES_DIR.iterdir():
                if p.is_dir() and p.name.lower() == lower_target:
                    return p.name
        return safe_name

    def _is_valid_name(self, safe_name: str) -> bool:
        if not safe_name:
            return False
        if safe_name.upper() in WINDOWS_RESERVED_NAMES:
            return False
        return True

    def _get_person_dir(self, folder_name: str) -> Path:
        return KNOWN_FACES_DIR / folder_name

    def _get_existing_photos(self, folder_name: str) -> List[Path]:
        pdir = self._get_person_dir(folder_name)
        if not pdir.is_dir():
            return []
        return sorted(
            [f for f in pdir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.name.lower(),
        )

    def _next_photo_index(self, folder_name: str) -> int:
        existing = self._get_existing_photos(folder_name)
        max_idx = 0
        pattern = re.compile(rf"^{re.escape(folder_name)}_(\d+)$", re.IGNORECASE)
        for f in existing:
            m = pattern.match(f.stem)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    def _on_name_changed(self, event=None) -> None:
        raw = self.name_entry.get().strip()
        if not raw:
            self.lbl_person_stats.config(
                text=f"Enter name to see progress (0 / {ENROLL_RECOMMENDED_COUNT} recommended).",
                fg=THEME["text_secondary"],
            )
            return

        safe = self._sanitize_name(raw)
        if not self._is_valid_name(safe):
            self.lbl_person_stats.config(text="Invalid name (reserved or empty).", fg=THEME["danger"])
            return

        resolved = self._resolve_person_dir_name(safe)
        count = len(self._get_existing_photos(resolved))
        rec = ENROLL_RECOMMENDED_COUNT

        if count > 25:
            hint = f"Folder '{resolved}': {count} photos (10-15 varied photos are enough)."
        elif count > 0:
            hint = f"Folder '{resolved}': {count} / {rec} recommended. Clicks will append."
        else:
            hint = f"New person '{resolved}': 0 / {rec} recommended."
        self.lbl_person_stats.config(text=hint, fg=THEME["text_secondary"])

    def capture_single_photo(self) -> None:
        raw_name = self.name_entry.get().strip()
        safe_name = self._sanitize_name(raw_name)
        if not self._is_valid_name(safe_name):
            messagebox.showwarning("Valid Name Required", "Please enter a valid personnel name before capturing.")
            return

        with self.lock:
            if self.current_frame is None:
                messagebox.showerror("Error", "No camera frame available to capture.")
                return
            frame_to_save = self.current_frame.copy()

        resolved_name = self._resolve_person_dir_name(safe_name)
        self.btn_capture.config(state=tk.DISABLED, text="Checking photo...")
        self.lbl_capture_feedback.config(text="Validating face clarity and uniqueness...", fg=THEME["text_secondary"])

        threading.Thread(
            target=self._validate_and_save_worker,
            args=(frame_to_save, resolved_name),
            daemon=True,
        ).start()

    def _validate_and_save_worker(self, frame_bgr: np.ndarray, folder_name: str) -> None:
        try:
            from deepface import DeepFace

            try:
                faces = DeepFace.extract_faces(
                    img_path=frame_bgr,
                    detector_backend=FACE_DETECTOR,
                    enforce_detection=True,
                )
            except Exception:
                self.capture_result_queue.put((False, "⚠️ No clear face detected - face the camera directly.", None))
                return

            if not faces:
                self.capture_result_queue.put((False, "⚠️ No clear face detected - face the camera directly.", None))
                return
            if len(faces) > 1:
                self.capture_result_queue.put((False, "⚠️ Multiple faces detected - only 1 person should be visible.", None))
                return

            face_info = faces[0]
            conf = float(face_info.get("confidence", 1.0))
            if conf < ENROLL_MIN_CONFIDENCE:
                self.capture_result_queue.put((False, f"⚠️ Low detection confidence ({conf:.2f}) - improve lighting.", None))
                return

            area = face_info.get("facial_area", {})
            fx, fy, fw, fh = int(area.get("x", 0)), int(area.get("y", 0)), int(area.get("w", 0)), int(area.get("h", 0))
            if fw < ENROLL_MIN_FACE_PX:
                self.capture_result_queue.put((False, f"⚠️ Face too small ({fw}px) - move closer to camera.", None))
                return

            # Sharpness test
            ih, iw = frame_bgr.shape[:2]
            fx1, fy1 = max(0, fx), max(0, fy)
            fx2, fy2 = min(iw, fx + fw), min(ih, fy + fh)
            tight_face = frame_bgr[fy1:fy2, fx1:fx2]
            gray_face = cv2.cvtColor(tight_face, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
            if sharpness < ENROLL_MIN_SHARPNESS:
                self.capture_result_queue.put((False, f"⚠️ Image blurry (sharpness {sharpness:.0f} < {ENROLL_MIN_SHARPNESS:.0f}) - hold still.", None))
                return

            # 40% padded face crop
            pad_x = int(fw * 0.40)
            pad_y = int(fh * 0.40)
            cx1, cy1 = max(0, fx - pad_x), max(0, fy - pad_y)
            cx2, cy2 = min(iw, fx + fw + pad_x), min(ih, fy + fh + pad_y)
            crop_to_save = frame_bgr[cy1:cy2, cx1:cx2]

            save_img = crop_to_save
            try:
                vec, _ = self.face_db.embed(crop_to_save)
            except Exception:
                save_img = frame_bgr
                vec, _ = self.face_db.embed(frame_bgr)

            # Duplicate guard
            prev_vec = self.last_saved_vec_by_person.get(folder_name.lower())
            if prev_vec is not None:
                sim = float(prev_vec @ vec)
                if sim > ENROLL_MAX_SIMILARITY:
                    self.capture_result_queue.put((False, "⚠️ Too similar to previous photo - change angle or expression.", None))
                    return

            # Save & sync cache
            person_dir = self._get_person_dir(folder_name)
            person_dir.mkdir(parents=True, exist_ok=True)
            next_idx = self._next_photo_index(folder_name)
            save_filename = f"{folder_name}_{next_idx}.jpg"
            save_path = person_dir / save_filename
            cv2.imwrite(str(save_path), save_img)

            self.last_saved_vec_by_person[folder_name.lower()] = vec
            self.face_db.reload()

            total_photos = len(self._get_existing_photos(folder_name))
            msg = f"✓ Saved #{next_idx} → {save_filename} ({total_photos} total in '{folder_name}')."
            self.capture_result_queue.put((True, msg, folder_name))

        except Exception as exc:
            self.capture_result_queue.put((False, f"⚠️ Validation error: {exc}", None))

    def _refresh_enrolled_list(self) -> None:
        self.personnel_listbox.delete(0, tk.END)
        if not KNOWN_FACES_DIR.exists():
            return

        total_people = 0
        for entry in sorted(KNOWN_FACES_DIR.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_dir():
                img_count = len([f for f in entry.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
                self.personnel_listbox.insert(tk.END, f"{entry.name}  ({img_count} photos)")
                total_people += 1
            elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                self.personnel_listbox.insert(tk.END, f"{entry.stem}  (1 photo)")
                total_people += 1

        self.lbl_total_enrolled.config(text=f"{total_people} People")

    def _on_list_select(self, event=None) -> None:
        sel = self.personnel_listbox.curselection()
        if not sel:
            return
        item_text = self.personnel_listbox.get(sel[0])
        folder_name = item_text.split("  (")[0].strip()
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, folder_name)
        self._on_name_changed()

    def delete_selected_person(self) -> None:
        sel = self.personnel_listbox.curselection()
        if not sel:
            messagebox.showinfo("Select Person", "Please select a person from the list to remove.")
            return

        item_text = self.personnel_listbox.get(sel[0])
        person_name = item_text.split("  (")[0].strip()
        target_dir = KNOWN_FACES_DIR / person_name
        target_file = KNOWN_FACES_DIR / f"{person_name}.jpg"

        if messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to permanently delete all enrolled photos for '{person_name}'?",
        ):
            if target_dir.is_dir():
                shutil.rmtree(target_dir, ignore_errors=True)
            if target_file.is_file():
                target_file.unlink()
            self.last_saved_vec_by_person.pop(person_name.lower(), None)

            threading.Thread(target=self.face_db.reload, daemon=True).start()
            self._refresh_enrolled_list()
            self.lbl_capture_feedback.config(
                text=f"Deleted '{person_name}' and scheduled cache refresh.",
                fg=THEME["text_secondary"],
            )
            self._on_name_changed()

    def close(self) -> None:
        """Release camera hardware cleanly, trigger callback and destroy window."""
        self.is_camera_running = False
        if self.cam_thread and self.cam_thread.is_alive():
            self.cam_thread.join(timeout=0.5)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.on_close_cb:
            try:
                self.on_close_cb()
            except Exception as e:
                print(f"[STUDIO] On-close callback error: {e}")

        self.window.destroy()


# ===========================================================================
# 7. MAIN AI SURVEILLANCE & THREAT MONITOR DASHBOARD
# ===========================================================================
class SurveillanceGUIApp:
    """Core graphical dashboard orchestrating live surveillance, tracking, and alerts."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Surveillance & Security Threat Monitor")
        self.root.geometry("1280x820")
        self.root.minsize(1100, 720)
        self.root.configure(bg=THEME["bg_main"])

        self._ensure_directories()

        # Camera & Stream state
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.video_thread: Optional[threading.Thread] = None
        self._enroll_dialog: Optional[FaceEnrollmentDialog] = None

        # Letterboxed canvas dimensions
        self.canvas_w: int = 780
        self.canvas_h: int = 560
        self.frame_orig_w: int = 640
        self.frame_orig_h: int = 480

        # Tripwire ROI coordinates in native frame space: [x1, y1, x2, y2]
        self.roi_frame: List[int] = [100, 60, 540, 420]
        self.is_drawing_roi: bool = False
        self.roi_drag_start: Tuple[int, int] = (0, 0)
        self.temp_drag_roi: Optional[List[int]] = None

        # Models & Biometrics
        self.model = None
        self.face_db = FaceDB(
            known_dir=KNOWN_FACES_DIR,
            cache_dir=CACHE_DIR,
            model=FACE_MODEL,
            detector=FACE_DETECTOR,
        )
        self._last_db_sig: Tuple[int, int] = self.face_db.signature()
        self._last_sig_check: float = 0.0
        self._db_reloading: bool = False

        # Tracking state
        self.person_tracker_db: Dict[int, Dict[str, Any]] = {}
        self.total_breaches: int = 0
        self.fps: float = 0.0
        self._overcrowd_since: Optional[float] = None
        self._last_overcrowd_alert: float = 0.0
        self._last_after_hours_alert: float = 0.0

        # Queues
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.log_queue: queue.Queue = queue.Queue()
        self.status_queue: queue.Queue = queue.Queue()
        self.verify_job_queue: queue.Queue = queue.Queue(maxsize=8)
        self.verify_result_queue: queue.Queue = queue.Queue()
        self.current_tk_image = None
        self._verify_error_streak: int = 0

        # Alert Manager
        self.alert_manager = AlertManager(
            outputs_dir=OUTPUTS_DIR,
            log_queue=self.log_queue,
        )

        self._configure_styles()
        self._build_ui()
        self._start_verify_worker()
        self._start_polling_loop()
        self._load_models_async()
        self.root.after(50, self._render_standby_screen)

    def _ensure_directories(self) -> None:
        for d in [KNOWN_FACES_DIR, OUTPUTS_DIR, CACHE_DIR, ASSETS_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Custom.Vertical.TScrollbar",
            background=THEME["bg_surface"],
            troughcolor=THEME["bg_main"],
            bordercolor=THEME["border"],
            arrowcolor=THEME["text_muted"],
            borderwidth=0,
            arrowsize=12,
        )

    def _build_ui(self) -> None:
        # Header Bar
        header = tk.Frame(self.root, bg=THEME["bg_surface"], height=64, padx=24)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        header_left = tk.Frame(header, bg=THEME["bg_surface"])
        header_left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            header_left,
            text="AI Surveillance System",
            font=("Segoe UI", 15, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_surface"],
        ).pack(side=tk.LEFT, pady=16)

        tk.Label(
            header_left,
            text=" |  Autonomous Anomaly Detection & Threat Monitoring",
            font=("Segoe UI", 11),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
        ).pack(side=tk.LEFT, pady=16)

        header_right = tk.Frame(header, bg=THEME["bg_surface"])
        header_right.pack(side=tk.RIGHT, fill=tk.Y)

        self.lbl_header_status = tk.Label(
            header_right,
            text="● LOADING MODELS...",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
            padx=16,
        )
        self.lbl_header_status.pack(side=tk.LEFT, pady=18)

        btn_exit = tk.Button(
            header_right,
            text="✕  Exit Dashboard",
            font=("Segoe UI", 10, "bold"),
            bg=THEME["neutral_btn"],
            fg=THEME["text_primary"],
            activebackground=THEME["danger"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=6,
            command=self.close,
        )
        btn_exit.pack(side=tk.LEFT, pady=14)

        # Content Split View
        content = tk.Frame(self.root, bg=THEME["bg_main"], padx=20, pady=18)
        content.pack(fill=tk.BOTH, expand=True)

        # Left Panel (Live Canvas)
        left_panel = tk.Frame(content, bg=THEME["bg_surface"], padx=16, pady=16)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        cam_hdr = tk.Frame(left_panel, bg=THEME["bg_surface"])
        cam_hdr.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            cam_hdr,
            text="Live Surveillance Stream",
            font=("Segoe UI", 12, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_surface"],
        ).pack(side=tk.LEFT)

        self.lbl_aspect_info = tk.Label(
            cam_hdr,
            text="Aspect Ratio: Preserved | Drag on feed to set Tripwire ROI",
            font=("Segoe UI", 9),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
        )
        self.lbl_aspect_info.pack(side=tk.RIGHT)

        self.video_container = tk.Frame(
            left_panel,
            bg="#020617",
            highlightbackground=THEME["border"],
            highlightthickness=1,
        )
        self.video_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.video_container,
            bg="#020617",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Bottom Toolbar below Canvas
        vtool_frame = tk.Frame(left_panel, bg=THEME["bg_surface"])
        vtool_frame.pack(fill=tk.X, pady=(12, 0))

        self.btn_toggle = tk.Button(
            vtool_frame,
            text="▶  Start Surveillance",
            font=("Segoe UI", 10, "bold"),
            bg=THEME["success"],
            fg=THEME["text_primary"],
            activebackground=THEME["success_hover"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            padx=18,
            pady=8,
            command=self.toggle_surveillance,
        )
        self.btn_toggle.pack(side=tk.LEFT, padx=(0, 10))

        btn_reset_roi = tk.Button(
            vtool_frame,
            text="↺ Reset Tripwire ROI",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["neutral_btn"],
            fg=THEME["text_primary"],
            activebackground=THEME["neutral_hover"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            padx=14,
            pady=8,
            command=self.reset_roi,
        )
        btn_reset_roi.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_roi_coords = tk.Label(
            vtool_frame,
            text=f"Tripwire: ({self.roi_frame[0]},{self.roi_frame[1]}) - ({self.roi_frame[2]},{self.roi_frame[3]})",
            font=("Segoe UI", 9),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
        )
        self.lbl_roi_coords.pack(side=tk.LEFT)

        # Right Panel (Controls, Alert Channels, Telemetry & Logs)
        right_panel = tk.Frame(content, bg=THEME["bg_main"], width=410)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)

        # Card 1: Input & Personnel Access
        source_card = tk.Frame(right_panel, bg=THEME["bg_surface"], padx=16, pady=12)
        source_card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            source_card,
            text="VIDEO INPUT & PERSONNEL ACCESS",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
        ).pack(anchor=tk.W)

        cam_status_box = tk.Frame(
            source_card,
            bg=THEME["bg_input"],
            padx=12,
            pady=9,
            highlightbackground=THEME["border"],
            highlightthickness=1,
        )
        cam_status_box.pack(fill=tk.X, pady=(8, 10))

        tk.Label(
            cam_status_box,
            text=f"🎥  Live Camera Feed (Device {CAMERA_INDEX})",
            font=("Segoe UI", 10, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_input"],
        ).pack(side=tk.LEFT)

        self.lbl_cam_live_tag = tk.Label(
            cam_status_box,
            text="● STANDBY",
            font=("Segoe UI", 8, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_input"],
        )
        self.lbl_cam_live_tag.pack(side=tk.RIGHT)

        btn_grid = tk.Frame(source_card, bg=THEME["bg_surface"])
        btn_grid.pack(fill=tk.X)

        btn_enroll = tk.Button(
            btn_grid,
            text="👤 Face Enrollment Studio",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["primary"],
            fg=THEME["text_primary"],
            activebackground=THEME["primary_hover"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            pady=6,
            command=self.launch_enrollment_gui,
        )
        btn_enroll.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        btn_reload_db = tk.Button(
            btn_grid,
            text="🔄 Refresh DB",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["neutral_btn"],
            fg=THEME["text_primary"],
            activebackground=THEME["neutral_hover"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            pady=6,
            command=lambda: self.refresh_face_database(manual=True),
        )
        btn_reload_db.pack(side=tk.RIGHT, padx=(6, 0))

        # Card 2: Incident & Audit Channels
        alert_card = tk.Frame(right_panel, bg=THEME["bg_surface"], padx=16, pady=10)
        alert_card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            alert_card,
            text="INCIDENT & AUDIT CHANNELS",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
        ).pack(anchor=tk.W, pady=(0, 6))

        alert_row = tk.Frame(alert_card, bg=THEME["bg_surface"])
        alert_row.pack(fill=tk.X)

        tk.Label(
            alert_row,
            text="● Incident CSV & Snapshots Active",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["success"],
            bg=THEME["bg_surface"],
        ).pack(side=tk.LEFT)

        btn_test_alert = tk.Button(
            alert_row,
            text="Send Test Alert",
            font=("Segoe UI", 8, "bold"),
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            activebackground=THEME["neutral_btn"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            padx=10,
            pady=4,
            command=self._send_test_alert,
        )
        btn_test_alert.pack(side=tk.RIGHT)

        # Card 3: Telemetry
        metrics_card = tk.Frame(right_panel, bg=THEME["bg_surface"], padx=16, pady=10)
        metrics_card.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            metrics_card,
            text="SYSTEM TELEMETRY",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
        ).pack(anchor=tk.W, pady=(0, 6))

        stats_row = tk.Frame(metrics_card, bg=THEME["bg_surface"])
        stats_row.pack(fill=tk.X)

        fps_box = tk.Frame(stats_row, bg=THEME["bg_card"], padx=10, pady=6)
        fps_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Label(fps_box, text="FPS", font=("Segoe UI", 8), fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack()
        self.lbl_fps_val = tk.Label(fps_box, text="0.0", font=("Segoe UI", 12, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"])
        self.lbl_fps_val.pack()

        targets_box = tk.Frame(stats_row, bg=THEME["bg_card"], padx=10, pady=6)
        targets_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(targets_box, text="IN TRIPWIRE", font=("Segoe UI", 8), fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack()
        self.lbl_targets_val = tk.Label(targets_box, text="0", font=("Segoe UI", 12, "bold"), fg="#38bdf8", bg=THEME["bg_card"])
        self.lbl_targets_val.pack()

        breach_box = tk.Frame(stats_row, bg=THEME["bg_card"], padx=10, pady=6)
        breach_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        tk.Label(breach_box, text="BREACHES", font=("Segoe UI", 8), fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack()
        self.lbl_breach_val = tk.Label(breach_box, text="0", font=("Segoe UI", 12, "bold"), fg=THEME["danger"], bg=THEME["bg_card"])
        self.lbl_breach_val.pack()

        # Card 4: Incident Log
        log_card = tk.Frame(right_panel, bg=THEME["bg_surface"], padx=16, pady=12)
        log_card.pack(fill=tk.BOTH, expand=True)

        log_hdr = tk.Frame(log_card, bg=THEME["bg_surface"])
        log_hdr.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            log_hdr,
            text="SECURITY ALERTS & INCIDENT LOG",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
        ).pack(side=tk.LEFT)

        btn_clear_log = tk.Button(
            log_hdr,
            text="Clear",
            font=("Segoe UI", 8),
            bg=THEME["bg_card"],
            fg=THEME["text_secondary"],
            relief="flat",
            cursor="hand2",
            padx=8,
            command=lambda: self.log_listbox.delete(0, tk.END),
        )
        btn_clear_log.pack(side=tk.RIGHT)

        log_container = tk.Frame(
            log_card,
            bg=THEME["bg_input"],
            highlightbackground=THEME["border"],
            highlightthickness=1,
        )
        log_container.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.log_listbox = tk.Listbox(
            log_container,
            font=("Consolas", 9),
            bg=THEME["bg_input"],
            fg="#fca5a5",
            selectbackground=THEME["bg_card"],
            selectforeground=THEME["text_primary"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.log_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        log_scroll = ttk.Scrollbar(
            log_container,
            orient=tk.VERTICAL,
            command=self.log_listbox.yview,
            style="Custom.Vertical.TScrollbar",
        )
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_listbox.config(yscrollcommand=log_scroll.set)

        out_btn_row = tk.Frame(log_card, bg=THEME["bg_surface"])
        out_btn_row.pack(fill=tk.X)

        btn_open_outputs = tk.Button(
            out_btn_row,
            text="📂  Open Snapshots Folder",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            activebackground=THEME["neutral_btn"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            pady=7,
            command=self._open_outputs_dir,
        )
        btn_open_outputs.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_open_csv = tk.Button(
            out_btn_row,
            text="📄  Open CSV Log",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            activebackground=THEME["neutral_btn"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            pady=7,
            padx=10,
            command=self._open_incidents_csv,
        )
        btn_open_csv.pack(side=tk.RIGHT, padx=(4, 0))

    def _send_test_alert(self) -> None:
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        dummy[:] = (30, 41, 59)
        cv2.putText(dummy, "TEST ALERT SNAPSHOT", (28, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 250, 252), 2, cv2.LINE_AA)
        self.alert_manager.emit(
            "SYSTEM",
            0,
            "🔔 TEST ALERT: Incident logging & snapshot verified",
            snapshot_bgr=dummy,
            extra={"state": "TEST"},
        )

    def _on_canvas_resize(self, event) -> None:
        self.canvas_w = max(event.width, 100)
        self.canvas_h = max(event.height, 100)
        if not self.is_running:
            self._render_standby_screen()

    def _render_standby_screen(self) -> None:
        if not self.canvas.winfo_exists():
            return

        self.canvas.delete("all")
        self.current_tk_image = None

        cw = max(self.canvas.winfo_width(), self.canvas_w, 200)
        ch = max(self.canvas.winfo_height(), self.canvas_h, 200)
        cx, cy = cw // 2, ch // 2

        self.canvas.create_rectangle(0, 0, cw, ch, fill="#020617", outline="")
        for r in [220, 150, 80]:
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#1e293b", width=1, dash=(3, 5))
        self.canvas.create_line(cx - 260, cy, cx + 260, cy, fill="#1e293b", width=1, dash=(2, 4))
        self.canvas.create_line(cx, cy - 180, cx, cy + 180, fill="#1e293b", width=1, dash=(2, 4))

        card_w, card_h = 420, 175
        self.canvas.create_rectangle(
            cx - card_w // 2, cy - card_h // 2, cx + card_w // 2, cy + card_h // 2,
            fill="#0f172a", outline="#334155", width=1,
        )
        badge_w, badge_h = 180, 26
        self.canvas.create_rectangle(
            cx - badge_w // 2, cy - 50 - badge_h // 2, cx + badge_w // 2, cy - 50 + badge_h // 2,
            fill="#1e293b", outline="#475569", width=1,
        )
        self.canvas.create_text(cx, cy - 50, text="● SURVEILLANCE OFFLINE", fill="#94a3b8", font=("Segoe UI", 9, "bold"))
        self.canvas.create_text(cx, cy - 12, text="Live Video Feed Paused", fill="#f8fafc", font=("Segoe UI", 15, "bold"))
        self.canvas.create_text(cx, cy + 20, text='Click "▶ Start Surveillance" below to activate camera', fill="#94a3b8", font=("Segoe UI", 10))
        self.canvas.create_text(cx, cy + 48, text="Tripwire ROI and AI threat detection are on standby", fill="#64748b", font=("Segoe UI", 8))

    def _get_canvas_to_frame_scale(self) -> Tuple[float, int, int]:
        cw, ch = self.canvas_w, self.canvas_h
        fw, fh = self.frame_orig_w, self.frame_orig_h
        if fw <= 0 or fh <= 0 or cw <= 0 or ch <= 0:
            return 1.0, 0, 0
        scale = min(cw / fw, ch / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        offset_x = (cw - disp_w) // 2
        offset_y = (ch - disp_h) // 2
        return scale, offset_x, offset_y

    def _canvas_to_frame_pt(self, cx: int, cy: int) -> Tuple[int, int]:
        scale, off_x, off_y = self._get_canvas_to_frame_scale()
        fx = int((cx - off_x) / scale)
        fy = int((cy - off_y) / scale)
        return int(np.clip(fx, 0, self.frame_orig_w)), int(np.clip(fy, 0, self.frame_orig_h))

    def _frame_to_canvas_rect(self, rect: List[int]) -> Tuple[int, int, int, int]:
        scale, off_x, off_y = self._get_canvas_to_frame_scale()
        x1, y1, x2, y2 = rect
        return int(x1 * scale) + off_x, int(y1 * scale) + off_y, int(x2 * scale) + off_x, int(y2 * scale) + off_y

    def _on_mouse_down(self, event) -> None:
        self.is_drawing_roi = True
        self.roi_drag_start = (event.x, event.y)
        self.temp_drag_roi = None

    def _on_mouse_drag(self, event) -> None:
        if not self.is_drawing_roi:
            return
        x1, y1 = self.roi_drag_start
        x2, y2 = event.x, event.y
        fx1, fy1 = self._canvas_to_frame_pt(min(x1, x2), min(y1, y2))
        fx2, fy2 = self._canvas_to_frame_pt(max(x1, x2), max(y1, y2))
        self.temp_drag_roi = [fx1, fy1, fx2, fy2]
        self.lbl_roi_coords.config(text=f"Tripwire: ({fx1},{fy1}) - ({fx2},{fy2})")

    def _on_mouse_up(self, event) -> None:
        self.is_drawing_roi = False
        if self.temp_drag_roi:
            fx1, fy1, fx2, fy2 = self.temp_drag_roi
            if abs(fx2 - fx1) > 20 and abs(fy2 - fy1) > 20:
                self.roi_frame = [fx1, fy1, fx2, fy2]
            self.temp_drag_roi = None
        self.lbl_roi_coords.config(
            text=f"Tripwire: ({self.roi_frame[0]},{self.roi_frame[1]}) - ({self.roi_frame[2]},{self.roi_frame[3]})"
        )

    def reset_roi(self) -> None:
        w, h = self.frame_orig_w, self.frame_orig_h
        self.roi_frame = [int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)]
        self.lbl_roi_coords.config(
            text=f"Tripwire: ({self.roi_frame[0]},{self.roi_frame[1]}) - ({self.roi_frame[2]},{self.roi_frame[3]})"
        )

    def _start_polling_loop(self) -> None:
        try:
            while not self.log_queue.empty():
                self._add_log_entry(self.log_queue.get_nowait())
        except Exception:
            pass

        try:
            while not self.status_queue.empty():
                text, color = self.status_queue.get_nowait()
                self.lbl_header_status.config(text=text, fg=color)
        except Exception:
            pass

        # Periodic signature check for external directory changes
        now = time.time()
        if (now - self._last_sig_check >= 2.0) and not self._db_reloading:
            self._last_sig_check = now
            sig = self.face_db.signature()
            if sig != self._last_db_sig:
                self._last_db_sig = sig
                self.refresh_face_database(manual=False)

        try:
            latest_frame = None
            while not self.frame_queue.empty():
                latest_frame = self.frame_queue.get_nowait()

            if latest_frame is not None and self.canvas.winfo_exists() and self.is_running:
                canvas_img, fps_val, targets, breaches_val = latest_frame
                img_pil = Image.fromarray(canvas_img)
                self.current_tk_image = ImageTk.PhotoImage(image=img_pil, master=self.canvas)

                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_tk_image)

                active_roi = self.temp_drag_roi if self.temp_drag_roi else self.roi_frame
                cx1, cy1, cx2, cy2 = self._frame_to_canvas_rect(active_roi)
                dash_style = (4, 4) if self.temp_drag_roi else None

                self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#38bdf8", width=2, dash=dash_style)
                self.canvas.create_text(
                    cx1 + 6, cy1 + 12, text="Tripwire Zone", fill="#38bdf8", anchor=tk.W, font=("Segoe UI", 9, "bold")
                )

                self.lbl_fps_val.config(text=f"{fps_val:.1f}")
                self.lbl_targets_val.config(text=str(targets))
                self.lbl_breach_val.config(text=str(breaches_val))
        except Exception:
            pass

        if self.root.winfo_exists():
            self.root.after(16, self._start_polling_loop)

    def _load_models_async(self) -> None:
        def load():
            try:
                from ultralytics import YOLO

                try:
                    self.model = YOLO(PERSON_MODEL_PATH)
                    print(f"[INIT] Loaded pose tracker model: {PERSON_MODEL_PATH}")
                except Exception as pose_err:
                    print(f"[INIT] Pose model unavailable ({pose_err}), falling back to {PERSON_FALLBACK_MODEL_PATH}")
                    self.model = YOLO(PERSON_FALLBACK_MODEL_PATH)

                try:
                    self.face_db.warmup()
                    people, photos = self.face_db.reload()
                    self._last_db_sig = self.face_db.signature()
                    if photos == 0:
                        self.status_queue.put(("● SYSTEM READY (No Enrolled Faces)", THEME["warning"]))
                    else:
                        self.status_queue.put((f"● SYSTEM READY ({people} People)", THEME["success"]))
                except Exception as fe:
                    print(f"[ERROR] FaceDB warmup/load error: {fe}")
                    self.status_queue.put(("● RECOGNITION ERROR", THEME["danger"]))
            except Exception as e:
                self.status_queue.put(("● YOLO LOAD ERROR", THEME["danger"]))
                print(f"[ERROR] Failed to load YOLO: {e}")

        threading.Thread(target=load, daemon=True).start()

    def launch_enrollment_gui(self) -> None:
        """Open built-in Face Enrollment Studio modal."""
        if self.is_running:
            if not messagebox.askyesno(
                "Open Enrollment Studio",
                "The camera is currently running for live surveillance.\n\n"
                "Would you like to stop surveillance now so the camera can be opened in the Face Enrollment Studio?",
            ):
                return
            self.stop_surveillance()

        if self._enroll_dialog is not None and self._enroll_dialog.window.winfo_exists():
            self._enroll_dialog.window.lift()
            self._enroll_dialog.window.focus_force()
            return

        self._enroll_dialog = FaceEnrollmentDialog(
            master=self.root,
            face_db=self.face_db,
            on_close=lambda: self.refresh_face_database(manual=False),
        )

    def refresh_face_database(self, manual: bool = False) -> None:
        """Incrementally reload FaceDB."""
        if self._db_reloading:
            return
        self._db_reloading = True
        self.status_queue.put(("● RELOADING FACE DB...", THEME["warning"]))

        def worker():
            try:
                people, photos = self.face_db.reload()
                self._last_db_sig = self.face_db.signature()
                self.verify_result_queue.put((-1, VerifyResult(kind="RELOAD")))
                msg = f"Face database reloaded ({people} people, {photos} photos)"
                self.log_queue.put(f"[{time.strftime('%H:%M:%S')}] 🔄 SYSTEM: {msg}")
                if photos == 0:
                    self.status_queue.put(("● SYSTEM READY (No Enrolled Faces)", THEME["warning"]))
                elif self.is_running:
                    self.status_queue.put(("● LIVE MONITORING", THEME["success"]))
                else:
                    self.status_queue.put((f"● SYSTEM READY ({people} People)", THEME["success"]))
            except Exception as exc:
                print(f"[FACE_DB] Reload error: {exc}")
                self.status_queue.put(("● RECOGNITION ERROR", THEME["danger"]))
            finally:
                self._db_reloading = False

        threading.Thread(target=worker, daemon=True).start()

    def _open_outputs_dir(self) -> None:
        try:
            open_path_in_os(OUTPUTS_DIR)
        except Exception:
            messagebox.showinfo("Outputs Directory", f"Snapshots stored at: {OUTPUTS_DIR}")

    def _open_incidents_csv(self) -> None:
        csv_path = OUTPUTS_DIR / "incidents.csv"
        if not csv_path.exists():
            messagebox.showinfo("No Incidents Yet", "No incidents recorded yet (`incidents.csv` does not exist).")
            return
        try:
            open_path_in_os(csv_path)
        except Exception:
            messagebox.showinfo("CSV Path", f"Incident log at: {csv_path}")

    def toggle_surveillance(self) -> None:
        if self.is_running:
            self.stop_surveillance()
        else:
            self.start_surveillance()

    def start_surveillance(self) -> None:
        if self._enroll_dialog is not None and self._enroll_dialog.window.winfo_exists():
            messagebox.showinfo(
                "Enrollment Studio Active",
                "Please close the Face Enrollment Studio before starting live surveillance so the webcam can be accessed.",
            )
            return

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Connection Error",
                f"Unable to open live webcam (Device {CAMERA_INDEX}).\n\nPlease verify that your camera is connected and not in use by another app.",
            )
            return

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.is_running = True
        self.btn_toggle.config(text="⏹  Stop Surveillance", bg=THEME["danger"], activebackground=THEME["danger_hover"])
        self.lbl_header_status.config(text="● LIVE MONITORING", fg=THEME["success"])
        self.lbl_cam_live_tag.config(text="● LIVE", fg=THEME["success"])

        self.video_thread = threading.Thread(target=self._process_stream_loop, daemon=True)
        self.video_thread.start()

    def stop_surveillance(self) -> None:
        self.is_running = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=0.5)

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                pass

        self.btn_toggle.config(text="▶  Start Surveillance", bg=THEME["success"], activebackground=THEME["success_hover"])
        self.lbl_header_status.config(text="● STANDBY", fg=THEME["text_muted"])
        self.lbl_cam_live_tag.config(text="● STANDBY", fg=THEME["text_muted"])
        self.lbl_fps_val.config(text="0.0")
        self.lbl_targets_val.config(text="0")
        self._render_standby_screen()

    def _start_verify_worker(self) -> None:
        def worker():
            while True:
                job = self.verify_job_queue.get()
                if job is None:
                    break
                tid, crop_bgr = job
                res = self.face_db.verify(crop_bgr)
                self.verify_result_queue.put((tid, res))

        self._verify_thread = threading.Thread(target=worker, daemon=True)
        self._verify_thread.start()

    def _submit_verify_job(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], tid: int, p_entry: Dict[str, Any], now: float) -> None:
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        head_y2 = int(y1 + height * 0.55)
        pad_x = int(width * 0.15)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1)
        cx2 = min(frame_bgr.shape[1], x2 + pad_x)
        cy2 = max(head_y2, y1 + 1)
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return

        p_entry["last_verify_submit"] = now
        try:
            self.verify_job_queue.put_nowait((tid, crop.copy()))
            p_entry["pending"] = True
        except queue.Full:
            p_entry["pending"] = False

    def _drain_verify_results(self, now: float) -> None:
        while not self.verify_result_queue.empty():
            tid, res = self.verify_result_queue.get_nowait()

            if tid == -1 and res.kind == "RELOAD":
                for entry in self.person_tracker_db.values():
                    entry["votes"].clear()
                    entry["reauth_no_match_streak"] = 0
                    if entry["state"] != "OUTSIDE":
                        entry["state"] = "SCANNING"
                        entry["name"] = None
                        entry["roi_enter_time"] = now
                continue

            p_entry = self.person_tracker_db.get(tid)
            if p_entry is None:
                continue
            p_entry["pending"] = False

            if res.kind == "ERROR":
                self._verify_error_streak += 1
                if self._verify_error_streak >= 3:
                    self.status_queue.put(("● RECOGNITION ERROR", THEME["danger"]))
                continue

            self._verify_error_streak = 0
            if res.kind == "LOW_QUALITY":
                continue

            if p_entry["state"] == "AUTHORIZED":
                if res.kind == "NO_MATCH":
                    p_entry["reauth_no_match_streak"] += 1
                    if p_entry["reauth_no_match_streak"] >= 2:
                        p_entry["state"] = "SCANNING"
                        p_entry["name"] = None
                        p_entry["votes"].clear()
                        p_entry["reauth_no_match_streak"] = 0
                elif res.kind == "MATCH":
                    p_entry["reauth_no_match_streak"] = 0
                continue

            p_entry["votes"].append((res.kind, res.name))
            p_entry["last_dist"] = res.distance

            match_names = [v_name for v_kind, v_name in p_entry["votes"] if v_kind == "MATCH" and v_name]
            if match_names:
                top_name, top_count = Counter(match_names).most_common(1)[0]
                if top_count >= VOTES_REQUIRED:
                    was_intruder = p_entry["state"] in {"INTRUDER", "OBSCURED"}
                    p_entry["state"] = "AUTHORIZED"
                    p_entry["name"] = top_name
                    p_entry["reauth_no_match_streak"] = 0
                    if was_intruder:
                        self.log_queue.put(f"[{time.strftime('%H:%M:%S')}] ✓ INFO: Late biometric match -> AUTHORIZED: {top_name} | ID:{tid}")

    def _process_stream_loop(self) -> None:
        prev_time = time.time()

        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret or not self.is_running or frame is None:
                time.sleep(0.01)
                continue

            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = inst_fps if self.fps == 0.0 else (0.9 * self.fps + 0.1 * inst_fps)
            prev_time = current_time

            fh, fw = frame.shape[:2]
            self.frame_orig_w, self.frame_orig_h = fw, fh

            self._drain_verify_results(current_time)

            rx1, ry1, rx2, ry2 = self.roi_frame
            fx1 = max(0, min(rx1, fw - 1))
            fy1 = max(0, min(ry1, fh - 1))
            fx2 = max(fx1 + 10, min(rx2, fw))
            fy2 = max(fy1 + 10, min(ry2, fh))
            roi_polygon = np.array([[fx1, fy1], [fx2, fy1], [fx2, fy2], [fx1, fy2]], np.int32)

            active_in_roi = 0
            boxes_data: List[Tuple[int, int, int, int, int, Optional[np.ndarray], Optional[np.ndarray]]] = []
            if self.model is not None:
                try:
                    res = self.model.track(
                        frame,
                        persist=True,
                        classes=[0],
                        conf=YOLO_CONF,
                        tracker=TRACKER_CFG,
                        verbose=False,
                    )
                    if res and res[0].boxes is not None and len(res[0].boxes):
                        kps_obj = getattr(res[0], "keypoints", None)
                        kps_xy = kps_obj.xy.cpu().numpy() if (kps_obj is not None and kps_obj.xy is not None) else None
                        kps_cf = kps_obj.conf.cpu().numpy() if (kps_obj is not None and kps_obj.conf is not None) else None

                        for idx_b, b in enumerate(res[0].boxes):
                            bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
                            tid_tensor = b.id
                            if tid_tensor is None:
                                continue
                            tid = int(tid_tensor.item())
                            kp_i = kps_xy[idx_b] if kps_xy is not None and idx_b < len(kps_xy) else None
                            cf_i = kps_cf[idx_b] if kps_cf is not None and idx_b < len(kps_cf) else None
                            boxes_data.append((bx1, by1, bx2, by2, tid, kp_i, cf_i))
                except Exception as e:
                    print(f"[YOLO TRACK ERROR] {e}")

            pending_alerts: List[Tuple[str, int, str, Dict[str, Any]]] = []

            for (bx1, by1, bx2, by2, tid, kp_i, cf_i) in boxes_data:
                in_roi = is_in_roi((bx1, by1, bx2, by2), roi_polygon, mode=ROI_MODE)

                p_entry = self.person_tracker_db.get(tid)
                if p_entry is None:
                    p_entry = {
                        "state": "OUTSIDE",
                        "name": None,
                        "roi_enter_time": None,
                        "last_seen": current_time,
                        "last_verify_submit": 0.0,
                        "pending": False,
                        "votes": deque(maxlen=VOTE_WINDOW),
                        "reauth_no_match_streak": 0,
                        "last_dist": None,
                        "breach_logged": False,
                        "loiter_logged": False,
                        "fall_logged": False,
                        "run_logged": False,
                        "fall_since": None,
                        "run_since": None,
                        "trail": deque(),
                    }
                    self.person_tracker_db[tid] = p_entry

                p_entry["last_seen"] = current_time

                if not in_roi:
                    if p_entry["roi_enter_time"] is not None:
                        p_entry["roi_enter_time"] = None
                        p_entry["breach_logged"] = False
                        p_entry["loiter_logged"] = False
                        p_entry["fall_logged"] = False
                        p_entry["run_logged"] = False
                        p_entry["fall_since"] = None
                        p_entry["run_since"] = None
                        if p_entry["state"] != "AUTHORIZED":
                            p_entry["votes"].clear()
                            p_entry["state"] = "OUTSIDE"
                else:
                    active_in_roi += 1
                    if p_entry["roi_enter_time"] is None:
                        p_entry["roi_enter_time"] = current_time
                        if p_entry["state"] != "AUTHORIZED":
                            p_entry["state"] = "SCANNING"

                    if p_entry["state"] != "AUTHORIZED":
                        if (not p_entry["pending"]) and (current_time - p_entry["last_verify_submit"] >= REVERIFY_INTERVAL_SEC):
                            self._submit_verify_job(frame, (bx1, by1, bx2, by2), tid, p_entry, current_time)
                    else:
                        if (not p_entry["pending"]) and (current_time - p_entry["last_verify_submit"] >= REAUTH_INTERVAL_SEC):
                            self._submit_verify_job(frame, (bx1, by1, bx2, by2), tid, p_entry, current_time)

                    if p_entry["state"] == "SCANNING" and (current_time - p_entry["roi_enter_time"] >= GRACE_PERIOD_SEC):
                        votes = list(p_entry["votes"])
                        no_face_cnt = sum(1 for vk, _ in votes if vk == "NO_FACE")
                        p_entry["state"] = "OBSCURED" if (votes and no_face_cnt >= len(votes) / 2.0) else "INTRUDER"

                dwell_time = (current_time - p_entry["roi_enter_time"]) if (in_roi and p_entry["roi_enter_time"] is not None) else 0.0
                state = p_entry["state"]
                is_auth = state == "AUTHORIZED"

                # Fall detection
                fall_cand, _ = is_fall_candidate((bx1, by1, bx2, by2), fh, kp_i, cf_i)
                if in_roi and fall_cand:
                    if p_entry["fall_since"] is None:
                        p_entry["fall_since"] = current_time
                    is_fall = (current_time - p_entry["fall_since"]) >= FALL_SUSTAIN_SEC
                else:
                    p_entry["fall_since"] = None
                    is_fall = False

                # Running detection
                cx_pt, cy_pt = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
                trail = p_entry["trail"]
                trail.append((current_time, cx_pt, cy_pt))
                while trail and (current_time - trail[0][0] > 1.0):
                    trail.popleft()
                body_speed = compute_body_speed(trail, max(by2 - by1, 1))
                if ENABLE_RUNNING and in_roi and (body_speed > RUN_SPEED_BODYHEIGHTS_PER_S):
                    if p_entry["run_since"] is None:
                        p_entry["run_since"] = current_time
                    is_running_fast = (current_time - p_entry["run_since"]) >= RUN_SUSTAIN_SEC
                else:
                    p_entry["run_since"] = None
                    is_running_fast = False

                is_loitering = in_roi and (not is_auth) and (state in {"INTRUDER", "OBSCURED"}) and (dwell_time > LOITERING_THRESHOLD_SEC)

                # Visual HUD Color & text
                if not in_roi:
                    if is_auth:
                        box_color = (128, 222, 74)
                        status_text = f"AUTH: {p_entry['name']} (OUTSIDE)"
                    else:
                        box_color = (0, 200, 255)
                        status_text = "OUTSIDE ZONE"
                elif state == "SCANNING":
                    box_color = (255, 180, 0)
                    status_text = "SCANNING / VERIFYING..."
                elif is_auth:
                    if is_fall:
                        box_color = (0, 140, 255)
                        status_text = f"AUTH: {p_entry['name']} | ! FALL DETECTED"
                    elif is_running_fast:
                        box_color = (0, 180, 255)
                        status_text = f"AUTH: {p_entry['name']} | RUNNING"
                    else:
                        box_color = (128, 222, 74)
                        status_text = f"AUTHORIZED: {p_entry['name']}"
                else:
                    box_color = (68, 68, 239)
                    base_lbl = "FACE OBSCURED / INTRUDER" if state == "OBSCURED" else "UNAUTHORIZED INTRUDER"
                    parts = [f"{base_lbl} ({dwell_time:.1f}s)"]
                    if is_loitering:
                        parts.append("LOITERING")
                    if is_fall:
                        parts.append("! FALL DETECTED")
                    if is_running_fast:
                        parts.append("RUNNING")
                    status_text = " | ".join(parts)

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
                badge_text = f"ID:{tid} | {status_text}"
                (lw, lh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                bg_x1 = max(0, min(bx1, max(0, fw - lw - 8)))
                bg_x2 = min(fw, bg_x1 + lw + 8)
                if by1 - lh - 8 >= 0:
                    bg_y1, bg_y2 = by1 - lh - 8, by1
                    txt_y = by1 - 4
                else:
                    bg_y1, bg_y2 = by1, min(fh, by1 + lh + 8)
                    txt_y = bg_y2 - 4

                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), box_color, -1)
                cv2.putText(frame, badge_text, (bg_x1 + 4, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 23, 42), 1, cv2.LINE_AA)

                extra_meta = {
                    "name": p_entry["name"] or state,
                    "state": state,
                    "distance": p_entry["last_dist"],
                    "dwell_sec": dwell_time,
                }
                if in_roi and state in {"INTRUDER", "OBSCURED"}:
                    if not p_entry["breach_logged"]:
                        p_entry["breach_logged"] = True
                        self.total_breaches += 1
                        ev = "FACE_OBSCURED" if state == "OBSCURED" else "INTRUSION"
                        reason = "Face covered / obscured" if state == "OBSCURED" else "Unauthorized entry"
                        pending_alerts.append((ev, tid, f"🚨 INTRUSION: {reason} | ID:{tid}", extra_meta))

                    if is_loitering and not p_entry["loiter_logged"]:
                        p_entry["loiter_logged"] = True
                        pending_alerts.append(("LOITERING", tid, f"⚠️ LOITERING: Intruder stayed ({dwell_time:.1f}s) | ID:{tid}", extra_meta))

                if in_roi and is_fall and not p_entry["fall_logged"]:
                    p_entry["fall_logged"] = True
                    msg = (
                        f"🚑 SAFETY ALERT: Fall detected | ID:{tid} (Staff: {p_entry['name']})"
                        if is_auth
                        else f"🚨 ALERT: Intruder fall detected | ID:{tid}"
                    )
                    pending_alerts.append(("FALL", tid, msg, extra_meta))

                if in_roi and is_running_fast and not p_entry["run_logged"]:
                    p_entry["run_logged"] = True
                    who = p_entry["name"] if is_auth else state
                    pending_alerts.append(("RUNNING", tid, f"⚡ RUNNING: Rapid movement ({body_speed:.1f} bh/s) | ID:{tid} ({who})", extra_meta))

            # Overcrowding & Restricted hours
            if ENABLE_OVERCROWDING and active_in_roi > MAX_PEOPLE_IN_ROI:
                if self._overcrowd_since is None:
                    self._overcrowd_since = current_time
                elif (current_time - self._overcrowd_since >= 2.0) and (current_time - self._last_overcrowd_alert >= ALERT_COOLDOWN_SEC):
                    self._last_overcrowd_alert = current_time
                    pending_alerts.append(("OVERCROWDING", 0, f"⚠️ OVERCROWDING: {active_in_roi} people in zone (max {MAX_PEOPLE_IN_ROI})", {"state": "CROWD"}))
            else:
                self._overcrowd_since = None

            if ENABLE_RESTRICTED_HOURS and active_in_roi > 0 and is_within_restricted_hours(RESTRICTED_HOURS):
                if current_time - self._last_after_hours_alert >= ALERT_COOLDOWN_SEC:
                    self._last_after_hours_alert = current_time
                    pending_alerts.append(("SYSTEM", 0, f"⚠️ AFTER_HOURS_PRESENCE: {active_in_roi} target(s) in restricted hours", {"state": "AFTER_HOURS"}))

            # Dispatch alerts with clean annotated snapshot
            if pending_alerts:
                snap = frame.copy()
                cv2.polylines(snap, [roi_polygon], isClosed=True, color=(245, 158, 11), thickness=2)
                cv2.putText(snap, "ACTIVE TRIPWIRE ZONE", (fx1 + 8, max(20, fy1 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 158, 11), 2, cv2.LINE_AA)
                for ev_type, tid_a, msg_a, extra_a in pending_alerts:
                    self.alert_manager.emit(ev_type, tid_a, msg_a, snapshot_bgr=snap, extra=extra_a)

            # Purge stale tracks
            stale_ids = [t_id for t_id, entry in self.person_tracker_db.items() if (current_time - entry["last_seen"]) > TRACK_STALE_SEC]
            for s_id in stale_ids:
                del self.person_tracker_db[s_id]

            # Render to canvas queue
            cw, ch = self.canvas_w, self.canvas_h
            scale, off_x, off_y = self._get_canvas_to_frame_scale()
            disp_w = max(int(fw * scale), 1)
            disp_h = max(int(fh * scale), 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

            canvas_img = np.zeros((ch, cw, 3), dtype=np.uint8)
            canvas_img[off_y : off_y + disp_h, off_x : off_x + disp_w] = rgb_resized

            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait((canvas_img, self.fps, active_in_roi, self.total_breaches))
            except Exception:
                pass

    def _add_log_entry(self, entry: str) -> None:
        self.log_listbox.insert(0, entry)
        if self.log_listbox.size() > 100:
            self.log_listbox.delete(100, tk.END)

    def close(self) -> None:
        self.stop_surveillance()
        try:
            self.verify_job_queue.put_nowait(None)
        except queue.Full:
            pass
        self.alert_manager.close()
        self.root.destroy()


# ===========================================================================
# 8. SYSTEM ENTRY POINT
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="AI Surveillance & Security Threat Monitor")
    parser.add_argument("--enroll", action="store_true", help="Launch directly into Face Enrollment Studio")
    args = parser.parse_args()

    if args.enroll:
        app = FaceEnrollmentDialog(master=None)
        app.window.mainloop()
    else:
        root = tk.Tk()
        app = SurveillanceGUIApp(root)
        root.protocol("WM_DELETE_WINDOW", app.close)
        root.mainloop()


if __name__ == "__main__":
    main()
