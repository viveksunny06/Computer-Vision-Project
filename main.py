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
# 2. DESIGN SYSTEM
#    One neutral palette, one accent, semantic colours reserved for state.
#    Type scale (px-ish pt): 9 caption | 10 body | 11 emphasis | 12 title | 20 numerals
#    Spacing scale: 4 / 8 / 12 / 16 / 20.  Card padding 16, gap between cards 12.
# ===========================================================================
THEME = {
    # surfaces
    "bg_main": "#0d1117",        # window
    "bg_surface": "#161b22",     # cards / panels
    "bg_raised": "#21262d",      # buttons, pills
    "bg_card": "#21262d",        # (legacy key) same as bg_raised
    "bg_input": "#0d1117",       # inset fields, log
    "bg_video": "#05080c",
    "selection": "#1c2d44",
    # lines
    "border": "#30363d",
    "border_light": "#484f58",
    # text
    "text_primary": "#e6edf3",
    "text_secondary": "#9da7b3",
    "text_muted": "#7d8590",
    # accent (primary action only)
    "primary": "#2563eb",
    "primary_hover": "#1d4ed8",
    "info": "#58a6ff",           # focus ring, selection accents
    "zone": "#38bdf8",           # restricted-zone overlay
    # state (never decorative)
    "success": "#3fb950",
    "success_hover": "#2ea043",
    "warning": "#d29922",
    "danger": "#f85149",         # text / dot on dark
    "danger_hover": "#da3633",
    "danger_solid": "#c62d2d",   # filled button with white text
    "danger_solid_hover": "#a82424",
    "neutral_btn": "#21262d",
    "neutral_hover": "#2d333b",
}

_FONTS: Dict[str, str] = {"ui": "Segoe UI", "semi": "Segoe UI Semibold", "mono": "Consolas"}


def init_ui_fonts() -> None:
    """Pick the best installed UI/mono families once a Tk root exists."""
    import tkinter.font as tkfont

    families = set(tkfont.families())
    for cand in ("Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"):
        if cand in families:
            _FONTS["ui"] = cand
            break
    _FONTS["semi"] = "Segoe UI Semibold" if "Segoe UI Semibold" in families else _FONTS["ui"]
    for cand in ("Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Courier New"):
        if cand in families:
            _FONTS["mono"] = cand
            break


def F(size: int = 10, weight: str = "normal") -> Tuple:
    """UI font tuple. weight='bold' maps to Segoe UI Semibold when available."""
    if weight == "bold":
        if _FONTS["semi"] != _FONTS["ui"]:
            return (_FONTS["semi"], size)
        return (_FONTS["ui"], size, "bold")
    return (_FONTS["ui"], size)


def FM(size: int = 9) -> Tuple:
    return (_FONTS["mono"], size)


def apply_ttk_styles() -> None:
    T = THEME
    style = ttk.Style()
    style.theme_use("clam")

    # Arrow-less slim scrollbar
    style.layout(
        "Slim.Vertical.TScrollbar",
        [("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})],
    )
    style.configure(
        "Slim.Vertical.TScrollbar",
        background=T["border_light"], troughcolor=T["bg_input"], bordercolor=T["bg_input"],
        lightcolor=T["border_light"], darkcolor=T["border_light"], width=8, borderwidth=0, relief="flat",
    )
    style.map("Slim.Vertical.TScrollbar", background=[("active", T["text_muted"])])

    # Borderless list
    style.layout("People.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
    style.configure(
        "People.Treeview",
        background=T["bg_input"], fieldbackground=T["bg_input"], foreground=T["text_primary"],
        borderwidth=0, relief="flat", rowheight=34, font=F(10),
    )
    style.map(
        "People.Treeview",
        background=[("selected", T["selection"])],
        foreground=[("selected", T["text_primary"])],
    )


def draw_person_badge(frame: np.ndarray, bbox: Tuple[int, int, int, int], tid: int, text: str, color: Tuple[int, int, int]) -> None:
    """Draw a bounding box and a readable label chip on a BGR frame (ASCII text only)."""
    bx1, by1, bx2, by2 = bbox
    fh, fw = frame.shape[:2]
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)

    label = f"#{tid}  {text}"
    scale, thick, pad = 0.5, 1, 6
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    chip_w, chip_h = lw + 2 * pad, lh + 2 * pad
    x1 = int(max(0, min(bx1, fw - chip_w)))
    y1 = by1 - chip_h if (by1 - chip_h) >= 0 else by1
    y1 = int(max(0, min(y1, fh - chip_h)))
    b, g, r = color
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    fg = (255, 255, 255) if luminance < 125 else (22, 27, 34)
    cv2.rectangle(frame, (x1, y1), (x1 + chip_w, y1 + chip_h), color, -1)
    cv2.putText(frame, label, (x1 + pad, y1 + pad + lh), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
class Button(tk.Label):
    """Flat button with variants: primary | secondary | ghost | danger | danger_text.

    Supports .config(text=..., state=..., variant=...). Keyboard accessible (Tab, Enter, Space).
    """

    def __init__(self, master, text: str, command: Optional[Callable[[], None]] = None, variant: str = "secondary", width: Optional[int] = None, **kw):
        self._variant = variant
        self._command = command
        self._disabled = False
        self._hover = False
        self._pressed = False
        self._parent_bg = master.cget("bg")
        opts: Dict[str, Any] = dict(
            text=text, font=F(10, "bold"), padx=14, pady=6, bd=0, relief="flat",
            cursor="hand2", takefocus=1, highlightthickness=2, anchor="center",
        )
        if width:
            opts["width"] = width
        opts.update(kw)
        super().__init__(master, **opts)
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Return>", lambda e: self.invoke())
        self.bind("<space>", lambda e: self.invoke())
        self.bind("<FocusIn>", lambda e: super(Button, self).configure(highlightbackground=THEME["info"]))
        self.bind("<FocusOut>", lambda e: self._paint())
        self._paint()

    def _palette(self) -> Tuple[str, str]:
        T = THEME
        table = {
            "primary": (T["primary"], T["primary_hover"], "#ffffff"),
            "secondary": (T["bg_raised"], T["neutral_hover"], T["text_primary"]),
            "ghost": (self._parent_bg, T["bg_raised"], T["text_secondary"]),
            "danger": (T["danger_solid"], T["danger_solid_hover"], "#ffffff"),
            "danger_text": (T["bg_raised"], T["neutral_hover"], T["danger"]),
        }
        bg, hover_bg, fg = table[self._variant]
        if self._disabled:
            return (self._parent_bg if self._variant == "ghost" else T["bg_raised"]), T["text_muted"]
        if self._pressed or self._hover:
            bg = hover_bg
            if self._variant == "ghost":
                fg = T["text_primary"]
        return bg, fg

    def _paint(self) -> None:
        bg, fg = self._palette()
        tk.Label.configure(self, bg=bg, fg=fg, highlightbackground=bg, cursor="arrow" if self._disabled else "hand2")

    def _set_hover(self, value: bool) -> None:
        self._hover = value
        self._paint()

    def _on_press(self, _e) -> None:
        self._pressed = True
        self._paint()

    def _on_release(self, e) -> None:
        was = self._pressed
        self._pressed = False
        self._paint()
        if was and 0 <= e.x < self.winfo_width() and 0 <= e.y < self.winfo_height():
            self.invoke()

    def invoke(self) -> None:
        if self._command and not self._disabled:
            self._command()

    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        variant = kw.pop("variant", None)
        state = kw.pop("state", None)
        for legacy in ("bg", "background", "fg", "foreground", "activebackground", "activeforeground"):
            kw.pop(legacy, None)
        if variant:
            self._variant = variant
        if state is not None:
            self._disabled = state == tk.DISABLED
        if kw:
            tk.Label.configure(self, **kw)
        self._paint()

    config = configure


class StatusPill(tk.Frame):
    """Single source of truth for system state: coloured dot + sentence-case text.

    .config(text="● SYSTEM READY (2 People)", fg=color) is accepted and normalised.
    """

    _REPLACE = (("Yolo load error", "Model load error"), (" db", " database"), ("...", "…"))

    def __init__(self, master, text: str = "", fg: Optional[str] = None):
        T = THEME
        super().__init__(master, bg=T["bg_raised"], highlightthickness=1, highlightbackground=T["border"])
        self._dot = tk.Canvas(self, width=8, height=8, bg=T["bg_raised"], highlightthickness=0)
        self._dot.pack(side=tk.LEFT, padx=(10, 7), pady=7)
        self._dot_id = self._dot.create_oval(0, 0, 8, 8, fill=fg or T["text_muted"], outline="")
        self._lbl = tk.Label(self, font=F(9, "bold"), fg=T["text_primary"], bg=T["bg_raised"])
        self._lbl.pack(side=tk.LEFT, padx=(0, 12))
        self.configure(text=text, fg=fg or T["text_muted"])

    @classmethod
    def _norm(cls, text: str) -> str:
        s = text.replace("●", "").strip().capitalize()
        for old, new in cls._REPLACE:
            s = s.replace(old, new)
        return s

    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        text = kw.pop("text", None)
        fg = kw.pop("fg", kw.pop("foreground", None))
        if text is not None:
            self._lbl.configure(text=self._norm(text))
        if fg is not None:
            self._dot.itemconfigure(self._dot_id, fill=fg)
        if kw:
            tk.Frame.configure(self, **kw)

    config = configure


class ProgressBar(tk.Canvas):
    def __init__(self, master, height: int = 4):
        super().__init__(master, height=height, bg=THEME["bg_surface"], highlightthickness=0, bd=0)
        self._value, self._max = 0, 1
        self.bind("<Configure>", lambda e: self._redraw())

    def set(self, value: int, maximum: int) -> None:
        self._value, self._max = max(0, value), max(1, maximum)
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        self.create_rectangle(0, 0, w, h, fill=THEME["bg_raised"], outline="")
        frac = min(1.0, self._value / self._max)
        color = THEME["success"] if self._value >= self._max else THEME["primary"]
        if frac > 0:
            self.create_rectangle(0, 0, int(w * frac), h, fill=color, outline="")


class SegmentBar(tk.Canvas):
    """Row of equal segments; segments before `current` are done, `current` is active."""

    def __init__(self, master, count: int, height: int = 4):
        super().__init__(master, height=height, bg=THEME["bg_surface"], highlightthickness=0, bd=0)
        self._count, self._current = count, 0
        self.bind("<Configure>", lambda e: self._redraw())

    def set(self, current: int) -> None:
        self._current = current
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        w, h, gap = self.winfo_width(), self.winfo_height(), 4
        seg = (w - gap * (self._count - 1)) / max(1, self._count)
        for i in range(self._count):
            x1 = i * (seg + gap)
            color = THEME["border_light"]
            if i < self._current:
                color = THEME["primary"]
            elif i == self._current:
                color = THEME["info"]
            self.create_rectangle(x1, 0, x1 + seg, h, fill=color, outline="")


class Notice(tk.Frame):
    """Reserved-height feedback line with a coloured left bar (no emoji)."""

    def __init__(self, master, height: int = 44):
        super().__init__(master, bg=THEME["bg_surface"], height=height)
        self.pack_propagate(False)
        self._bar = tk.Frame(self, bg=THEME["bg_surface"], width=3)
        self._bar.pack(side=tk.LEFT, fill=tk.Y)
        self._lbl = tk.Label(self, text="", font=F(9), fg=THEME["text_secondary"], bg=THEME["bg_surface"], anchor="nw", justify=tk.LEFT)
        self._lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self.bind("<Configure>", lambda e: self._lbl.configure(wraplength=max(100, e.width - 24)))

    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        text = kw.pop("text", None)
        fg = kw.pop("fg", kw.pop("foreground", None))
        if text is not None:
            clean = re.sub(r"^[\s\u2190-\u2BFF\U0001F000-\U0001FFFF\uFE0F]+", "", text)
            self._lbl.configure(text=clean)
            if not clean:
                self._bar.configure(bg=THEME["bg_surface"])
        if fg is not None and self._lbl.cget("text"):
            self._bar.configure(bg=THEME["border_light"] if fg == THEME["text_secondary"] else fg)
            self._lbl.configure(fg=THEME["text_primary"])
        if kw:
            tk.Frame.configure(self, **kw)

    config = configure


_LOG_EVENTS: Dict[str, Tuple[str, str]] = {
    "INTRUSION": ("Intrusion", "danger"),
    "ALERT": ("Alert", "danger"),
    "SAFETY ALERT": ("Fall", "danger"),
    "LOITERING": ("Loitering", "warning"),
    "RUNNING": ("Running", "warning"),
    "OVERCROWDING": ("Crowding", "warning"),
    "AFTER_HOURS_PRESENCE": ("After hours", "warning"),
    "TEST ALERT": ("Test", "info"),
    "INFO": ("Info", "muted"),
    "SYSTEM": ("System", "muted"),
}
_LOG_RE = re.compile(r"^\[(?P<t>[^\]]+)\]\s*(?P<body>.*)$")


def parse_log_entry(entry: str) -> Tuple[str, str, str, str]:
    """'[12:00:01] 🚨 INTRUSION: Unauthorized entry | ID:3' -> (time, label, severity, detail)."""
    m = _LOG_RE.match(entry.strip())
    ts, body = (m.group("t"), m.group("body")) if m else (time.strftime("%H:%M:%S"), entry)
    body = re.sub(r"^[^\w]+", "", body)
    head, sep, rest = body.partition(":")
    key = head.strip().upper()
    if sep and key in _LOG_EVENTS:
        label, sev = _LOG_EVENTS[key]
        detail = rest.strip()
    elif sep and len(head) <= 24:
        label, sev, detail = head.strip().capitalize(), "muted", rest.strip()
    else:
        label, sev, detail = "System", "muted", body.strip()
    detail = detail.replace("->", "→").replace(" | ", "  ·  ")
    detail = re.sub(r"\bID:(\d+)", r"ID \1", detail)
    return ts, label, sev, detail


class LogView(tk.Frame):
    """Read-only incident feed, newest first: coloured event + right-aligned time, detail below."""

    MAX_ENTRIES = 150

    def __init__(self, master):
        T = THEME
        super().__init__(master, bg=T["bg_input"], highlightthickness=1, highlightbackground=T["border"])
        self.text = tk.Text(
            self, bg=T["bg_input"], fg=T["text_primary"], relief="flat", bd=0, highlightthickness=0,
            wrap="word", cursor="arrow", padx=12, pady=4, state="disabled", insertwidth=0, takefocus=0,
            selectbackground=T["selection"], inactiveselectbackground=T["selection"], font=F(10),
        )
        self._scroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview, style="Slim.Vertical.TScrollbar")
        self.text.configure(yscrollcommand=self._on_scroll)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.text.tag_configure("head", spacing1=12)
        self.text.tag_configure("time", font=FM(9), foreground=T["text_muted"])
        self.text.tag_configure("body", font=F(10), foreground=T["text_primary"], spacing3=2)
        for sev, color in (("danger", T["danger"]), ("warning", T["warning"]), ("info", T["info"]), ("muted", T["text_secondary"])):
            self.text.tag_configure(f"sev_{sev}", font=F(10, "bold"), foreground=color)

        self._empty = tk.Label(self, text="No incidents recorded", font=F(10), fg=T["text_muted"], bg=T["bg_input"])
        self._empty.place(relx=0.5, rely=0.5, anchor="center")
        self._count = 0
        self.text.bind("<Configure>", self._on_resize)

    def _on_scroll(self, first, last) -> None:
        self._scroll.set(first, last)
        if float(first) <= 0.0 and float(last) >= 1.0:
            self._scroll.pack_forget()
        else:
            self._scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_resize(self, event) -> None:
        self.text.configure(tabs=(max(event.width - 26, 120), "right"))

    def add(self, entry: str) -> None:
        ts, label, sev, detail = parse_log_entry(entry)
        self._empty.place_forget()
        self.text.configure(state="normal")
        self.text.insert(
            "1.0",
            label, ("head", f"sev_{sev}"),
            f"\t{ts}\n", ("head", "time"),
            f"{detail}\n", ("body",),
        )
        self._count += 1
        if self._count > self.MAX_ENTRIES:
            self.text.delete(f"{self.MAX_ENTRIES * 2 + 1}.0", "end")
            self._count = self.MAX_ENTRIES
        self.text.configure(state="disabled")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self._count = 0
        self._empty.place(relx=0.5, rely=0.5, anchor="center")


def make_card(master, padx: int = 16, pady: int = 14) -> tk.Frame:
    return tk.Frame(master, bg=THEME["bg_surface"], highlightthickness=1, highlightbackground=THEME["border"], highlightcolor=THEME["border"], padx=padx, pady=pady)


def make_input(master) -> Tuple[tk.Frame, tk.Entry]:
    """Single-line input with 1px border that turns accent-coloured on focus."""
    T = THEME
    wrap = tk.Frame(master, bg=T["border"])
    entry = tk.Entry(
        wrap, font=F(11), bg=T["bg_input"], fg=T["text_primary"], insertbackground=T["text_primary"],
        relief="flat", bd=0, highlightthickness=0, selectbackground=T["primary"], selectforeground="#ffffff",
    )
    entry.pack(fill=tk.X, padx=1, pady=1, ipady=7, ipadx=8)
    entry.bind("<FocusIn>", lambda e: wrap.configure(bg=T["info"]))
    entry.bind("<FocusOut>", lambda e: wrap.configure(bg=T["border"]))
    return wrap, entry


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

        self.window.title("Enroll Personnel")
        self.window.geometry("1120x740")
        self.window.minsize(980, 660)
        self._has_frame = False
        self._placeholder = ("Starting camera...", "")
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
        init_ui_fonts()
        apply_ttk_styles()

    def _build_ui(self) -> None:
        T = THEME

        # ---- Title bar -----------------------------------------------------
        header = tk.Frame(self.window, bg=T["bg_surface"], height=52)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Frame(self.window, bg=T["border"], height=1).pack(fill=tk.X)

        tk.Label(header, text="Enroll Personnel", font=F(12, "bold"), fg=T["text_primary"], bg=T["bg_surface"]).pack(side=tk.LEFT, padx=(20, 0))
        self.lbl_cam_indicator = StatusPill(header, text="Connecting...", fg=T["text_muted"])
        self.lbl_cam_indicator.pack(side=tk.RIGHT, padx=(0, 20))

        content = tk.Frame(self.window, bg=T["bg_main"], padx=16, pady=16)
        content.pack(fill=tk.BOTH, expand=True)

        # ---- Right column (packed first so it keeps its width) -------------
        side = tk.Frame(content, bg=T["bg_main"], width=340)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        # ---- Left: preview + pose guide ------------------------------------
        left = make_card(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        cam_hdr = tk.Frame(left, bg=T["bg_surface"])
        cam_hdr.pack(fill=tk.X, pady=(0, 10))
        tk.Label(cam_hdr, text="Camera preview", font=F(10, "bold"), fg=T["text_primary"], bg=T["bg_surface"]).pack(side=tk.LEFT)
        tk.Label(cam_hdr, text=f"Camera {CAMERA_INDEX}", font=F(9), fg=T["text_muted"], bg=T["bg_surface"]).pack(side=tk.RIGHT)

        self.video_container = tk.Frame(left, bg=T["bg_video"], highlightthickness=1, highlightbackground=T["border"])
        self.video_container.pack(fill=tk.BOTH, expand=True)
        self.video_canvas = tk.Canvas(self.video_container, bg=T["bg_video"], highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Configure>", lambda e: self._draw_placeholder())

        guide = tk.Frame(left, bg=T["bg_surface"])
        guide.pack(fill=tk.X, pady=(14, 0))
        guide_top = tk.Frame(guide, bg=T["bg_surface"])
        guide_top.pack(fill=tk.X)
        tk.Label(guide_top, text="Next pose", font=F(9), fg=T["text_secondary"], bg=T["bg_surface"]).pack(side=tk.LEFT)
        self.lbl_pose_step = tk.Label(guide_top, text="", font=F(9), fg=T["text_secondary"], bg=T["bg_surface"])
        self.lbl_pose_step.pack(side=tk.RIGHT)
        self.lbl_pose_prompt = tk.Label(guide, text="", font=F(11, "bold"), fg=T["text_primary"], bg=T["bg_surface"], anchor="w")
        self.lbl_pose_prompt.pack(fill=tk.X, pady=(2, 8))
        self.pose_bar = SegmentBar(guide, len(POSE_PROMPTS))
        self.pose_bar.pack(fill=tk.X)
        self._set_pose(0)

        # ---- Right: person card --------------------------------------------
        person = make_card(side)
        person.pack(fill=tk.X, pady=(0, 12))
        tk.Label(person, text="Person", font=F(10, "bold"), fg=T["text_primary"], bg=T["bg_surface"]).pack(anchor="w")
        tk.Label(person, text="Name", font=F(9), fg=T["text_secondary"], bg=T["bg_surface"]).pack(anchor="w", pady=(10, 4))

        wrap, self.name_entry = make_input(person)
        wrap.pack(fill=tk.X)
        self.name_entry.focus()
        self.name_entry.bind("<KeyRelease>", self._on_name_changed)

        self.lbl_person_stats = tk.Label(person, text="", font=F(9), fg=T["text_secondary"], bg=T["bg_surface"], anchor="nw", justify=tk.LEFT, wraplength=290, height=2)
        self.lbl_person_stats.pack(fill=tk.X, pady=(8, 6))
        self.progress = ProgressBar(person)
        self.progress.pack(fill=tk.X, pady=(0, 14))

        self.btn_capture = Button(person, "Capture photo", variant="primary", command=self.capture_single_photo)
        self.btn_capture.pack(fill=tk.X)
        self.lbl_capture_feedback = Notice(person)
        self.lbl_capture_feedback.pack(fill=tk.X, pady=(12, 0))

        # ---- Right: enrolled list ------------------------------------------
        people = make_card(side)
        people.pack(fill=tk.BOTH, expand=True)

        list_hdr = tk.Frame(people, bg=T["bg_surface"])
        list_hdr.pack(fill=tk.X, pady=(0, 10))
        tk.Label(list_hdr, text="Enrolled", font=F(10, "bold"), fg=T["text_primary"], bg=T["bg_surface"]).pack(side=tk.LEFT)
        self.lbl_total_enrolled = tk.Label(list_hdr, text="0 enrolled", font=F(9), fg=T["text_secondary"], bg=T["bg_surface"])
        self.lbl_total_enrolled.pack(side=tk.RIGHT)

        Button(people, "Delete selected", variant="danger_text", command=self.delete_selected_person).pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))

        list_box = tk.Frame(people, bg=T["bg_input"], highlightthickness=1, highlightbackground=T["border"])
        list_box.pack(fill=tk.BOTH, expand=True)
        self.personnel_tree = ttk.Treeview(list_box, style="People.Treeview", columns=("count",), show="tree", selectmode="browse")
        self.personnel_tree.column("#0", stretch=True, anchor="w", minwidth=120)
        self.personnel_tree.column("count", width=84, anchor="e", stretch=False)
        self.personnel_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.personnel_tree.bind("<<TreeviewSelect>>", self._on_list_select)
        self._lbl_list_empty = tk.Label(list_box, text="No one enrolled yet", font=F(10), fg=T["text_muted"], bg=T["bg_input"])

        self._on_name_changed()

    # -- helpers used by the UI --------------------------------------------
    def _set_pose(self, idx: int) -> None:
        self.lbl_pose_prompt.config(text=POSE_PROMPTS[idx])
        self.lbl_pose_step.config(text=f"{idx + 1} of {len(POSE_PROMPTS)}")
        self.pose_bar.set(idx)

    def _draw_placeholder(self) -> None:
        if getattr(self, "_has_frame", False):
            return
        T = THEME
        c = self.video_canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 200), max(c.winfo_height(), 120)
        title, sub = getattr(self, "_placeholder", ("Starting camera...", ""))
        c.create_text(w // 2, h // 2 - (10 if sub else 0), text=title, fill=T["text_secondary"], font=F(11, "bold"))
        if sub:
            c.create_text(w // 2, h // 2 + 14, text=sub, fill=T["text_muted"], font=F(9))


    def _poll_queues(self) -> None:
        try:
            while not self.status_queue.empty():
                text, color = self.status_queue.get_nowait()
                self.lbl_cam_indicator.config(text=text, fg=color)
                if "OFFLINE" in text.upper():
                    self._placeholder = ("Camera unavailable", "Close any app that is using the camera, then reopen this window.")
                    self._draw_placeholder()
        except Exception:
            pass

        try:
            while not self.capture_result_queue.empty():
                ok, msg, saved_name = self.capture_result_queue.get_nowait()
                self.btn_capture.config(state=tk.NORMAL, text="Capture photo")
                if ok:
                    self.flash_alpha = 0.70
                    self.pose_idx = (self.pose_idx + 1) % len(POSE_PROMPTS)
                    self._set_pose(self.pose_idx)
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
                canvas_img = np.full((ch, cw, 3), (12, 8, 5), dtype=np.uint8)
                canvas_img[oy : oy + dh, ox : ox + dw] = resized

                pil_img = Image.fromarray(canvas_img)
                self.current_tk_image = ImageTk.PhotoImage(image=pil_img, master=self.video_canvas)
                self.video_canvas.delete("all")
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_tk_image)
                self._has_frame = True
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
        rec = ENROLL_RECOMMENDED_COUNT
        raw = self.name_entry.get().strip()
        if not raw:
            self.lbl_person_stats.config(text="Enter a name to begin.", fg=THEME["text_secondary"])
            self.progress.set(0, rec)
            return

        safe = self._sanitize_name(raw)
        if not self._is_valid_name(safe):
            self.lbl_person_stats.config(text="Invalid name (empty or reserved).", fg=THEME["danger"])
            self.progress.set(0, rec)
            return

        resolved = self._resolve_person_dir_name(safe)
        count = len(self._get_existing_photos(resolved))
        if count > 25:
            hint = f"{resolved}: {count} photos. 10-15 varied photos are enough."
        elif count > 0:
            hint = f"{resolved}: {count} of {rec} recommended photos."
        else:
            hint = f"New person. 0 of {rec} recommended photos."
        self.lbl_person_stats.config(text=hint, fg=THEME["text_secondary"])
        self.progress.set(count, rec)

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
        tree = self.personnel_tree
        tree.delete(*tree.get_children())
        total = 0

        def _label(n: int) -> str:
            return f"{n} photo" if n == 1 else f"{n} photos"

        if KNOWN_FACES_DIR.exists():
            for entry in sorted(KNOWN_FACES_DIR.iterdir(), key=lambda p: p.name.lower()):
                if entry.is_dir():
                    n = len([f for f in entry.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
                    iid = entry.name
                elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                    n, iid = 1, entry.stem
                else:
                    continue
                if not tree.exists(iid):
                    tree.insert("", tk.END, iid=iid, text="  " + iid, values=(_label(n),))
                    total += 1

        self.lbl_total_enrolled.config(text=f"{total} enrolled")
        if total == 0:
            self._lbl_list_empty.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self._lbl_list_empty.place_forget()

    def _on_list_select(self, event=None) -> None:
        sel = self.personnel_tree.selection()
        if not sel:
            return
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, sel[0])
        self._on_name_changed()

    def delete_selected_person(self) -> None:
        sel = self.personnel_tree.selection()
        if not sel:
            messagebox.showinfo("Select Person", "Please select a person from the list to remove.")
            return

        person_name = sel[0]
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
                text=f"Deleted '{person_name}'. Face database will refresh.",
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
        self.root.title("Surveillance Monitor")
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
        init_ui_fonts()
        apply_ttk_styles()

    def _make_metric(self, parent: tk.Frame, column: int, caption: str) -> tk.Label:
        T = THEME
        cell = tk.Frame(parent, bg=T["bg_surface"])
        cell.grid(row=0, column=column, sticky="nsew")
        if column > 0:
            tk.Frame(cell, bg=T["border"], width=1).pack(side=tk.LEFT, fill=tk.Y)
        inner = tk.Frame(cell, bg=T["bg_surface"])
        inner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(16 if column > 0 else 0, 0))
        value = tk.Label(inner, text="0", font=F(20, "bold"), fg=T["text_primary"], bg=T["bg_surface"], anchor="w")
        value.pack(anchor="w")
        tk.Label(inner, text=caption, font=F(9), fg=T["text_secondary"], bg=T["bg_surface"], anchor="w").pack(anchor="w")
        return value

    def _build_ui(self) -> None:
        T = THEME

        # ---- Title bar -----------------------------------------------------
        header = tk.Frame(self.root, bg=T["bg_surface"], height=52)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Frame(self.root, bg=T["border"], height=1).pack(fill=tk.X)

        tk.Label(header, text="Surveillance Monitor", font=F(12, "bold"), fg=T["text_primary"], bg=T["bg_surface"]).pack(side=tk.LEFT, padx=(20, 0))
        self.lbl_header_status = StatusPill(header, text="Loading models...", fg=T["text_muted"])
        self.lbl_header_status.pack(side=tk.RIGHT, padx=(0, 20))

        content = tk.Frame(self.root, bg=T["bg_main"], padx=16, pady=16)
        content.pack(fill=tk.BOTH, expand=True)

        # ---- Right column (packed first so it keeps its width) -------------
        side = tk.Frame(content, bg=T["bg_main"], width=360)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        # ---- Left: live view ------------------------------------------------
        left = make_card(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        cam_hdr = tk.Frame(left, bg=T["bg_surface"])
        cam_hdr.pack(fill=tk.X, pady=(0, 10))
        tk.Label(cam_hdr, text="Live view", font=F(10, "bold"), fg=T["text_primary"], bg=T["bg_surface"]).pack(side=tk.LEFT)
        tk.Label(cam_hdr, text=f"Camera {CAMERA_INDEX}", font=F(9), fg=T["text_muted"], bg=T["bg_surface"]).pack(side=tk.RIGHT)

        self.video_container = tk.Frame(left, bg=T["bg_video"], highlightthickness=1, highlightbackground=T["border"])
        self.video_container.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.video_container, bg=T["bg_video"], highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        toolbar = tk.Frame(left, bg=T["bg_surface"])
        toolbar.pack(fill=tk.X, pady=(12, 0))
        self.btn_toggle = Button(toolbar, "Start surveillance", variant="primary", width=18, command=self.toggle_surveillance)
        self.btn_toggle.pack(side=tk.LEFT)
        Button(toolbar, "Reset zone", variant="secondary", command=self.reset_roi).pack(side=tk.LEFT, padx=(8, 0))
        self.lbl_roi_coords = tk.Label(toolbar, text="", font=FM(9), fg=T["text_secondary"], bg=T["bg_surface"])
        self.lbl_roi_coords.pack(side=tk.RIGHT)
        self._set_zone_label(self.roi_frame)

        # ---- Right: metrics --------------------------------------------------
        metrics = make_card(side, pady=12)
        metrics.pack(fill=tk.X, pady=(0, 12))
        metrics.columnconfigure((0, 1, 2), weight=1, uniform="metric")
        self.lbl_fps_val = self._make_metric(metrics, 0, "FPS")
        self.lbl_targets_val = self._make_metric(metrics, 1, "In zone")
        self.lbl_breach_val = self._make_metric(metrics, 2, "Breaches")
        self.lbl_fps_val.config(text="0.0")

        # ---- Right: personnel ---------------------------------------------------
        people = make_card(side)
        people.pack(fill=tk.X, pady=(0, 12))
        p_hdr = tk.Frame(people, bg=T["bg_surface"])
        p_hdr.pack(fill=tk.X, pady=(0, 10))
        tk.Label(p_hdr, text="Personnel", font=F(10, "bold"), fg=T["text_primary"], bg=T["bg_surface"]).pack(side=tk.LEFT)
        self.lbl_enrolled = tk.Label(p_hdr, text="", font=F(9), fg=T["text_secondary"], bg=T["bg_surface"])
        self.lbl_enrolled.pack(side=tk.RIGHT)
        p_btns = tk.Frame(people, bg=T["bg_surface"])
        p_btns.pack(fill=tk.X)
        p_btns.columnconfigure((0, 1), weight=1, uniform="pb")
        Button(p_btns, "Enroll faces", variant="secondary", command=self.launch_enrollment_gui).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        Button(p_btns, "Reload database", variant="secondary", command=lambda: self.refresh_face_database(manual=True)).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self._update_enrolled_label()

        # ---- Right: incidents ----------------------------------------------------
        incidents = make_card(side)
        incidents.pack(fill=tk.BOTH, expand=True)
        i_hdr = tk.Frame(incidents, bg=T["bg_surface"])
        i_hdr.pack(fill=tk.X, pady=(0, 10))
        tk.Label(i_hdr, text="Incidents", font=F(10, "bold"), fg=T["text_primary"], bg=T["bg_surface"]).pack(side=tk.LEFT)
        Button(i_hdr, "Clear", variant="ghost", font=F(9, "bold"), padx=8, pady=2, command=lambda: self.log_view.clear()).pack(side=tk.RIGHT)
        Button(i_hdr, "Send test", variant="ghost", font=F(9, "bold"), padx=8, pady=2, command=self._send_test_alert).pack(side=tk.RIGHT, padx=(0, 2))

        i_btns = tk.Frame(incidents, bg=T["bg_surface"])
        i_btns.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0))
        i_btns.columnconfigure((0, 1), weight=1, uniform="ib")
        Button(i_btns, "Open snapshots", variant="secondary", command=self._open_outputs_dir).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        Button(i_btns, "Open CSV log", variant="secondary", command=self._open_incidents_csv).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.log_view = LogView(incidents)
        self.log_view.pack(fill=tk.BOTH, expand=True)

    def _set_zone_label(self, rect: List[int]) -> None:
        x1, y1, x2, y2 = rect
        self.lbl_roi_coords.config(text=f"Zone  {x1}, {y1}  –  {x2}, {y2}")

    def _update_enrolled_label(self) -> None:
        n = 0
        try:
            for entry in KNOWN_FACES_DIR.iterdir():
                if entry.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in entry.iterdir()):
                    n += 1
                elif entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                    n += 1
        except Exception:
            pass
        self.lbl_enrolled.config(text=f"{n} enrolled")


    def _send_test_alert(self) -> None:
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        dummy[:] = (30, 41, 59)
        cv2.putText(dummy, "TEST ALERT SNAPSHOT", (28, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 250, 252), 2, cv2.LINE_AA)
        self.alert_manager.emit(
            "SYSTEM",
            0,
            "TEST ALERT: Logging and snapshot check completed",
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

        T = THEME
        self.canvas.delete("all")
        self.current_tk_image = None

        cw = max(self.canvas.winfo_width(), self.canvas_w, 200)
        ch = max(self.canvas.winfo_height(), self.canvas_h, 200)
        cx, cy = cw // 2, ch // 2

        self.canvas.create_rectangle(0, 0, cw, ch, fill=T["bg_video"], outline="")
        self.canvas.create_text(cx, cy - 22, text="Surveillance is stopped", fill=T["text_primary"], font=F(15, "bold"))
        self.canvas.create_text(cx, cy + 8, text=f"Camera {CAMERA_INDEX} is idle. Press Start to begin monitoring.", fill=T["text_secondary"], font=F(10))
        self.canvas.create_text(cx, cy + 34, text="While running, drag on the video to redraw the zone.", fill=T["text_muted"], font=F(9))

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
        self._set_zone_label([fx1, fy1, fx2, fy2])

    def _on_mouse_up(self, event) -> None:
        self.is_drawing_roi = False
        if self.temp_drag_roi:
            fx1, fy1, fx2, fy2 = self.temp_drag_roi
            if abs(fx2 - fx1) > 20 and abs(fy2 - fy1) > 20:
                self.roi_frame = [fx1, fy1, fx2, fy2]
            self.temp_drag_roi = None
        self._set_zone_label(self.roi_frame)

    def reset_roi(self) -> None:
        w, h = self.frame_orig_w, self.frame_orig_h
        self.roi_frame = [int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)]
        self._set_zone_label(self.roi_frame)

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
            self._update_enrolled_label()
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

                zc = THEME["zone"]
                self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=zc, width=2, dash=dash_style)
                chip_w, chip_h = 112, 20
                chip_y = cy1 - chip_h if cy1 - chip_h >= 0 else cy1
                self.canvas.create_rectangle(cx1 - 1, chip_y, cx1 - 1 + chip_w, chip_y + chip_h, fill=zc, outline="")
                self.canvas.create_text(cx1 + 7, chip_y + chip_h // 2, text="Restricted zone", fill="#0d1117", anchor=tk.W, font=F(9, "bold"))

                self.lbl_fps_val.config(text=f"{fps_val:.1f}")
                self.lbl_targets_val.config(text=str(targets), fg=THEME["info"] if targets else THEME["text_primary"])
                self.lbl_breach_val.config(text=str(breaches_val), fg=THEME["danger"] if breaches_val else THEME["text_primary"])
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
        self.btn_toggle.config(text="Stop surveillance", variant="danger")
        self.lbl_header_status.config(text="● LIVE MONITORING", fg=THEME["success"])

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

        self.btn_toggle.config(text="Start surveillance", variant="primary")
        self.lbl_header_status.config(text="● STANDBY", fg=THEME["text_muted"])
        self.lbl_fps_val.config(text="0.0")
        self.lbl_targets_val.config(text="0", fg=THEME["text_primary"])
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
                        status_text = f"{p_entry['name']} (outside zone)"
                    else:
                        box_color = (0, 200, 255)
                        status_text = "Outside zone"
                elif state == "SCANNING":
                    box_color = (255, 180, 0)
                    status_text = "Verifying..."
                elif is_auth:
                    if is_fall:
                        box_color = (0, 140, 255)
                        status_text = f"{p_entry['name']} | FALL"
                    elif is_running_fast:
                        box_color = (0, 180, 255)
                        status_text = f"{p_entry['name']} | Running"
                    else:
                        box_color = (128, 222, 74)
                        status_text = f"Authorized: {p_entry['name']}"
                else:
                    box_color = (68, 68, 239)
                    base_lbl = "Face covered" if state == "OBSCURED" else "Intruder"
                    parts = [f"{base_lbl} {dwell_time:.1f}s"]
                    if is_loitering:
                        parts.append("Loitering")
                    if is_fall:
                        parts.append("FALL")
                    if is_running_fast:
                        parts.append("Running")
                    status_text = " | ".join(parts)

                draw_person_badge(frame, (bx1, by1, bx2, by2), tid, status_text, box_color)

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
                cv2.putText(snap, "RESTRICTED ZONE", (fx1 + 8, max(20, fy1 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 158, 11), 2, cv2.LINE_AA)
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
        self.log_view.add(entry)

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
