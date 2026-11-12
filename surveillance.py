#!/usr/bin/env python3
"""
surveillance.py
-------------------
Modern Graphical User Interface for the AI Surveillance & Anomaly Detection System.

Key Updates & Features:
  - Clean, modern, accessible design matching the Personnel Enrollment Studio palette
    (Slate 900/800/700, crisp typography, generous whitespace, no garish colors).
  - True Aspect Ratio Video Viewport: Dynamic letterboxing and proportional scaling
    ensuring webcam streams (4:3, 16:9, etc.) are rendered without distortion or compression.
  - Interactive ROI Tripwire: Click and drag directly on the video canvas to define the
    threat zone; coordinates accurately map to raw frame resolution.
  - Asynchronous Face Verification Engine:
      * Uses Facenet embeddings with YOLOv8 face detector (with seamless fallback).
      * Bypasses the broken OpenCV 5.0.0 CascadeClassifier issue.
      * Background thread worker ensures zero video stutter or framerate drops during recognition.
  - Anomaly & Threat Detection:
      * Loitering detection (configurable dwell time threshold).
      * Fall / distress detection (bounding box aspect ratio anomaly).
      * Automatic threat snapshots saved into './outputs/'.
  - Real-time HUD and Security Alert Logs with time-stamped breach events.
  - Dedicated "Exit Dashboard" button for graceful camera release and shutdown.
"""

from __future__ import annotations

import os
import sys
import time
import glob
import queue
import tempfile
import threading
import subprocess
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Global Configurations
# ---------------------------------------------------------------------------
KNOWN_FACES_DIR: str = "./known_faces"
OUTPUTS_DIR: str = "./outputs"
TEST_MEDIA_DIR: str = "./test_media"

LOITERING_THRESHOLD_SEC: float = 4.0
FALL_ASPECT_RATIO_THRESHOLD: float = 1.65

# ---------------------------------------------------------------------------
# Design System (Minimalist, Professional Slate Palette)
# ---------------------------------------------------------------------------
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


class SurveillanceGUIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Surveillance & Security Threat Monitor")
        self.root.geometry("1280x800")
        self.root.minsize(1100, 720)
        self.root.configure(bg=THEME["bg_main"])

        self._ensure_directories()

        # Camera & Stream state
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.video_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Canvas display scaling state (to prevent aspect ratio compression)
        self.canvas_w = 780
        self.canvas_h = 560
        self.disp_scale: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.frame_orig_w: int = 640
        self.frame_orig_h: int = 480

        # ROI coordinates in raw video frame space: [x1, y1, x2, y2]
        self.roi_frame = [100, 60, 540, 420]
        self.is_drawing_roi: bool = False
        self.roi_drag_start = (0, 0)
        self.temp_drag_roi: Optional[List[int]] = None

        # Tracking and Verification Database
        self.model = None
        self.person_tracker_db: Dict[int, Dict[str, Any]] = {}
        self.verifying_ids: set = set()
        self.total_breaches: int = 0
        self.fps: float = 0.0

        # Thread-safe UI Queues
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.log_queue: queue.Queue = queue.Queue()
        self.status_queue: queue.Queue = queue.Queue()
        self.current_tk_image = None

        self._configure_styles()
        self._build_ui()
        self._start_polling_loop()
        self._load_yolo_async()
        self.root.after(50, self._render_standby_screen)

    def _start_polling_loop(self):
        """Main GUI thread poll loop for receiving thread-safe updates."""
        # 1. Process log messages
        try:
            while not self.log_queue.empty():
                log_msg = self.log_queue.get_nowait()
                self._add_log_entry(log_msg)
        except Exception:
            pass

        # 2. Process status label updates
        try:
            while not self.status_queue.empty():
                text, color = self.status_queue.get_nowait()
                self.lbl_header_status.config(text=text, fg=color)
        except Exception:
            pass

        # 3. Process video frame render
        try:
            latest_frame = None
            while not self.frame_queue.empty():
                latest_frame = self.frame_queue.get_nowait()

            if latest_frame is not None and self.canvas.winfo_exists() and self.is_running:
                canvas_img, fps_val, targets = latest_frame
                img_pil = Image.fromarray(canvas_img)
                self.current_tk_image = ImageTk.PhotoImage(image=img_pil, master=self.canvas)

                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_tk_image)

                # Overlay tripwire rectangle
                active_roi = self.temp_drag_roi if self.temp_drag_roi else self.roi_frame
                cx1, cy1, cx2, cy2 = self._frame_to_canvas_rect(active_roi)
                dash_style = (4, 4) if self.temp_drag_roi else None

                self.canvas.create_rectangle(
                    cx1, cy1, cx2, cy2,
                    outline="#38bdf8",
                    width=2,
                    dash=dash_style,
                )
                self.canvas.create_text(
                    cx1 + 6, cy1 + 12,
                    text="Tripwire Zone",
                    fill="#38bdf8",
                    anchor=tk.W,
                    font=("Segoe UI", 9, "bold"),
                )

                self.lbl_fps_val.config(text=f"{fps_val:.1f}")
                self.lbl_targets_val.config(text=str(targets))
                self.lbl_breach_val.config(text=str(self.total_breaches))
        except Exception as e:
            pass

        if self.root.winfo_exists():
            self.root.after(16, self._start_polling_loop)

    def _ensure_directories(self):
        for d in [KNOWN_FACES_DIR, OUTPUTS_DIR, TEST_MEDIA_DIR, "models"]:
            os.makedirs(d, exist_ok=True)

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Custom Combobox
        style.configure(
            "Custom.TCombobox",
            fieldbackground=THEME["bg_input"],
            background=THEME["bg_surface"],
            foreground=THEME["text_primary"],
            arrowcolor=THEME["text_secondary"],
            bordercolor=THEME["border"],
            lightcolor=THEME["border"],
            darkcolor=THEME["border"],
            padding=5,
        )
        style.map(
            "Custom.TCombobox",
            fieldbackground=[("readonly", THEME["bg_input"])],
            selectbackground=[("readonly", THEME["bg_card"])],
            selectforeground=[("readonly", THEME["text_primary"])],
        )

        # Custom Scrollbar
        style.configure(
            "Custom.Vertical.TScrollbar",
            background=THEME["bg_surface"],
            troughcolor=THEME["bg_main"],
            bordercolor=THEME["border"],
            arrowcolor=THEME["text_muted"],
            borderwidth=0,
            arrowsize=12,
        )

    def _build_ui(self):
        # -------------------------------------------------------------------
        # 1. Header Navigation Bar
        # -------------------------------------------------------------------
        header = tk.Frame(self.root, bg=THEME["bg_surface"], height=64, padx=24)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        header_left = tk.Frame(header, bg=THEME["bg_surface"])
        header_left.pack(side=tk.LEFT, fill=tk.Y)

        title_lbl = tk.Label(
            header_left,
            text="AI Surveillance System",
            font=("Segoe UI", 15, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_surface"],
        )
        title_lbl.pack(side=tk.LEFT, pady=16)

        subtitle_lbl = tk.Label(
            header_left,
            text=" |  Autonomous Anomaly Detection & Threat Monitoring",
            font=("Segoe UI", 11),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
        )
        subtitle_lbl.pack(side=tk.LEFT, pady=16)

        # Header Right Controls: Status & Exit Button
        header_right = tk.Frame(header, bg=THEME["bg_surface"])
        header_right.pack(side=tk.RIGHT, fill=tk.Y)

        self.lbl_header_status = tk.Label(
            header_right,
            text="● STANDBY",
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

        # -------------------------------------------------------------------
        # 2. Main Content Split View
        # -------------------------------------------------------------------
        content = tk.Frame(self.root, bg=THEME["bg_main"], padx=20, pady=18)
        content.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANEL: Aspect-Ratio Preserving Surveillance Canvas ---
        left_panel = tk.Frame(content, bg=THEME["bg_surface"], padx=16, pady=16)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        # Viewport Header Bar
        cam_hdr = tk.Frame(left_panel, bg=THEME["bg_surface"])
        cam_hdr.pack(fill=tk.X, pady=(0, 10))

        lbl_cam_title = tk.Label(
            cam_hdr,
            text="Live Surveillance Stream",
            font=("Segoe UI", 12, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_surface"],
        )
        lbl_cam_title.pack(side=tk.LEFT)

        self.lbl_aspect_info = tk.Label(
            cam_hdr,
            text="Aspect Ratio: Preserved (4:3 / 16:9) | Drag on feed to set Tripwire ROI",
            font=("Segoe UI", 9),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
        )
        self.lbl_aspect_info.pack(side=tk.RIGHT)

        # Video Canvas Container
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

        # Bind mouse events on canvas for interactive ROI dragging
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Video Action Toolbar below Canvas
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

        # --- RIGHT PANEL: Controls, Telemetry & Security Alert Logs ---
        right_panel = tk.Frame(content, bg=THEME["bg_main"], width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)

        # Card 1: Live Hardware Feed & Personnel Database
        source_card = tk.Frame(right_panel, bg=THEME["bg_surface"], padx=16, pady=14)
        source_card.pack(fill=tk.X, pady=(0, 14))

        tk.Label(
            source_card,
            text="VIDEO INPUT & PERSONNEL ACCESS",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
        ).pack(anchor=tk.W)

        # Clean Live Camera Indicator Box
        cam_status_box = tk.Frame(
            source_card,
            bg=THEME["bg_input"],
            padx=12,
            pady=10,
            highlightbackground=THEME["border"],
            highlightthickness=1,
        )
        cam_status_box.pack(fill=tk.X, pady=(8, 12))

        tk.Label(
            cam_status_box,
            text="🎥  Live Camera Feed (Device 0)",
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

        # Studio & DB Actions in Source Card
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
            command=self.refresh_face_database,
        )
        btn_reload_db.pack(side=tk.RIGHT, padx=(6, 0))

        # Card 2: Real-time Telemetry & Metrics
        metrics_card = tk.Frame(right_panel, bg=THEME["bg_surface"], padx=16, pady=12)
        metrics_card.pack(fill=tk.X, pady=(0, 14))

        tk.Label(
            metrics_card,
            text="SYSTEM TELEMETRY",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
        ).pack(anchor=tk.W, pady=(0, 8))

        stats_row = tk.Frame(metrics_card, bg=THEME["bg_surface"])
        stats_row.pack(fill=tk.X)

        # Stat 1: FPS
        fps_box = tk.Frame(stats_row, bg=THEME["bg_card"], padx=10, pady=8)
        fps_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Label(fps_box, text="FPS", font=("Segoe UI", 8), fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack()
        self.lbl_fps_val = tk.Label(fps_box, text="0.0", font=("Segoe UI", 12, "bold"), fg=THEME["text_primary"], bg=THEME["bg_card"])
        self.lbl_fps_val.pack()

        # Stat 2: Targets in ROI
        targets_box = tk.Frame(stats_row, bg=THEME["bg_card"], padx=10, pady=8)
        targets_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(targets_box, text="IN TRIPWIRE", font=("Segoe UI", 8), fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack()
        self.lbl_targets_val = tk.Label(targets_box, text="0", font=("Segoe UI", 12, "bold"), fg="#38bdf8", bg=THEME["bg_card"])
        self.lbl_targets_val.pack()

        # Stat 3: Total Breaches
        breach_box = tk.Frame(stats_row, bg=THEME["bg_card"], padx=10, pady=8)
        breach_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        tk.Label(breach_box, text="BREACHES", font=("Segoe UI", 8), fg=THEME["text_secondary"], bg=THEME["bg_card"]).pack()
        self.lbl_breach_val = tk.Label(breach_box, text="0", font=("Segoe UI", 12, "bold"), fg=THEME["danger"], bg=THEME["bg_card"])
        self.lbl_breach_val.pack()

        # Card 3: Live Security Alerts & Incident Log
        log_card = tk.Frame(right_panel, bg=THEME["bg_surface"], padx=16, pady=14)
        log_card.pack(fill=tk.BOTH, expand=True)

        log_hdr = tk.Frame(log_card, bg=THEME["bg_surface"])
        log_hdr.pack(fill=tk.X, pady=(0, 8))

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
        log_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

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

        # Output folder button
        btn_open_outputs = tk.Button(
            log_card,
            text="📂  Open Incident Snapshots Folder",
            font=("Segoe UI", 9, "bold"),
            bg=THEME["bg_card"],
            fg=THEME["text_primary"],
            activebackground=THEME["neutral_btn"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            pady=8,
            command=self._open_outputs_dir,
        )
        btn_open_outputs.pack(fill=tk.X)

    # -----------------------------------------------------------------------
    # Interactive Aspect Ratio & Canvas ROI Handling
    # -----------------------------------------------------------------------
    def _on_canvas_resize(self, event):
        self.canvas_w = max(event.width, 100)
        self.canvas_h = max(event.height, 100)
        if not self.is_running:
            self._render_standby_screen()

    def _render_standby_screen(self):
        """Draws a clean, professional standby display when live stream is stopped."""
        if not self.canvas.winfo_exists():
            return

        self.canvas.delete("all")
        self.current_tk_image = None

        cw = max(self.canvas.winfo_width(), self.canvas_w, 200)
        ch = max(self.canvas.winfo_height(), self.canvas_h, 200)
        cx, cy = cw // 2, ch // 2

        # 1. Dark canvas background
        self.canvas.create_rectangle(0, 0, cw, ch, fill="#020617", outline="")

        # 2. Subtle surveillance crosshairs & concentric target rings in background
        for r in [220, 150, 80]:
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="#1e293b", width=1, dash=(3, 5)
            )
        self.canvas.create_line(cx - 260, cy, cx + 260, cy, fill="#1e293b", width=1, dash=(2, 4))
        self.canvas.create_line(cx, cy - 180, cx, cy + 180, fill="#1e293b", width=1, dash=(2, 4))

        # 3. Center Standby Card
        card_w, card_h = 420, 175
        self.canvas.create_rectangle(
            cx - card_w // 2, cy - card_h // 2,
            cx + card_w // 2, cy + card_h // 2,
            fill="#0f172a", outline="#334155", width=1
        )

        # 4. Status Indicator Badge
        badge_w, badge_h = 180, 26
        self.canvas.create_rectangle(
            cx - badge_w // 2, cy - 50 - badge_h // 2,
            cx + badge_w // 2, cy - 50 + badge_h // 2,
            fill="#1e293b", outline="#475569", width=1
        )
        self.canvas.create_text(
            cx, cy - 50,
            text="● SURVEILLANCE OFFLINE",
            fill="#94a3b8",
            font=("Segoe UI", 9, "bold"),
        )

        # 5. Main Card Content
        self.canvas.create_text(
            cx, cy - 12,
            text="Live Video Feed Paused",
            fill="#f8fafc",
            font=("Segoe UI", 15, "bold"),
        )

        self.canvas.create_text(
            cx, cy + 20,
            text='Click "▶ Start Surveillance" below to activate camera',
            fill="#94a3b8",
            font=("Segoe UI", 10),
        )

        self.canvas.create_text(
            cx, cy + 48,
            text="Tripwire ROI and AI threat detection are on standby",
            fill="#64748b",
            font=("Segoe UI", 8),
        )

    def _get_canvas_to_frame_scale(self) -> Tuple[float, int, int]:
        """Calculates scaling factor and offsets preserving true aspect ratio."""
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
        fx = int(np.clip(fx, 0, self.frame_orig_w))
        fy = int(np.clip(fy, 0, self.frame_orig_h))
        return fx, fy

    def _frame_to_canvas_rect(self, rect: List[int]) -> Tuple[int, int, int, int]:
        scale, off_x, off_y = self._get_canvas_to_frame_scale()
        x1, y1, x2, y2 = rect
        cx1 = int(x1 * scale) + off_x
        cy1 = int(y1 * scale) + off_y
        cx2 = int(x2 * scale) + off_x
        cy2 = int(y2 * scale) + off_y
        return cx1, cy1, cx2, cy2

    def _on_mouse_down(self, event):
        self.is_drawing_roi = True
        self.roi_drag_start = (event.x, event.y)
        self.temp_drag_roi = None

    def _on_mouse_drag(self, event):
        if not self.is_drawing_roi:
            return
        x1, y1 = self.roi_drag_start
        x2, y2 = event.x, event.y
        fx1, fy1 = self._canvas_to_frame_pt(min(x1, x2), min(y1, y2))
        fx2, fy2 = self._canvas_to_frame_pt(max(x1, x2), max(y1, y2))
        self.temp_drag_roi = [fx1, fy1, fx2, fy2]
        self.lbl_roi_coords.config(text=f"Tripwire: ({fx1},{fy1}) - ({fx2},{fy2})")

    def _on_mouse_up(self, event):
        self.is_drawing_roi = False
        if self.temp_drag_roi:
            fx1, fy1, fx2, fy2 = self.temp_drag_roi
            if abs(fx2 - fx1) > 20 and abs(fy2 - fy1) > 20:
                self.roi_frame = [fx1, fy1, fx2, fy2]
            self.temp_drag_roi = None
        self.lbl_roi_coords.config(
            text=f"Tripwire: ({self.roi_frame[0]},{self.roi_frame[1]}) - ({self.roi_frame[2]},{self.roi_frame[3]})"
        )

    def reset_roi(self):
        # Default center ROI inside native frame
        w, h = self.frame_orig_w, self.frame_orig_h
        self.roi_frame = [int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)]
        self.lbl_roi_coords.config(
            text=f"Tripwire: ({self.roi_frame[0]},{self.roi_frame[1]}) - ({self.roi_frame[2]},{self.roi_frame[3]})"
        )

    # -----------------------------------------------------------------------
    # Model Loading & Database Management
    # -----------------------------------------------------------------------
    def _load_yolo_async(self):
        def load():
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")

                # Preload Facenet model into memory so first verification has zero cold-start delay
                try:
                    from deepface import DeepFace
                    DeepFace.build_model("Facenet")
                except Exception as fe:
                    print(f"[DEBUG] Facenet model pre-warm: {fe}")

                self.status_queue.put(("● SYSTEM READY", THEME["success"]))
            except Exception as e:
                self.status_queue.put(("● YOLO LOAD ERROR", THEME["danger"]))
                print(f"[ERROR] Failed to load YOLO: {e}")

        threading.Thread(target=load, daemon=True).start()

    def launch_enrollment_gui(self):
        try:
            subprocess.Popen([sys.executable, "register_face.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Face Enrollment Studio:\n{e}")

    def refresh_face_database(self):
        """Invalidates DeepFace datastore/pickle caches and re-indexes."""
        try:
            count = 0
            for item in os.listdir(KNOWN_FACES_DIR):
                if item.endswith(".pkl") or item.endswith(".npz"):
                    try:
                        os.remove(os.path.join(KNOWN_FACES_DIR, item))
                        count += 1
                    except Exception:
                        pass
            # Clear cached verification state for any active tracking
            with self.lock:
                for tid in self.person_tracker_db:
                    self.person_tracker_db[tid]["auth_status"] = None
                    self.person_tracker_db[tid]["name"] = "UNKNOWN"
                    self.person_tracker_db[tid]["breach_logged"] = False
                    self.person_tracker_db[tid]["loiter_logged"] = False
                    self.person_tracker_db[tid]["fall_logged"] = False
            self.verifying_ids.clear()
            messagebox.showinfo("Face Database Refreshed", f"Cleared cached representations ({count} files). New face data will be indexed.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh face database: {e}")

    def _open_outputs_dir(self):
        try:
            os.startfile(os.path.abspath(OUTPUTS_DIR))
        except Exception:
            messagebox.showinfo("Outputs Directory", f"Snapshots stored at: {os.path.abspath(OUTPUTS_DIR)}")

    # -----------------------------------------------------------------------
    # Surveillance Lifecycle & Live Webcam Stream Thread
    # -----------------------------------------------------------------------
    def toggle_surveillance(self):
        if self.is_running:
            self.stop_surveillance()
        else:
            self.start_surveillance()

    def start_surveillance(self):
        # Dedicated live camera device 0
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Connection Error",
                "Unable to open live webcam (Device 0).\n\nPlease verify that your camera is plugged in, permissions are granted, and no other app (like Zoom or Camera app) is using it."
            )
            return

        self.is_running = True
        self.btn_toggle.config(
            text="⏹  Stop Surveillance",
            bg=THEME["danger"],
            activebackground=THEME["danger_hover"],
        )
        self.lbl_header_status.config(text="● LIVE MONITORING", fg=THEME["success"])
        self.lbl_cam_live_tag.config(text="● LIVE", fg=THEME["success"])

        self.video_thread = threading.Thread(target=self._process_stream_loop, daemon=True)
        self.video_thread.start()

    def stop_surveillance(self):
        self.is_running = False

        # Allow stream loop thread to gracefully exit before releasing hardware handle
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=0.3)

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # Empty any pending video frames so they do not overwrite the standby screen
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                pass

        self.btn_toggle.config(
            text="▶  Start Surveillance",
            bg=THEME["success"],
            activebackground=THEME["success_hover"],
        )
        self.lbl_header_status.config(text="● STANDBY", fg=THEME["text_muted"])
        self.lbl_cam_live_tag.config(text="● STANDBY", fg=THEME["text_muted"])
        self.lbl_fps_val.config(text="0.0")
        self.lbl_targets_val.config(text="0")

        # Immediately draw clean standby screen
        self._render_standby_screen()

    # -----------------------------------------------------------------------
    # Asynchronous Face Verification Worker
    # -----------------------------------------------------------------------
    def _verify_face_async(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], tid: int):
        """Runs DeepFace Facenet + YuNet facial landmark verification in background."""
        def worker():
            name = "INTRUDER"
            is_auth = False
            try:
                x1, y1, x2, y2 = bbox
                height = y2 - y1
                width = x2 - x1
                # Crop upper body/head region with generous margins for face detection
                head_y2 = int(y1 + height * 0.55)
                pad_x = int(width * 0.15)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1)
                cx2 = min(frame_bgr.shape[1], x2 + pad_x)
                cy2 = max(head_y2, y1 + 1)
                face_crop = frame_bgr[cy1:cy2, cx1:cx2]

                if face_crop.size > 0:
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                    os.close(tmp_fd)
                    cv2.imwrite(tmp_path, face_crop)

                    # Check if enrolled faces exist
                    has_known = False
                    for root, _, files in os.walk(KNOWN_FACES_DIR):
                        if any(f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) for f in files):
                            has_known = True
                            break

                    if has_known:
                        from deepface import DeepFace
                        try:
                            # 1. Require genuine human facial landmark detection via YuNet
                            results = DeepFace.find(
                                img_path=tmp_path,
                                db_path=KNOWN_FACES_DIR,
                                model_name="Facenet",
                                detector_backend="yunet",
                                enforce_detection=True,
                                silent=True,
                            )

                            if results and len(results) > 0 and not results[0].empty:
                                df = results[0]
                                best_dist = float(df.iloc[0]["distance"])
                                threshold = float(df.iloc[0]["threshold"])
                                match_full_path = str(df.iloc[0]["identity"])

                                # Strict biometric threshold matching
                                if best_dist <= threshold:
                                    rel_path = os.path.relpath(match_full_path, KNOWN_FACES_DIR)
                                    parts = rel_path.split(os.sep)
                                    recognized_name = parts[0].replace("_", " ") if len(parts) > 1 else os.path.splitext(parts[0])[0].replace("_", " ")
                                    name = recognized_name
                                    is_auth = True
                                    print(f"[BIOMETRIC VERIFIED] ID:{tid} matched '{name}' (dist: {best_dist:.3f} <= {threshold:.2f})")
                                else:
                                    name = "UNAUTHORIZED INTRUDER"
                                    is_auth = False
                                    print(f"[ACCESS DENIED] ID:{tid} face does not match authorized personnel (dist: {best_dist:.3f} > {threshold:.2f})")
                            else:
                                name = "UNAUTHORIZED INTRUDER"
                                is_auth = False
                                print(f"[ACCESS DENIED] ID:{tid} face not in authorized database")

                        except Exception as det_err:
                            # YuNet could not detect a valid human face (e.g. T-shirt covered, masked, turned away)
                            name = "FACE OBSCURED"
                            is_auth = False
                            print(f"[THREAT ALERT] ID:{tid} has NO visible face (covered/obscured): {det_err}")

                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

            except Exception as e:
                print(f"[VERIFY ERROR] ID:{tid}: {e}")

            finally:
                with self.lock:
                    if tid in self.person_tracker_db:
                        self.person_tracker_db[tid]["auth_status"] = is_auth
                        self.person_tracker_db[tid]["name"] = name
                    self.verifying_ids.discard(tid)

        threading.Thread(target=worker, daemon=True).start()

    # -----------------------------------------------------------------------
    # Main Streaming & Detection Loop
    # -----------------------------------------------------------------------
    def _process_stream_loop(self):
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
            self.fps = 1.0 / dt if dt > 0 else self.fps
            prev_time = current_time

            fh, fw = frame.shape[:2]
            self.frame_orig_w, self.frame_orig_h = fw, fh

            # Ensure ROI is clamped to current frame dimensions
            rx1, ry1, rx2, ry2 = self.roi_frame
            fx1 = max(0, min(rx1, fw - 1))
            fy1 = max(0, min(ry1, fh - 1))
            fx2 = max(fx1 + 10, min(rx2, fw))
            fy2 = max(fy1 + 10, min(ry2, fh))
            roi_polygon = np.array([[fx1, fy1], [fx2, fy1], [fx2, fy2], [fx1, fy2]], np.int32)

            # YOLO inference
            active_in_roi = 0
            boxes_data = []
            if self.model is not None:
                try:
                    res = self.model.track(frame, persist=True, classes=[0], verbose=False)
                    if res and res[0].boxes is not None and len(res[0].boxes):
                        for b in res[0].boxes:
                            bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
                            tid_tensor = b.id
                            if tid_tensor is None:
                                continue
                            tid = int(tid_tensor.item())
                            boxes_data.append((bx1, by1, bx2, by2, tid))
                except Exception as e:
                    print(f"[YOLO TRACK ERROR] {e}")

            # Process each tracked person
            for (bx1, by1, bx2, by2, tid) in boxes_data:
                centroid = (int((bx1 + bx2) / 2), by2)
                in_roi = cv2.pointPolygonTest(roi_polygon, (float(centroid[0]), float(centroid[1])), False) >= 0

                with self.lock:
                    if tid not in self.person_tracker_db:
                        self.person_tracker_db[tid] = {
                            "entry_time": current_time,
                            "auth_status": None,
                            "name": "UNKNOWN",
                            "breach_logged": False,
                            "loiter_logged": False,
                            "fall_logged": False,
                        }
                    p_entry = self.person_tracker_db[tid]

                # Trigger face verification as soon as person enters ROI
                if in_roi and p_entry["auth_status"] is None and tid not in self.verifying_ids:
                    self.verifying_ids.add(tid)
                    p_entry["entry_time"] = current_time
                    self._verify_face_async(frame.copy(), (bx1, by1, bx2, by2), tid)

                # Anomaly & Behavioral Metrics
                w = bx2 - bx1
                h = max(by2 - by1, 1)
                aspect_ratio = w / h
                dwell_time = current_time - p_entry["entry_time"] if in_roi else 0.0

                is_loitering = in_roi and (dwell_time > LOITERING_THRESHOLD_SEC)
                is_fall = in_roi and (aspect_ratio > FALL_ASPECT_RATIO_THRESHOLD) and (by2 > int(fh * 0.65)) and (w * h > 8000)
                is_auth = p_entry["auth_status"] is True
                is_verifying = tid in self.verifying_ids

                if in_roi:
                    active_in_roi += 1

                # Visual HUD styling based on security rules:
                # 1. Outside ROI
                if not in_roi:
                    if is_auth:
                        box_color = (74, 222, 128) # Emerald Green
                        status_text = f"AUTH: {p_entry['name']} (OUTSIDE)"
                    else:
                        box_color = (0, 200, 255) # Yellow/Slate
                        status_text = "OUTSIDE ZONE"

                # 2. Inside ROI & actively scanning
                elif is_verifying:
                    box_color = (255, 180, 0) # Cyan / Sky Blue
                    status_text = "SCANNING / VERIFYING..."

                # 3. Inside ROI & Verified Authorized Personnel (ALWAYS GREEN unless medical fall emergency)
                elif is_auth:
                    if is_fall:
                        box_color = (0, 140, 255) # Amber Warning
                        status_text = f"AUTH: {p_entry['name']} | ⚠️ FALL DETECTED!"
                    else:
                        box_color = (74, 222, 128) # Emerald Green (Remains permanently green for authorized staff)
                        status_text = f"AUTHORIZED: {p_entry['name']}"

                # 4. Inside ROI & Unauthorized Intruder / Obscured Face (ALWAYS CRIMSON RED)
                else:
                    box_color = (68, 68, 239) # Crimson Red
                    if p_entry["name"] == "FACE OBSCURED":
                        threat_label = f"FACE OBSCURED / INTRUDER ({dwell_time:.1f}s)"
                    else:
                        threat_label = f"UNAUTHORIZED INTRUDER ({dwell_time:.1f}s)"

                    threat_parts = [threat_label]
                    if is_loitering:
                        threat_parts.append("SUSPICIOUS LOITERING")
                    if is_fall:
                        threat_parts.append("FALL DETECTED!")
                    status_text = " | ".join(threat_parts)

                # Draw Bounding Box and High-Legibility Badge
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
                badge_text = f"ID:{tid} | {status_text}"
                (lw, lh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (bx1, by1 - lh - 8), (bx1 + lw + 8, by1), box_color, -1)
                cv2.putText(
                    frame,
                    badge_text,
                    (bx1 + 4, by1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (15, 23, 42), # Dark Slate text on bright badge
                    1,
                    cv2.LINE_AA,
                )

                # Incident & Security Breach Event Logging
                if in_roi and (not is_auth) and not is_verifying:
                    # Immediate breach upon unauthorized tripwire zone entry
                    if not p_entry.get("breach_logged", False):
                        p_entry["breach_logged"] = True
                        self.total_breaches += 1
                        ts_str = time.strftime("%Y%m%d_%H%M%S")
                        snap_name = f"ALERT_INTRUDER_ID{tid}_{ts_str}.jpg"
                        snap_path = os.path.join(OUTPUTS_DIR, snap_name)
                        cv2.imwrite(snap_path, frame)
                        reason = "Face covered / obscured" if p_entry["name"] == "FACE OBSCURED" else "Unauthorized target"
                        self.log_queue.put(f"[{time.strftime('%H:%M:%S')}] 🚨 INTRUSION: {reason} | ID:{tid}")

                    # Secondary alert for prolonged loitering
                    if is_loitering and not p_entry.get("loiter_logged", False):
                        p_entry["loiter_logged"] = True
                        self.log_queue.put(
                            f"[{time.strftime('%H:%M:%S')}] ⚠️ LOITERING: Intruder loitering ({dwell_time:.1f}s) | ID:{tid}"
                        )

                # Medical / Safety Alert for Fall Detection
                if in_roi and is_fall and not p_entry.get("fall_logged", False):
                    p_entry["fall_logged"] = True
                    ts_str = time.strftime("%Y%m%d_%H%M%S")
                    target_tag = f"Staff_{p_entry['name']}" if is_auth else "Intruder"
                    snap_name = f"ALERT_FALL_ID{tid}_{target_tag}_{ts_str}.jpg"
                    snap_path = os.path.join(OUTPUTS_DIR, snap_name)
                    cv2.imwrite(snap_path, frame)

                    if is_auth:
                        self.log_queue.put(
                            f"[{time.strftime('%H:%M:%S')}] 🚑 SAFETY ALERT: Fall detected | ID:{tid} (Staff: {p_entry['name']})"
                        )
                    else:
                        self.log_queue.put(
                            f"[{time.strftime('%H:%M:%S')}] 🚨 ALERT: Intruder fall detected | ID:{tid}"
                        )

            # Draw ROI Tripwire Polygon on video frame
            cv2.polylines(frame, [roi_polygon], isClosed=True, color=(245, 158, 11), thickness=2)
            cv2.putText(
                frame,
                "ACTIVE TRIPWIRE ZONE",
                (fx1 + 8, fy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 158, 11),
                2,
                cv2.LINE_AA,
            )

            # ---------------------------------------------------------------
            # Render Aspect-Ratio Preserved Frame to Canvas
            # ---------------------------------------------------------------
            cw, ch = self.canvas_w, self.canvas_h
            scale, off_x, off_y = self._get_canvas_to_frame_scale()
            disp_w = max(int(fw * scale), 1)
            disp_h = max(int(fh * scale), 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

            # Create full canvas background with letterbox
            canvas_img = np.zeros((ch, cw, 3), dtype=np.uint8)
            canvas_img[off_y:off_y + disp_h, off_x:off_x + disp_w] = rgb_resized

            # Push frame to queue for the main GUI thread polling loop
            try:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait((canvas_img, self.fps, active_in_roi))
            except Exception:
                pass

            time.sleep(0.015)

    def _add_log_entry(self, entry: str):
        self.log_listbox.insert(0, entry)
        if self.log_listbox.size() > 100:
            self.log_listbox.delete(100, tk.END)

    def close(self):
        self.stop_surveillance()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SurveillanceGUIApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
