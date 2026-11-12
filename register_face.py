#!/usr/bin/env python3
"""
register_face.py
------------------
Interactive Graphical User Interface for Personnel Face Enrollment.

Key Features & Updates:
  - Clean, modern, accessible design (neutral dark/slate palette, clear typography,
    generous spacing, no garish high-saturation colors).
  - Manual, on-demand shutter button ("📸 Capture Photo"): User can capture as many
    photos as needed at their own pace without arbitrary forced timers.
  - Case-sensitive folder persistence: If "Vivek" already exists, new captures
    cleanly append into the existing folder with sequential indices (e.g., Vivek_1, Vivek_2, ...).
  - Live photo counter showing existing and session-added photos for the entered name.
  - Dedicated "Exit Studio" button to safely release the camera and close the window.
  - Enrolled personnel management (live directory list, sample counts, deletion, and auto-cache refresh).
"""

from __future__ import annotations

import os
import re
import sys
import time
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image, ImageTk

KNOWN_FACES_DIR = "./known_faces"

# ---------------------------------------------------------------------------
# Design System (Minimalist, Professional Slate Palette)
# ---------------------------------------------------------------------------
THEME = {
    "bg_main": "#0f172a",        # Slate 900
    "bg_surface": "#1e293b",     # Slate 800
    "bg_card": "#334155",        # Slate 700
    "bg_input": "#0f172a",       # Slate 900
    "border": "#475569",         # Slate 600
    "text_primary": "#f8fafc",   # Slate 50
    "text_secondary": "#94a3b8", # Slate 400
    "text_muted": "#64748b",     # Slate 500
    "primary": "#2563eb",        # Royal Blue 600
    "primary_hover": "#1d4ed8",  # Royal Blue 700
    "success": "#16a34a",        # Emerald 600
    "success_hover": "#15803d",  # Emerald 700
    "danger": "#dc2626",         # Crimson 600
    "danger_hover": "#b91c1c",   # Crimson 700
    "neutral_btn": "#475569",    # Slate 600
    "neutral_hover": "#334155",  # Slate 700
}


class FaceEnrollmentApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Personnel Enrollment Studio")
        self.root.geometry("1080x720")
        self.root.minsize(960, 640)
        self.root.configure(bg=THEME["bg_main"])

        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        # Session tracking
        self.last_shutter_time = 0.0
        self.flash_alpha = 0.0

        self._configure_styles()
        self._build_ui()
        self._start_camera()
        self._refresh_enrolled_list()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Custom TEntry
        style.configure(
            "Custom.TEntry",
            fieldbackground=THEME["bg_input"],
            foreground=THEME["text_primary"],
            insertcolor=THEME["text_primary"],
            borderwidth=1,
            relief="solid",
        )

        # Custom Scrollbar
        style.configure(
            "Custom.Vertical.TScrollbar",
            background=THEME["bg_surface"],
            troughcolor=THEME["bg_main"],
            borderwidth=0,
            arrowsize=12,
        )

    def _build_ui(self):
        # 1. Header Navigation Bar
        header = tk.Frame(self.root, bg=THEME["bg_surface"], height=64, padx=24)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        header_left = tk.Frame(header, bg=THEME["bg_surface"])
        header_left.pack(side=tk.LEFT, fill=tk.Y)

        title_lbl = tk.Label(
            header_left,
            text="Personnel Enrollment Studio",
            font=("Segoe UI", 15, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_surface"],
        )
        title_lbl.pack(side=tk.LEFT, pady=16)

        subtitle_lbl = tk.Label(
            header_left,
            text=" |  Authorized Face Database Management",
            font=("Segoe UI", 11),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
        )
        subtitle_lbl.pack(side=tk.LEFT, pady=16)

        # Header Right: Exit Studio Button
        btn_exit = tk.Button(
            header,
            text="✕  Exit Studio",
            font=("Segoe UI", 10, "bold"),
            bg=THEME["neutral_btn"],
            fg=THEME["text_primary"],
            activebackground=THEME["neutral_hover"],
            activeforeground=THEME["text_primary"],
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=6,
            command=self.close,
        )
        btn_exit.pack(side=tk.RIGHT, pady=14)

        # 2. Main Content Split View
        content = tk.Frame(self.root, bg=THEME["bg_main"], padx=24, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANEL: Live Camera Stream & Capture ---
        left_panel = tk.Frame(content, bg=THEME["bg_surface"], padx=16, pady=16)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 16))

        # Camera Header Info
        cam_hdr = tk.Frame(left_panel, bg=THEME["bg_surface"])
        cam_hdr.pack(fill=tk.X, pady=(0, 12))

        lbl_cam_title = tk.Label(
            cam_hdr,
            text="Live Camera Preview",
            font=("Segoe UI", 12, "bold"),
            fg=THEME["text_primary"],
            bg=THEME["bg_surface"],
        )
        lbl_cam_title.pack(side=tk.LEFT)

        self.lbl_cam_indicator = tk.Label(
            cam_hdr,
            text="● CONNECTING...",
            font=("Segoe UI", 9, "bold"),
            fg=THEME["text_muted"],
            bg=THEME["bg_surface"],
        )
        self.lbl_cam_indicator.pack(side=tk.RIGHT)

        # Video Viewport
        video_container = tk.Frame(
            left_panel,
            bg=THEME["bg_main"],
            highlightbackground=THEME["border"],
            highlightthickness=1,
        )
        video_container.pack(fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Label(video_container, bg=THEME["bg_main"])
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Viewport Footer / Instructions
        guidance_lbl = tk.Label(
            left_panel,
            text="Capture clear photos across multiple natural angles (front, slight tilts, smile) for robust matching.",
            font=("Segoe UI", 9),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
            pady=8,
        )
        guidance_lbl.pack(fill=tk.X)

        # --- RIGHT PANEL: Controls & Enrolled Personnel Database ---
        right_panel = tk.Frame(content, bg=THEME["bg_main"], width=380)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)

        # Card 1: Capture Controller
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
            text="Full Name / Identifier (Case-Sensitive):",
            font=("Segoe UI", 10),
            fg=THEME["text_primary"],
            bg=THEME["bg_surface"],
        ).pack(anchor=tk.W, pady=(8, 4))

        self.name_entry = ttk.Entry(
            capture_card,
            font=("Segoe UI", 11),
            style="Custom.TEntry",
        )
        self.name_entry.pack(fill=tk.X, pady=(0, 10))
        self.name_entry.focus()
        self.name_entry.bind("<KeyRelease>", self._on_name_changed)

        # Dynamic sample status for this name
        self.lbl_person_stats = tk.Label(
            capture_card,
            text="Enter name to see existing photos.",
            font=("Segoe UI", 9),
            fg=THEME["text_secondary"],
            bg=THEME["bg_surface"],
            anchor=tk.W,
        )
        self.lbl_person_stats.pack(fill=tk.X, pady=(0, 14))

        # Main Capture Button (Manual Trigger, Any Count)
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
            wraplength=340,
            justify=tk.LEFT,
        )
        self.lbl_capture_feedback.pack(fill=tk.X, pady=(10, 0))

        # Card 2: Enrolled Directory List
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

        # Listbox container
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
            style="Custom.Vertical.TScrollbar",
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.personnel_listbox.config(yscrollcommand=scrollbar.set)

        # List Action Buttons
        list_actions = tk.Frame(list_card, bg=THEME["bg_surface"])
        list_actions.pack(fill=tk.X, pady=(12, 0))

        btn_delete = tk.Button(
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
        )
        btn_delete.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        btn_refresh = tk.Button(
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
        )
        btn_refresh.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 0))

    # -----------------------------------------------------------------------
    # Camera Stream Lifecycle
    # -----------------------------------------------------------------------
    def _start_camera(self):
        self.is_camera_running = True
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

    def _camera_loop(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            fallback = "test_media/synthetic_test.mp4"
            if os.path.exists(fallback):
                self.cap = cv2.VideoCapture(fallback)
                self.lbl_cam_indicator.config(text="● SYNTHETIC FEED", fg="#eab308")
            else:
                self.lbl_cam_indicator.config(text="● CAMERA OFFLINE", fg=THEME["danger"])
                return
        else:
            self.lbl_cam_indicator.config(text="● LIVE CAMERA", fg=THEME["success"])

        while self.is_camera_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.03)
                continue

            with self.lock:
                self.current_frame = frame.copy()

            # Optional visual shutter flash effect
            display_frame = frame.copy()
            if self.flash_alpha > 0:
                white_overlay = np.full_like(display_frame, 255)
                cv2.addWeighted(white_overlay, self.flash_alpha, display_frame, 1.0 - self.flash_alpha, 0, display_frame)
                self.flash_alpha = max(0.0, self.flash_alpha - 0.25)

            # Render to Tkinter Label Viewport
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            target_w = 640
            target_h = int(h * (target_w / w))
            rgb_resized = cv2.resize(rgb, (target_w, target_h))

            img_pil = Image.fromarray(rgb_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            def update_canvas(img=img_tk):
                if self.video_canvas.winfo_exists():
                    self.video_canvas.img_tk = img
                    self.video_canvas.configure(image=img)

            self.root.after(0, update_canvas)
            time.sleep(0.03)

    # -----------------------------------------------------------------------
    # Manual Photo Capture (Append to Case-Sensitive Folder)
    # -----------------------------------------------------------------------
    def _sanitize_name(self, raw_name: str) -> str:
        """Sanitize name while strictly preserving case."""
        trimmed = raw_name.strip()
        # Keep letters, numbers, hyphens, underscores; replace spaces with underscores
        safe = "".join(c for c in trimmed if c.isalnum() or c in (" ", "_", "-")).rstrip()
        safe = safe.replace(" ", "_")
        return safe

    def _get_person_dir(self, safe_name: str) -> str:
        return os.path.join(KNOWN_FACES_DIR, safe_name)

    def _get_existing_photos(self, safe_name: str) -> List[str]:
        person_dir = self._get_person_dir(safe_name)
        if not os.path.isdir(person_dir):
            return []
        return [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    def _on_name_changed(self, event=None):
        name = self.name_entry.get().strip()
        if not name:
            self.lbl_person_stats.config(text="Enter name to see existing photos.", fg=THEME["text_secondary"])
            return

        safe_name = self._sanitize_name(name)
        existing = self._get_existing_photos(safe_name)
        count = len(existing)

        if count > 0:
            self.lbl_person_stats.config(
                text=f"Folder '{safe_name}' exists: {count} photo(s) on disk. New clicks will append.",
                fg=THEME["text_secondary"],
            )
        else:
            self.lbl_person_stats.config(
                text=f"New person '{safe_name}' will be created on first capture.",
                fg=THEME["text_secondary"],
            )

    def capture_single_photo(self):
        raw_name = self.name_entry.get().strip()
        if not raw_name:
            messagebox.showwarning("Name Required", "Please enter a personnel name before capturing.")
            return

        with self.lock:
            if self.current_frame is None:
                messagebox.showerror("Error", "No camera frame available to capture.")
                return
            frame_to_save = self.current_frame.copy()

        # Verify that a clear human face is present before enrolling
        try:
            from deepface import DeepFace
            faces = DeepFace.extract_faces(frame_to_save, detector_backend="yunet", enforce_detection=True)
            if not faces or len(faces) == 0:
                self.lbl_capture_feedback.config(
                    text="⚠️ No clear face detected! Please face the camera directly.",
                    fg=THEME["danger"],
                )
                return
        except Exception:
            self.lbl_capture_feedback.config(
                text="⚠️ No clear face detected! Please face the camera directly.",
                fg=THEME["danger"],
            )
            return

        safe_name = self._sanitize_name(raw_name)
        person_dir = self._get_person_dir(safe_name)
        os.makedirs(person_dir, exist_ok=True)

        # Find the next available numeric index in the folder
        existing_files = self._get_existing_photos(safe_name)
        max_idx = 0
        pattern = re.compile(rf"^{re.escape(safe_name)}_(\d+)\.(?:jpg|jpeg|png|bmp)$", re.IGNORECASE)

        for filename in existing_files:
            match = pattern.match(filename)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
            else:
                max_idx = max(max_idx, len(existing_files))

        next_idx = max_idx + 1
        save_filename = f"{safe_name}_{next_idx}.jpg"
        save_path = os.path.join(person_dir, save_filename)

        # Save photo
        cv2.imwrite(save_path, frame_to_save)

        # Trigger shutter flash
        self.flash_alpha = 0.75

        # Update feedback & list
        total_photos = len(self._get_existing_photos(safe_name))
        self.lbl_capture_feedback.config(
            text=f"✓ Saved photo #{next_idx} → {save_filename}\nFolder '{safe_name}' now contains {total_photos} photo(s).",
            fg=THEME["success"],
        )
        self._on_name_changed()
        self._precompute_representations_cache()
        self._refresh_enrolled_list()

    def _precompute_representations_cache(self):
        """Invalidate stale representations cache so DeepFace picks up new photos seamlessly."""
        try:
            for item in os.listdir(KNOWN_FACES_DIR):
                if item.endswith(".pkl") or item.endswith(".npz"):
                    try:
                        os.remove(os.path.join(KNOWN_FACES_DIR, item))
                    except Exception:
                        pass
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Database Directory Management
    # -----------------------------------------------------------------------
    def _refresh_enrolled_list(self):
        self.personnel_listbox.delete(0, tk.END)
        if not os.path.exists(KNOWN_FACES_DIR):
            return

        entries = os.listdir(KNOWN_FACES_DIR)
        total_people = 0

        for entry in sorted(entries):
            entry_path = os.path.join(KNOWN_FACES_DIR, entry)
            # Folder representation: known_faces/<Person_Name>/
            if os.path.isdir(entry_path):
                img_count = len([
                    f for f in os.listdir(entry_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ])
                self.personnel_listbox.insert(tk.END, f"{entry}  ({img_count} photos)")
                total_people += 1
            # Single photo backwards compatibility
            elif entry.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                person_name = os.path.splitext(entry)[0]
                self.personnel_listbox.insert(tk.END, f"{person_name}  (1 photo)")
                total_people += 1

        self.lbl_total_enrolled.config(text=f"{total_people} People")

    def _on_list_select(self, event=None):
        sel = self.personnel_listbox.curselection()
        if not sel:
            return
        item_text = self.personnel_listbox.get(sel[0])
        # Extract folder name before the parenthetical count
        folder_name = item_text.split("  (")[0].strip()
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, folder_name)
        self._on_name_changed()

    def delete_selected_person(self):
        sel = self.personnel_listbox.curselection()
        if not sel:
            messagebox.showinfo("Select Person", "Please select a person from the list to remove.")
            return

        item_text = self.personnel_listbox.get(sel[0])
        person_name = item_text.split("  (")[0].strip()

        target_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        target_file = os.path.join(KNOWN_FACES_DIR, f"{person_name}.jpg")

        if messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to permanently delete all enrolled photos for '{person_name}'?"
        ):
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)
            if os.path.isfile(target_file):
                os.remove(target_file)

            self._precompute_representations_cache()
            self._refresh_enrolled_list()
            self.lbl_capture_feedback.config(
                text=f"Deleted '{person_name}' and updated face cache.",
                fg=THEME["text_secondary"],
            )
            self._on_name_changed()

    def close(self):
        """Gracefully release hardware resources and close."""
        self.is_camera_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceEnrollmentApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
