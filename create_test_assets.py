#!/usr/bin/env python3
"""
create_test_assets.py
---------------------
Synthetic test-video generator for offline testing of the surveillance system.

Renders a 10-second 640×480 MP4 video with:
  - A dark background and a labeled yellow polygon ROI box.
  - A moving circle/rectangle that:
      1. Enters the ROI from the left.
      2. Pauses for ~5 seconds inside the ROI (simulates loitering).
      3. Turns horizontal (wide, short bbox — simulates a fall).
      4. Exits the ROI to the right.

Usage:
    python create_test_assets.py
"""

import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR: str = "test_media"
OUTPUT_FILE: str = os.path.join(OUTPUT_DIR, "synthetic_test.mp4")
WIDTH: int = 640
HEIGHT: int = 480
FPS: int = 30
DURATION_SEC: float = 10.0
TOTAL_FRAMES: int = int(FPS * DURATION_SEC)

# ROI polygon (must match surveillance_system.py defaults)
ROI_POLYGON: np.ndarray = np.array(
    [[100, 100], [500, 100], [500, 450], [100, 450]], np.int32
)


def create_output_dir() -> None:
    """Ensure the output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_roi(frame: np.ndarray) -> None:
    """Draw the labeled yellow ROI polygon on *frame*."""
    cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.putText(
        frame, "ROI ZONE", (110, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )


def generate_synthetic_video() -> str:
    """
    Generate the synthetic test video and return its file path.

    Animation phases (frame ranges approximate):
      Phase 1 (0 – 59):    Circle enters from left, moves toward ROI center.
      Phase 2 (60 – 209):  Circle pauses inside ROI (~5 s loitering).
      Phase 3 (210 – 249): Shape becomes wide rectangle (fall simulation).
      Phase 4 (250 – 299): Rectangle exits to the right.
    """
    create_output_dir()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {OUTPUT_FILE}")

    # Timing boundaries (in frames)
    enter_end = 60           # frames 0-59:   entering
    loiter_end = 210         # frames 60-209: loitering (~5 s)
    fall_end = 250           # frames 210-249: fall
    # frames 250-299: exit

    # Starting / target positions
    start_x, start_y = 30, 275       # left of frame
    roi_cx, roi_cy = 300, 275        # center of ROI
    exit_x = 620                     # right edge

    for f in range(TOTAL_FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # dark grey background

        # --- Draw ROI ---
        draw_roi(frame)

        # --- Determine shape position and dimensions per phase ---
        if f < enter_end:
            # Phase 1: linear interpolation from start to ROI center
            t = f / max(enter_end - 1, 1)
            cx = int(start_x + t * (roi_cx - start_x))
            cy = int(start_y + t * (roi_cy - start_y))
            w, h = 50, 100  # upright person-like rectangle
            shape = "rect"
            label = "ENTERING"
            color = (0, 200, 255)  # orange

        elif f < loiter_end:
            # Phase 2: stationary inside ROI (loitering)
            cx, cy = roi_cx, roi_cy
            w, h = 50, 100
            shape = "rect"
            label = "LOITERING"
            color = (0, 0, 255)  # red

        elif f < fall_end:
            # Phase 3: aspect ratio flips — simulating a fall
            cx, cy = roi_cx, roi_cy + 30
            w, h = 120, 40  # wide & short → W/H > 1.2
            shape = "rect"
            label = "FALL"
            color = (0, 0, 255)  # red

        else:
            # Phase 4: exit to the right
            t = (f - fall_end) / max(TOTAL_FRAMES - fall_end - 1, 1)
            cx = int(roi_cx + t * (exit_x - roi_cx))
            cy = roi_cy
            w, h = 50, 100
            shape = "rect"
            label = "EXITING"
            color = (200, 200, 0)  # cyan-ish

        # --- Draw shape ---
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        # Small circle as "head"
        head_radius = 18
        cv2.circle(frame, (cx, y1 - head_radius - 2), head_radius, color, -1)

        # --- Label ---
        cv2.putText(
            frame, label, (x1, y1 - head_radius * 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        # --- Frame counter & phase indicator ---
        cv2.putText(
            frame,
            f"Frame {f}/{TOTAL_FRAMES}  |  Phase: {label}",
            (10, HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
        )

        writer.write(frame)

    writer.release()
    print(f"[✓] Synthetic test video saved → {OUTPUT_FILE}  "
          f"({TOTAL_FRAMES} frames, {DURATION_SEC}s @ {FPS} FPS)")
    return OUTPUT_FILE


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_synthetic_video()

