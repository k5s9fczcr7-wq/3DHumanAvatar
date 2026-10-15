"""
Face Tracker

Threaded MediaPipe FaceLandmarker tracker on the Logitech C920 webcam.
Captures at ~10fps and exposes thread-safe face state for downstream
consumers (face_recognizer.py: face_recognition; face_emotion.py: HSEmotion).

Uses the MediaPipe Tasks API (0.10.33+).
Runs in a daemon thread so it doesn't block the asyncio orchestrator.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger("face_tracker")

# ── Configuration ────────────────────────────────────────────────────────────

CAMERA_INDEX = 0               # Windows: first DirectShow/MSMF camera (C920). Linux: /dev/video0.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 10                # ~100ms per frame
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
FACE_ABSENT_TIMEOUT_S = 2.0   # face "gone" after this many seconds without detection

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "face_landmarker.task"
)


@dataclass
class FaceState:
    """Snapshot of the current face detection state. Thread-safe via copy."""
    present: bool = False
    bbox: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) in pixels
    landmarks: Optional[np.ndarray] = None              # (478, 3) normalized landmarks
    face_crop: Optional[np.ndarray] = None              # BGR crop of face region
    full_frame: Optional[np.ndarray] = None             # full BGR frame
    timestamp: float = 0.0
    fps: float = 0.0


class FaceTracker:
    """Threaded webcam face tracker using MediaPipe FaceLandmarker."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        target_fps: int = TARGET_FPS,
    ):
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._frame_interval = 1.0 / target_fps

        # Thread-safe state
        self._lock = threading.Lock()
        self._state = FaceState()
        self._last_face_time: float = 0.0

        # Control
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="face-tracker")
        self._thread.start()
        logger.info(f"Face tracker started (camera={self._camera_index}, target_fps={self._target_fps})")

    def stop(self):
        """Stop the capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Face tracker stopped")

    @property
    def state(self) -> FaceState:
        """Return a snapshot of the current face state."""
        with self._lock:
            return FaceState(
                present=self._state.present,
                bbox=self._state.bbox,
                landmarks=self._state.landmarks.copy() if self._state.landmarks is not None else None,
                face_crop=self._state.face_crop.copy() if self._state.face_crop is not None else None,
                full_frame=self._state.full_frame.copy() if self._state.full_frame is not None else None,
                timestamp=self._state.timestamp,
                fps=self._state.fps,
            )

    @property
    def face_present(self) -> bool:
        with self._lock:
            return self._state.present

    # ── Internal capture loop ────────────────────────────────────────────────

    def _run(self):
        """Main capture + detection loop (runs in thread)."""
        # Default Windows backend is MSMF (Media Foundation). Reliable IF the
        # device handle is in a clean state — never force-kill while the cam
        # is open, use graceful shutdown via `{"type":"shutdown"}` WS message
        # (handled in orchestrator._run_websocket_server).
        #
        # FOURCC=MJPG MUST be set before resolution: the C920 over USB 2.0 can't
        # sustain raw YUY2 at 720p (~27 MB/s), so MSMF returns E_PENDING on every
        # grab. MJPG is hardware-compressed and fits comfortably.
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {self._camera_index}")
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        logger.info(
            f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
            f"{cap.get(cv2.CAP_PROP_FPS):.0f}fps"
        )

        # MediaPipe FaceLandmarker (Tasks API)
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        landmarker = FaceLandmarker.create_from_options(options)

        fps_counter = _FPSCounter()
        frame_ts_ms = 0  # monotonic timestamp for VIDEO mode

        try:
            while self._running:
                loop_start = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read failed, retrying...")
                    time.sleep(0.1)
                    continue

                # MediaPipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                frame_ts_ms += int(self._frame_interval * 1000)
                results = landmarker.detect_for_video(mp_image, frame_ts_ms)

                now = time.monotonic()
                fps_counter.tick()

                if results.face_landmarks:
                    lm_list = results.face_landmarks[0]
                    bbox, lm_array = self._extract_face_data(lm_list, frame)
                    face_crop = self._crop_face(frame, bbox)
                    self._last_face_time = now

                    with self._lock:
                        self._state.present = True
                        self._state.bbox = bbox
                        self._state.landmarks = lm_array
                        self._state.face_crop = face_crop
                        self._state.full_frame = frame
                        self._state.timestamp = now
                        self._state.fps = fps_counter.fps
                else:
                    # Face absent — keep present=True briefly to avoid flicker
                    timed_out = (now - self._last_face_time) > FACE_ABSENT_TIMEOUT_S
                    with self._lock:
                        if timed_out:
                            self._state.present = False
                            self._state.bbox = None
                            self._state.landmarks = None
                            self._state.face_crop = None
                        self._state.full_frame = frame
                        self._state.timestamp = now
                        self._state.fps = fps_counter.fps

                # Throttle to target FPS
                elapsed = time.monotonic() - loop_start
                sleep_time = self._frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception:
            logger.exception("Face tracker crashed")
        finally:
            landmarker.close()
            cap.release()
            logger.info("Camera released")

    @staticmethod
    def _extract_face_data(
        lm_list, frame: np.ndarray
    ) -> tuple[tuple[int, int, int, int], np.ndarray]:
        """Extract bounding box and landmark array from FaceLandmarker results."""
        h, w = frame.shape[:2]
        lm_array = np.array(
            [(lm.x, lm.y, lm.z) for lm in lm_list],
            dtype=np.float32,
        )

        # Bounding box from landmark extremes with 20% padding
        xs = lm_array[:, 0] * w
        ys = lm_array[:, 1] * h
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        pad_x = (x_max - x_min) * 0.2
        pad_y = (y_max - y_min) * 0.2
        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(w, int(x_max + pad_x))
        y2 = min(h, int(y_max + pad_y))

        return (x1, y1, x2 - x1, y2 - y1), lm_array

    @staticmethod
    def _crop_face(
        frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Crop face region from frame using bounding box."""
        x, y, bw, bh = bbox
        return frame[y : y + bh, x : x + bw].copy()


class _FPSCounter:
    """Simple rolling FPS counter."""

    def __init__(self, window: int = 30):
        self._times: list[float] = []
        self._window = window
        self.fps: float = 0.0

    def tick(self):
        now = time.monotonic()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times = self._times[-self._window :]
        if len(self._times) >= 2:
            dt = self._times[-1] - self._times[0]
            self.fps = (len(self._times) - 1) / dt if dt > 0 else 0.0
