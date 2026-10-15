"""
Gaze Calculator

Where the visitor is, seen from the avatar's eyes on the physical screen, and how far her eyes should turn toward
them.

Unreal (UEyeGazeComponent) aims her eyes at the camera that films her - on a flat screen a gaze straight into
that camera reads as "she is looking at me" from anywhere in front of it - and turns them by the offset sent from
here, so her eyes visibly follow the visitor without leaving them:

    offset (degrees, + = her left / up) = follow gain x (angle of the visitor seen from her eyes on the screen
                                          - that angle for a visitor at the standing spot) + trim

The standing spot is directly in front of the camera ([tracking] follow_center = spot), so a visitor standing
there gets eyes straight into the camera; the trim is a perceptual correction, set by eye.

follow gain 1.0 = her eyes point at the visitor's physical position as if the screen were a window; on a flat
screen that reads as looking past people who stand off to the side, so the default follows a quarter of the
angle.

Pipeline: face_tracker landmarks -> iris midpoint + interpupillary distance -> visitor position (inches) from the
camera's lens model, mounting position and tilt -> angle from her eyes on the screen -> offset for Unreal.

Physical frame (inches): x = toward the viewer's right (= her left), y = up, z = out of the screen toward the
viewer; the origin is her eyes on the screen. Her eye position comes from Unreal (it reports where it draws her
eyes, "eyes_on_screen") or from [avatar] in kiosk_config.ini. Every value lives in kiosk_config.ini and is
re-read when the file changes, so the geometry can be retuned while she runs.
"""

import configparser
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("gaze_calculator")

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "kiosk_config.ini"
)


@dataclass
class KioskGeometry:
    """Physical layout of the kiosk, loaded from INI (inches and degrees)."""

    # Camera
    fov_diagonal_deg: float
    capture_width: int
    capture_height: int
    cam_offset_from_left: float   # lens centre, inches from the LEFT edge of the viewable area (viewer's left)
    cam_offset_above_top: float   # lens centre, inches above the TOP edge of the viewable area

    # Monitor
    monitor_width: float          # viewable width in inches
    monitor_height: float         # viewable height in inches

    # Her eyes on the screen (inches from the bottom-left corner of the viewable area), from [avatar]
    eye_x: float
    eye_y: float

    # Tracking
    reference_ipd_mm: float
    left_iris_index: int
    right_iris_index: int
    gaze_smoothing: float
    max_yaw_deg: float            # the visitor angle is followed up to this far to the side...
    max_pitch_deg: float          # ...and up/down

    # Gaze follow
    cam_tilt_down_deg: float = 0.0   # camera pitched down toward the visitor (0 = level)
    image_mirrored: bool = False     # true if the camera driver delivers a mirror image
    follow_gain_yaw: float = 0.25    # share of the visitor's angle her eyes turn (0 = always straight into the camera)
    follow_gain_pitch: float = 0.25
    eye_position_source: str = "auto"  # auto = where Unreal draws her eyes; config = [avatar] values
    # Where the follow is zero (her eyes straight into the camera): "eyes" = a visitor straight out from her eyes on
    # the screen; "spot" = a visitor at the standing spot, directly in front of the camera with their eyes
    # spot_below_camera_in below the lens.
    follow_center: str = "eyes"
    spot_below_camera_in: float = 1.6
    # Perceptual trim added to every offset sent while she looks at someone, degrees (+ = her left / up).
    trim_yaw_deg: float = 0.0
    trim_pitch_deg: float = 0.0

    # Derived
    focal_length_px: float = 0.0  # computed from FOV + resolution

    def __post_init__(self):
        diag_px = math.sqrt(self.capture_width ** 2 + self.capture_height ** 2)
        half_fov_rad = math.radians(self.fov_diagonal_deg / 2)
        self.focal_length_px = diag_px / (2 * math.tan(half_fov_rad))

    @property
    def camera_from_eyes(self) -> np.ndarray:
        """The camera lens relative to her eyes on the screen (x right, y up, z out), inches - config eye position."""
        return camera_from_eyes(self, self.eye_x, self.eye_y)


def camera_from_eyes(g: KioskGeometry, eye_x: float, eye_y: float) -> np.ndarray:
    return np.array([g.cam_offset_from_left - eye_x, g.monitor_height + g.cam_offset_above_top - eye_y, 0.0])


def _f(section, key, default):
    try:
        return float(section.get(key, str(default)))
    except (TypeError, ValueError):
        return float(default)


def load_config(path: str = CONFIG_PATH) -> KioskGeometry:
    """Load kiosk geometry from INI file."""
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        raise FileNotFoundError(f"Kiosk config not found: {path}")

    cam = cfg["camera"]
    mon = cfg["monitor"]
    av = cfg["avatar"]
    trk = cfg["tracking"]

    # Her eyes from the face box (used when Unreal hasn't reported them, or eye_position = config)
    face_left = float(av["face_left"])
    face_right = float(av["face_right"])
    face_chin = float(av["face_chin"])
    face_top = float(av["face_top"])
    eye_x = float(av.get("eye_x_override", 0) or 0) or (face_left + face_right) / 2
    eye_y = float(av.get("eye_y_override", 0) or 0) or face_chin + 0.575 * (face_top - face_chin)

    geom = KioskGeometry(
        fov_diagonal_deg=float(cam["fov_diagonal_deg"]),
        capture_width=int(cam["capture_width"]),
        capture_height=int(cam["capture_height"]),
        cam_offset_from_left=float(cam["offset_from_left"]),
        cam_offset_above_top=float(cam["offset_above_top"]),
        monitor_width=float(mon["viewable_width"]),
        monitor_height=float(mon["viewable_height"]),
        eye_x=eye_x,
        eye_y=eye_y,
        reference_ipd_mm=float(trk["reference_ipd_mm"]),
        left_iris_index=int(trk["left_iris_index"]),
        right_iris_index=int(trk["right_iris_index"]),
        gaze_smoothing=float(trk["gaze_smoothing"]),
        max_yaw_deg=float(trk["max_yaw_deg"]),
        max_pitch_deg=float(trk["max_pitch_deg"]),
        cam_tilt_down_deg=_f(cam, "tilt_down_deg", 0.0),
        image_mirrored=cam.get("mirrored", "false").strip().lower() in ("1", "true", "yes", "on"),
        follow_gain_yaw=_f(trk, "follow_gain_yaw", 0.25),
        follow_gain_pitch=_f(trk, "follow_gain_pitch", 0.25),
        eye_position_source=av.get("eye_position", "auto").strip().lower() or "auto",
        follow_center="spot" if trk.get("follow_center", "eyes").strip().lower() == "spot" else "eyes",
        spot_below_camera_in=_f(trk, "spot_below_camera_in", 1.6),
        trim_yaw_deg=_f(trk, "trim_yaw_deg", 0.0),
        trim_pitch_deg=_f(trk, "trim_pitch_deg", 0.0),
    )

    c = geom.camera_from_eyes
    logger.info(
        f"Kiosk geometry loaded: focal_length={geom.focal_length_px:.0f}px, tilt {geom.cam_tilt_down_deg:.1f} deg, "
        f"eyes (config) ({geom.eye_x:.1f}\", {geom.eye_y:.1f}\"), camera from eyes ({c[0]:+.1f}\", {c[1]:+.1f}\"), "
        f"follow gain {geom.follow_gain_yaw:.2f}/{geom.follow_gain_pitch:.2f} centred on the {geom.follow_center}, "
        f"trim {geom.trim_yaw_deg:+.1f}/{geom.trim_pitch_deg:+.1f} deg, eye position {geom.eye_position_source}"
    )
    return geom


@dataclass
class GazeResult:
    """Output of the gaze calculator."""
    yaw_deg: float = 0.0      # offset to send: + = turn her eyes to her left (the viewer's right)
    pitch_deg: float = 0.0    # + = up
    distance_in: float = 0.0  # visitor's distance from the camera, inches
    focus_in: float = 0.0     # the visitor's eyes' distance from HER on-screen eyes, inches: where her eyes converge
    angle_yaw_deg: float = 0.0    # the visitor's own angle from her eyes, before the follow gain
    angle_pitch_deg: float = 0.0
    valid: bool = False


class GazeCalculator:
    """Converts face tracker state into gaze offsets for the avatar."""

    def __init__(self, geom: KioskGeometry, config_path: str = CONFIG_PATH):
        self._geom = geom
        self._path = config_path
        self._mtime = self._file_mtime()
        self._next_check = 0.0
        self._eyes_ue: Optional[tuple] = None      # (x_in, y_in) where Unreal draws her eyes
        self._smooth_yaw: float = 0.0
        self._smooth_pitch: float = 0.0
        self._smooth_focus: float = 0.0
        self._initialized = False

    @property
    def geometry(self) -> KioskGeometry:
        return self._geom

    # ── configuration ──────────────────────────────────────────────────

    def _file_mtime(self) -> float:
        try:
            return os.path.getmtime(self._path)
        except OSError:
            return 0.0

    def maybe_reload(self, now: float) -> bool:
        """Re-read kiosk_config.ini when it changed (checked every 2 s), so geometry and gains tune live."""
        if now < self._next_check:
            return False
        self._next_check = now + 2.0
        m = self._file_mtime()
        if not m or m == self._mtime:
            return False
        self._mtime = m
        try:
            self._geom = load_config(self._path)
            logger.info("Gaze geometry reloaded from kiosk_config.ini")
            return True
        except Exception as e:                     # a half-saved file: keep the old geometry
            logger.warning(f"Gaze geometry not reloaded ({e}); keeping the previous values")
            return False

    def set_eyes_on_screen(self, x_px: float, y_px: float, width_px: float, height_px: float) -> None:
        """Unreal's report of where it draws her eyes (viewport pixels; the viewport fills the viewable area)."""
        if width_px <= 0 or height_px <= 0:
            return
        g = self._geom
        eyes = (x_px / width_px * g.monitor_width, (1.0 - y_px / height_px) * g.monitor_height)
        if self._eyes_ue is None or abs(eyes[0] - self._eyes_ue[0]) + abs(eyes[1] - self._eyes_ue[1]) > 0.3:
            logger.info(f"Her eyes on the screen (from Unreal): {eyes[0]:.1f}\" from the left, {eyes[1]:.1f}\" up "
                        f"({x_px:.0f}, {y_px:.0f} px of {width_px:.0f}x{height_px:.0f}); config said "
                        f"{g.eye_x:.1f}\", {g.eye_y:.1f}\"")
        self._eyes_ue = eyes

    def eye_position(self) -> tuple:
        g = self._geom
        if g.eye_position_source == "auto" and self._eyes_ue is not None:
            return self._eyes_ue
        return g.eye_x, g.eye_y

    # ── geometry ──────────────────────────────────────────────────────

    def visitor_position(self, cx_px: float, cy_px: float, depth_in: float) -> np.ndarray:
        """The visitor's eyes relative to HER eyes on the screen (x right, y up, z out), inches.

        The camera looks out of the screen (+z) toward the visitor, so its image-right is the viewer's LEFT (-x)
        unless the driver mirrors the picture; image-down is -y; a downward tilt turns its axis toward -y.
        depth_in = distance along the camera's axis (what the interpupillary-distance estimate measures)."""
        g = self._geom
        xr = (cx_px - g.capture_width / 2) / g.focal_length_px
        yr = (cy_px - g.capture_height / 2) / g.focal_length_px
        if g.image_mirrored:
            xr = -xr
        t = math.radians(g.cam_tilt_down_deg)
        right = np.array([-1.0, 0.0, 0.0])
        down = np.array([0.0, -math.cos(t), -math.sin(t)])
        forward = np.array([0.0, -math.sin(t), math.cos(t)])
        cam = camera_from_eyes(g, *self.eye_position())
        return cam + depth_in * (forward + xr * right + yr * down)

    def follow_reference(self, depth_in: float) -> tuple:
        """The visitor angle (yaw, pitch) at which the follow is zero and her eyes go straight into the camera.

        follow_center = spot: a visitor at the kiosk's standing spot, directly in front of the camera with their eyes
        spot_below_camera_in below it, at the given distance from the screen. A visitor there is seen ~10 deg above
        her eyes (for example, with the camera 165 cm above the floor and the visitor's eyes at 161 cm), so a follow
        centred on her eyes would turn her eyes up toward them as soon as tracking begins."""
        g = self._geom
        if g.follow_center != "spot":
            return 0.0, 0.0
        cam = camera_from_eyes(g, *self.eye_position())
        x, y, z = float(cam[0]), float(cam[1]) - g.spot_below_camera_in, max(float(depth_in), 1.0)
        return math.degrees(math.atan2(x, z)), math.degrees(math.atan2(y, math.hypot(x, z)))

    def looking_into_camera(self) -> tuple:
        """The offset that reads as looking straight at whoever watches through the camera (a video call)."""
        return self._geom.trim_yaw_deg, self._geom.trim_pitch_deg

    def update(self, landmarks: Optional[np.ndarray]) -> GazeResult:
        """Compute the gaze offset from face landmarks ((478, 3) normalized array, or None if no face)."""
        if landmarks is None or len(landmarks) < 474:
            return GazeResult(valid=False)

        g = self._geom
        left_iris = landmarks[g.left_iris_index]   # (x, y, z) normalized [0..1]
        right_iris = landmarks[g.right_iris_index]

        # Interpupillary distance in pixels -> distance along the camera axis (pinhole model)
        ipd_px = math.hypot((left_iris[0] - right_iris[0]) * g.capture_width,
                            (left_iris[1] - right_iris[1]) * g.capture_height)
        if ipd_px < 5:  # too small - bad detection
            return GazeResult(valid=False)
        depth_in = (g.reference_ipd_mm / 25.4) * g.focal_length_px / ipd_px

        cx_px = (left_iris[0] + right_iris[0]) / 2 * g.capture_width
        cy_px = (left_iris[1] + right_iris[1]) / 2 * g.capture_height
        v = self.visitor_position(cx_px, cy_px, depth_in)

        raw_yaw = math.degrees(math.atan2(v[0], v[2]))
        raw_pitch = math.degrees(math.atan2(v[1], math.hypot(v[0], v[2])))
        raw_focus = float(np.linalg.norm(v))

        alpha = g.gaze_smoothing
        if not self._initialized:
            self._smooth_yaw, self._smooth_pitch, self._smooth_focus = raw_yaw, raw_pitch, raw_focus
            self._initialized = True
        else:
            self._smooth_yaw = alpha * self._smooth_yaw + (1 - alpha) * raw_yaw
            self._smooth_pitch = alpha * self._smooth_pitch + (1 - alpha) * raw_pitch
            self._smooth_focus = alpha * self._smooth_focus + (1 - alpha) * raw_focus

        ang_yaw = max(-g.max_yaw_deg, min(g.max_yaw_deg, self._smooth_yaw))
        ang_pitch = max(-g.max_pitch_deg, min(g.max_pitch_deg, self._smooth_pitch))
        ref_yaw, ref_pitch = self.follow_reference(v[2])
        return GazeResult(
            yaw_deg=round(g.follow_gain_yaw * (ang_yaw - ref_yaw) + g.trim_yaw_deg, 2),
            pitch_deg=round(g.follow_gain_pitch * (ang_pitch - ref_pitch) + g.trim_pitch_deg, 2),
            distance_in=round(float(np.linalg.norm(v - camera_from_eyes(g, *self.eye_position()))), 1),
            focus_in=round(self._smooth_focus, 1),
            angle_yaw_deg=round(ang_yaw, 1),
            angle_pitch_deg=round(ang_pitch, 1),
            valid=True,
        )

    def reset(self):
        """Reset smoothing state (e.g., when user leaves and returns)."""
        self._initialized = False
        self._smooth_yaw = 0.0
        self._smooth_pitch = 0.0
        self._smooth_focus = 0.0
