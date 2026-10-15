"""
Natural gaze: she glances away now and then and looks back.

A gaze that never leaves the visitor looks unnatural, so she looks away occasionally, timed by the conversation:
often and longer when nobody is talking, now and then while the visitor talks (as listeners do), steady eye contact
while she talks. Each glance is short, and then her eyes come back.

A glance is an offset (degrees, + = her left / up) added to where her eyes look; Unreal moves her eyes there at once
and turns her head a share of it (UEyeGazeComponent HeadShare), then both come back when the glance ends.

  listening  the visitor is talking: every listening_every_s, held listening_hold_s, mostly sideways or down
  thinking   she is preparing an answer: with thinking_chance, a glance up and to one side until she speaks
  quiet      someone is there and nobody is talking: every quiet_every_s, held quiet_hold_s
  speaking   she is talking: steady eye contact (speaking_every_s = 0), or rarely if set
  away       nobody there: no glances (her eyes rest in her head anyway)

[glances] in kiosk_config.ini sets every value ("a-b" = a random value between a and b, seconds or degrees);
enabled = false switches it off. Saved changes apply within a few seconds.
"""

import configparser
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Optional

from .gaze_calculator import CONFIG_PATH

logger = logging.getLogger("glance_planner")

MODES = ("listening", "thinking", "quiet", "speaking", "away")


@dataclass
class GlanceConfig:
    enabled: bool = True
    listening_every_s: tuple = (7.0, 14.0)
    listening_hold_s: tuple = (0.5, 1.2)
    quiet_every_s: tuple = (4.0, 9.0)
    quiet_hold_s: tuple = (0.8, 2.0)
    thinking_chance: float = 0.5
    thinking_hold_s: tuple = (0.6, 1.6)
    speaking_every_s: tuple = (0.0, 0.0)     # 0 = steady eye contact while she talks
    speaking_hold_s: tuple = (0.4, 0.8)
    size_deg: tuple = (7.0, 13.0)


def _range(text: str, default: tuple) -> tuple:
    try:
        parts = [float(p) for p in str(text).replace(" ", "").split("-") if p != ""]
    except ValueError:
        return default
    if not parts:
        return default
    lo, hi = parts[0], parts[-1]
    return (min(lo, hi), max(lo, hi))


def load_glance_config(path: str = CONFIG_PATH) -> GlanceConfig:
    cfg = GlanceConfig()
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except (configparser.Error, OSError) as e:
        logger.warning(f"[glances] not read ({e}); using defaults")
        return cfg
    if not parser.has_section("glances"):
        return cfg
    sec = parser["glances"]
    cfg.enabled = sec.get("enabled", "true").strip().lower() in ("1", "true", "yes", "on")
    for name in ("listening_every_s", "listening_hold_s", "quiet_every_s", "quiet_hold_s", "thinking_hold_s",
                 "speaking_every_s", "speaking_hold_s", "size_deg"):
        if name in sec:
            setattr(cfg, name, _range(sec[name], getattr(cfg, name)))
    try:
        cfg.thinking_chance = max(0.0, min(1.0, float(sec.get("thinking_chance", cfg.thinking_chance))))
    except ValueError:
        pass
    return cfg


@dataclass
class _Glance:
    yaw: float
    pitch: float
    until: float
    mode: str


class GlancePlanner:
    """Decides, ~10 times a second, whether she is glancing away right now and where (call offset())."""

    def __init__(self, cfg: GlanceConfig, rng: Optional[random.Random] = None, config_path: str = CONFIG_PATH):
        self.cfg = cfg
        self._rng = rng or random.Random()
        self._path = config_path
        self._mtime = self._file_mtime()
        self._next_check = 0.0
        self._mode = "away"
        self._glance: Optional[_Glance] = None
        self._next_at: Optional[float] = None

    # ── configuration ──────────────────────────────────────────────────────

    def _file_mtime(self) -> float:
        try:
            return os.path.getmtime(self._path)
        except OSError:
            return 0.0

    def maybe_reload(self, now: float) -> None:
        if now < self._next_check:
            return
        self._next_check = now + 2.0
        m = self._file_mtime()
        if m and m != self._mtime:
            self._mtime = m
            self.cfg = load_glance_config(self._path)
            self._next_at = None
            logger.info("[glances] settings reloaded from kiosk_config.ini")

    # ── planning ───────────────────────────────────────────────────────────

    def _every(self, mode: str) -> tuple:
        return {"listening": self.cfg.listening_every_s, "quiet": self.cfg.quiet_every_s,
                "speaking": self.cfg.speaking_every_s}.get(mode, (0.0, 0.0))

    def _hold(self, mode: str) -> tuple:
        return {"listening": self.cfg.listening_hold_s, "quiet": self.cfg.quiet_hold_s,
                "speaking": self.cfg.speaking_hold_s, "thinking": self.cfg.thinking_hold_s}.get(mode, (0.5, 1.0))

    def _schedule(self, now: float, mode: str) -> None:
        lo, hi = self._every(mode)
        self._next_at = now + self._rng.uniform(lo, hi) if hi > 0 else None

    def _direction(self, mode: str) -> tuple:
        lo, hi = self.cfg.size_deg
        size = self._rng.uniform(lo, hi)
        side = self._rng.choice((-1.0, 1.0))
        if mode == "thinking":                       # people look up and aside while they think
            return side * size * 0.8, self._rng.uniform(4.0, 8.0)
        if self._rng.random() < 0.6:                 # aside, a little down
            return side * size, -self._rng.uniform(2.0, 6.0)
        return side * self._rng.uniform(0.0, 4.0), -size * 0.8      # down

    def _start(self, now: float, mode: str) -> None:
        yaw, pitch = self._direction(mode)
        lo, hi = self._hold(mode)
        hold = self._rng.uniform(lo, hi)
        self._glance = _Glance(yaw, pitch, now + hold, mode)
        side = "her left" if yaw > 0 else "her right"
        logger.info(f"[GAZE] glance away {math.hypot(yaw, pitch):.0f} deg ({side}, {'up' if pitch > 0 else 'down'}) "
                    f"for {hold:.1f} s while {mode}")

    def offset(self, now: float, mode: str) -> tuple:
        """Where she glances right now, (yaw, pitch) degrees; (0, 0) = at the visitor."""
        if mode not in MODES:
            mode = "quiet"
        if not self.cfg.enabled or mode == "away":
            self._glance, self._next_at, self._mode = None, None, mode
            return 0.0, 0.0
        if mode != self._mode:
            # A new phase of the conversation. Speaking cuts a glance short (steady contact while she talks, unless
            # set otherwise); thinking may start one; the others start their own rhythm.
            if self._glance and (mode == "speaking" or self._glance.mode == "thinking"):
                self._glance = None
            self._mode = mode
            if mode == "thinking":
                self._next_at = None
                if self._glance is None and self._rng.random() < self.cfg.thinking_chance:
                    self._start(now, mode)
            else:
                self._schedule(now, mode)
        if self._glance is not None:
            if now < self._glance.until:
                return self._glance.yaw, self._glance.pitch
            self._glance = None
            self._schedule(now, mode)
            return 0.0, 0.0
        if self._next_at is not None and now >= self._next_at:
            self._start(now, mode)
            return self._glance.yaw, self._glance.pitch
        return 0.0, 0.0
