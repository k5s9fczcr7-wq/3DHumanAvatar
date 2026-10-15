"""
Identity mode - multi-user kiosk vs single-user companion.

kiosk_config.ini [identity] mode picks how the avatar decides who she is talking to:

  multi_user   (default) face recognition + name/PIN sign-in, a separate
               memory per person, anonymous chat allowed. FaceRecognizer and
               IdentificationManager handle it.
  single_user  one person and no identification at all; see single_user.py.
               None of its code runs in multi_user mode.
"""

import configparser
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .gaze_calculator import CONFIG_PATH

logger = logging.getLogger("identity_mode")

IDENTITY_MODES = ("multi_user", "single_user")

# Orchestrator startup log lines (its launcher truncates the log at every
# start, so the head of the log describes the running process).
SINGLE_USER_LOG_MARK = "[SINGLE] single_user mode"   # orchestrator._start_single_user
READY_LOG_MARK = "Orchestrator initialized"          # both modes, once startup is done


@dataclass
class IdentityConfig:
    mode: str = "multi_user"

    @property
    def single_user(self) -> bool:
        return self.mode == "single_user"


def load_identity_config(path: str = CONFIG_PATH) -> IdentityConfig:
    """Read [identity] from kiosk_config.ini. A missing section means
    multi_user; an unknown mode is logged as an error and also means
    multi_user, which works for everyone - sign-in is only offered."""
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8-sig")
    mode = cfg.get("identity", "mode", fallback="multi_user").strip().lower()
    if mode not in IDENTITY_MODES:
        logger.error(f"[IDENTITY] Unknown mode '{mode}' in {path} - running multi_user")
        return IdentityConfig()
    return IdentityConfig(mode=mode)


def write_identity_mode(mode: str, path: str = CONFIG_PATH):
    """Set [identity] mode in kiosk_config.ini (used by the /admin switch),
    leaving every other line - comments, line endings, a BOM - as it was.
    Adds the section or the line if missing. Written atomically."""
    if mode not in IDENTITY_MODES:
        raise ValueError(f"unknown identity mode '{mode}'")
    with open(path, "rb") as f:
        raw = f.read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    lines = raw.decode("utf-8-sig").splitlines(keepends=True)
    newline = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"

    section = None
    header_at = None
    done = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if section == "identity" and not done:
                break  # section ended without a mode line
            section = stripped[1:-1].strip().lower()
            if section == "identity":
                header_at = i
            continue
        if section == "identity" and re.match(r"\s*mode\s*[=:]", line):
            lines[i] = f"mode = {mode}{newline}"
            done = True
            break
    if not done:
        if header_at is not None:
            lines.insert(header_at + 1, f"mode = {mode}{newline}")
        else:
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += newline
            lines += [newline, f"[identity]{newline}", f"mode = {mode}{newline}"]

    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write((b"\xef\xbb\xbf" if bom else b"") + "".join(lines).encode("utf-8"))
    os.replace(tmp, path)


def running_mode_from_log(lines: Iterable[str]) -> Optional[str]:
    """The mode the running orchestrator started in, from the head of its
    log. None until its startup has finished."""
    head = list(lines)
    if not any(READY_LOG_MARK in line for line in head):
        return None
    return "single_user" if any(SINGLE_USER_LOG_MARK in line for line in head) else "multi_user"
