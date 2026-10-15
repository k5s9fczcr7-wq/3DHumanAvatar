"""
Procedural Viseme Mapper (fallback for Audio2Face, A2F)

Maps TTS word timestamps to blendshape keyframes via phoneme→viseme lookup.
Produces 264-byte UDP packets at 30fps for FA2FLiveLinkSource.

Flow:
  1. TTS returns word_timestamps: [{"word": "Hello", "start_ms": 0, "end_ms": 240}, ...]
  2. Each word is mapped to a sequence of viseme poses (blendshape targets)
  3. Frames are interpolated at 30fps between viseme targets
  4. Output: list of 264-byte packets ready for UDP pacing
"""

import struct
import math
import numpy as np
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────

PACKET_MAGIC = 0x41324642  # "A2FB"
PACKET_FMT = "<II I 52f 4f 3f 2f 2f"
PACKET_SIZE = 264
FPS = 30

# ARKit blendshape indices (matching FA2FLiveLinkSource order)
IDX_JAW_OPEN = 20
IDX_MOUTH_CLOSE = 22
IDX_MOUTH_FUNNEL = 27
IDX_MOUTH_PUCKER = 33
IDX_MOUTH_LOWER_DOWN_L = 29
IDX_MOUTH_LOWER_DOWN_R = 30
IDX_MOUTH_UPPER_UP_L = 43
IDX_MOUTH_UPPER_UP_R = 44
IDX_MOUTH_PRESS_L = 31
IDX_MOUTH_PRESS_R = 32
IDX_MOUTH_STRETCH_L = 41
IDX_MOUTH_STRETCH_R = 42
IDX_MOUTH_SMILE_L = 39
IDX_MOUTH_SMILE_R = 40
IDX_MOUTH_FROWN_L = 25
IDX_MOUTH_FROWN_R = 26
IDX_MOUTH_ROLL_LOWER = 35
IDX_MOUTH_ROLL_UPPER = 36
IDX_TONGUE_OUT = 47

# ── Viseme Definitions ─────────────────────────────────────────────────────────
#
# Standard viseme set based on Preston Blair phoneme groups.
# Each viseme is a dict of {blendshape_index: target_value}.
# Values are conservative (0.1-0.4 range) to avoid exaggeration.

VISEMES = {
    # Silence / rest
    "SIL": {},

    # AA — "father", "bot" (open jaw)
    "AA": {
        IDX_JAW_OPEN: 0.35,
        IDX_MOUTH_LOWER_DOWN_L: 0.15,
        IDX_MOUTH_LOWER_DOWN_R: 0.15,
    },

    # AE — "cat", "bat" (wide open)
    "AE": {
        IDX_JAW_OPEN: 0.30,
        IDX_MOUTH_STRETCH_L: 0.10,
        IDX_MOUTH_STRETCH_R: 0.10,
        IDX_MOUTH_LOWER_DOWN_L: 0.10,
        IDX_MOUTH_LOWER_DOWN_R: 0.10,
    },

    # AH — "but", "cup" (relaxed open)
    "AH": {
        IDX_JAW_OPEN: 0.25,
        IDX_MOUTH_LOWER_DOWN_L: 0.10,
        IDX_MOUTH_LOWER_DOWN_R: 0.10,
    },

    # EE — "see", "me" (wide smile)
    "EE": {
        IDX_JAW_OPEN: 0.10,
        IDX_MOUTH_SMILE_L: 0.15,
        IDX_MOUTH_SMILE_R: 0.15,
        IDX_MOUTH_STRETCH_L: 0.15,
        IDX_MOUTH_STRETCH_R: 0.15,
    },

    # EH — "bed", "get" (slightly open, wide)
    "EH": {
        IDX_JAW_OPEN: 0.20,
        IDX_MOUTH_STRETCH_L: 0.10,
        IDX_MOUTH_STRETCH_R: 0.10,
    },

    # IH — "sit", "bit" (between EE and EH)
    "IH": {
        IDX_JAW_OPEN: 0.15,
        IDX_MOUTH_SMILE_L: 0.10,
        IDX_MOUTH_SMILE_R: 0.10,
    },

    # OH — "go", "boat" (rounded)
    "OH": {
        IDX_JAW_OPEN: 0.25,
        IDX_MOUTH_FUNNEL: 0.15,
        IDX_MOUTH_PUCKER: 0.10,
    },

    # OO — "boot", "you" (tight round)
    "OO": {
        IDX_JAW_OPEN: 0.10,
        IDX_MOUTH_PUCKER: 0.25,
        IDX_MOUTH_FUNNEL: 0.20,
    },

    # PP — "pop", "baby", "map" (lips pressed, bilabial)
    "PP": {
        IDX_MOUTH_PRESS_L: 0.25,
        IDX_MOUTH_PRESS_R: 0.25,
        IDX_MOUTH_UPPER_UP_L: 0.10,
        IDX_MOUTH_UPPER_UP_R: 0.10,
        IDX_MOUTH_ROLL_LOWER: 0.10,
        IDX_MOUTH_ROLL_UPPER: 0.10,
    },

    # FF — "fan", "vat" (lower lip to upper teeth)
    "FF": {
        IDX_MOUTH_UPPER_UP_L: 0.15,
        IDX_MOUTH_UPPER_UP_R: 0.15,
        IDX_MOUTH_ROLL_LOWER: 0.15,
        IDX_JAW_OPEN: 0.05,
    },

    # TH — "think", "the" (tongue tip)
    "TH": {
        IDX_JAW_OPEN: 0.10,
        IDX_TONGUE_OUT: 0.15,
        IDX_MOUTH_LOWER_DOWN_L: 0.05,
        IDX_MOUTH_LOWER_DOWN_R: 0.05,
    },

    # DD — "day", "ten", "nine" (tongue tap, alveolar)
    "DD": {
        IDX_JAW_OPEN: 0.15,
        IDX_MOUTH_LOWER_DOWN_L: 0.05,
        IDX_MOUTH_LOWER_DOWN_R: 0.05,
    },

    # KK — "cat", "go", "king" (back tongue)
    "KK": {
        IDX_JAW_OPEN: 0.20,
        IDX_MOUTH_LOWER_DOWN_L: 0.08,
        IDX_MOUTH_LOWER_DOWN_R: 0.08,
    },

    # SS — "sit", "zoo" (teeth close, hissing)
    "SS": {
        IDX_JAW_OPEN: 0.05,
        IDX_MOUTH_SMILE_L: 0.08,
        IDX_MOUTH_SMILE_R: 0.08,
        IDX_MOUTH_STRETCH_L: 0.08,
        IDX_MOUTH_STRETCH_R: 0.08,
    },

    # SH — "she", "measure" (rounded hissing)
    "SH": {
        IDX_JAW_OPEN: 0.08,
        IDX_MOUTH_PUCKER: 0.12,
        IDX_MOUTH_FUNNEL: 0.10,
    },

    # RR — "red", "run" (slight pucker)
    "RR": {
        IDX_JAW_OPEN: 0.12,
        IDX_MOUTH_PUCKER: 0.08,
        IDX_MOUTH_FUNNEL: 0.05,
    },

    # LL — "let", "all" (tongue up, jaw open)
    "LL": {
        IDX_JAW_OPEN: 0.15,
        IDX_MOUTH_LOWER_DOWN_L: 0.05,
        IDX_MOUTH_LOWER_DOWN_R: 0.05,
    },

    # NN — "no", "man" (closed, nasal)
    "NN": {
        IDX_JAW_OPEN: 0.05,
        IDX_MOUTH_CLOSE: 0.10,
    },

    # WW — "we", "way" (tight round like OO)
    "WW": {
        IDX_MOUTH_PUCKER: 0.20,
        IDX_MOUTH_FUNNEL: 0.15,
        IDX_JAW_OPEN: 0.08,
    },
}

# ── Word to Viseme Mapping ─────────────────────────────────────────────────────
#
# Simple English grapheme→viseme rules. Not perfect but covers common patterns.
# Each word gets a sequence of viseme IDs with relative timing.

def _char_to_viseme(char: str, next_char: str = "") -> str:
    """Map a character (or digraph) to a viseme ID."""
    c = char.lower()
    n = next_char.lower()

    # Digraphs
    if c == 's' and n == 'h':
        return "SH"
    if c == 't' and n == 'h':
        return "TH"
    if c == 'c' and n == 'h':
        return "SH"

    # Vowels
    if c in ('a',):
        if n in ('i', 'y'):
            return "AE"
        if n in ('w', 'u'):
            return "AA"
        return "AH"
    if c in ('e',):
        if n in ('e', 'a'):
            return "EE"
        return "EH"
    if c in ('i',):
        if n in ('e', 'g'):
            return "EE"
        return "IH"
    if c in ('o',):
        if n in ('o', 'u', 'w'):
            return "OO"
        return "OH"
    if c in ('u',):
        if n in ('e',):
            return "OO"
        return "AH"
    if c == 'y':
        return "EE"

    # Consonants
    if c in ('p', 'b', 'm'):
        return "PP"
    if c in ('f', 'v'):
        return "FF"
    if c in ('t', 'd', 'n'):
        return "DD"
    if c in ('k', 'g', 'q'):
        return "KK"
    if c in ('s', 'z', 'c'):
        return "SS"
    if c in ('r',):
        return "RR"
    if c in ('l',):
        return "LL"
    if c in ('w',):
        return "WW"
    if c in ('j',):
        return "SH"
    if c in ('h',):
        return "AH"  # breathy, open

    return "SIL"


def word_to_visemes(word: str) -> list[str]:
    """Convert a word to a list of viseme IDs."""
    visemes = []
    chars = list(word.lower())
    i = 0
    while i < len(chars):
        if not chars[i].isalpha():
            i += 1
            continue
        next_c = chars[i + 1] if i + 1 < len(chars) else ""
        v = _char_to_viseme(chars[i], next_c)

        # Skip digraph second character
        if chars[i] in ('s', 't', 'c') and next_c in ('h',) and v in ("SH", "TH"):
            i += 1

        # Don't repeat the same viseme consecutively
        if not visemes or visemes[-1] != v:
            visemes.append(v)
        i += 1

    return visemes if visemes else ["SIL"]


# ── Frame Generation ───────────────────────────────────────────────────────────

def _lerp_blendshapes(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two blendshape arrays."""
    return a * (1.0 - t) + b * t


def _viseme_to_array(viseme_id: str) -> np.ndarray:
    """Convert a viseme ID to a 52-float blendshape array."""
    arr = np.zeros(52, dtype=np.float32)
    targets = VISEMES.get(viseme_id, {})
    for idx, val in targets.items():
        arr[idx] = val
    return arr


def generate_frames_from_timestamps(
    word_timestamps: list[dict],
    total_duration_ms: float,
    fps: int = 30,
) -> bytes:
    """Generate 264-byte frame packets from word timestamps.

    Args:
        word_timestamps: [{"word": "Hello", "start_ms": 0, "end_ms": 240}, ...]
        total_duration_ms: Total audio duration in ms
        fps: Target frame rate

    Returns:
        Concatenated 264-byte packets (ready for base64 or direct UDP send)
    """
    if not word_timestamps or total_duration_ms <= 0:
        return b""

    total_frames = max(1, int(total_duration_ms / 1000.0 * fps))
    frame_duration_ms = 1000.0 / fps

    # Build a timeline of (time_ms, viseme_array) keyframes
    keyframes: list[tuple[float, np.ndarray]] = []
    keyframes.append((0.0, _viseme_to_array("SIL")))  # start silent

    for wt in word_timestamps:
        word = wt.get("word", "")
        start = wt.get("start_ms", 0)
        end = wt.get("end_ms", start + 100)
        duration = end - start

        viseme_ids = word_to_visemes(word)
        if not viseme_ids:
            continue

        # Spread visemes across the word's duration
        vis_duration = duration / len(viseme_ids)
        for j, vid in enumerate(viseme_ids):
            t = start + j * vis_duration
            keyframes.append((t, _viseme_to_array(vid)))

        # Return to slight rest after word
        keyframes.append((end, _viseme_to_array("SIL") * 0.3))

    # End silent
    keyframes.append((total_duration_ms, _viseme_to_array("SIL")))

    # Sort by time
    keyframes.sort(key=lambda x: x[0])

    # Interpolate at fps rate
    packets = bytearray()
    for i in range(total_frames):
        t_ms = i * frame_duration_ms

        # Find surrounding keyframes
        prev_kf = keyframes[0]
        next_kf = keyframes[-1]
        for k in range(len(keyframes) - 1):
            if keyframes[k][0] <= t_ms <= keyframes[k + 1][0]:
                prev_kf = keyframes[k]
                next_kf = keyframes[k + 1]
                break

        # Interpolate
        kf_duration = next_kf[0] - prev_kf[0]
        if kf_duration > 0:
            alpha = (t_ms - prev_kf[0]) / kf_duration
        else:
            alpha = 0.0
        alpha = max(0.0, min(1.0, alpha))

        blendshapes = _lerp_blendshapes(prev_kf[1], next_kf[1], alpha)

        # Pack into 264-byte packet
        ts_ms = int(t_ms)
        head_quat = (0.0, 0.0, 0.0, 1.0)
        head_pos = (0.0, 0.0, 0.0)
        left_eye = (0.0, 0.0)
        right_eye = (0.0, 0.0)

        packet = struct.pack(
            PACKET_FMT,
            PACKET_MAGIC, i + 1, ts_ms,
            *blendshapes.tolist(),
            *head_quat, *head_pos,
            *left_eye, *right_eye,
        )
        packets.extend(packet)

    return bytes(packets)
