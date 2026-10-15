"""
Facial Emotion Detector

Runs HSEmotion enet_b0 (ONNX, 16MB) on face crops from FaceTracker.
Returns one of 8 emotion labels per frame (~20ms inference, 2 ONNX threads).

Also provides emotion → mood mapping and face+voice emotion fusion.
The mood name is what gets prepended to user messages as [MOOD: X].
"""

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("face_emotion")

# ── Model Configuration ──────────────────────────────────────────────────────

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "enet_b0_8_best_afew.onnx"
)

INPUT_SIZE = 224  # enet_b0 expects 224x224
ONNX_THREADS = 2  # pin to 2 threads to prevent greedy spawning

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 8-class AffectNet label order (index → label)
EMOTION_LABELS = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]

# ── Mood Mapping ─────────────────────────────────────────────────────────────
#
# Mood modes:
#   WARM     — returning user, friendly context
#   UPBEAT   — user is happy/excited/surprised
#   CALM     — user is neutral/relaxed
#   GENTLE   — user is sad/fearful/vulnerable
#   DIRECT   — user is angry/frustrated/contemptuous
#
# The LLM system prompt has pre-loaded behaviors for each mood.
# Raw emotion labels never reach the LLM — only the mood name.

EMOTION_TO_MOOD = {
    "anger":     "DIRECT",
    "contempt":  "DIRECT",
    "disgust":   "DIRECT",
    "fear":      "GENTLE",
    "happiness": "UPBEAT",
    "neutral":   "CALM",
    "sadness":   "GENTLE",
    "surprise":  "UPBEAT",
}

# Voice emotion labels (from emotion2vec in orchestrator) → mood
VOICE_EMOTION_TO_MOOD = {
    "angry":     "DIRECT",
    "disgust":   "DIRECT",
    "fearful":   "GENTLE",
    "happy":     "UPBEAT",
    "neutral":   "CALM",
    "sad":       "GENTLE",
    "surprised": "UPBEAT",
    "other":     "CALM",
    "unknown":   "CALM",
}

# Temperature per mood (passed to LLM)
MOOD_TEMPERATURE = {
    "WARM":   0.7,
    "UPBEAT": 0.8,
    "CALM":   0.6,
    "GENTLE": 0.5,
    "DIRECT": 0.5,
}

DEFAULT_MOOD = "CALM"


class FaceEmotionDetector:
    """Runs HSEmotion enet_b0 on face crops. Thread-safe for single-threaded use."""

    def __init__(self, model_path: str = MODEL_PATH):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ONNX_THREADS
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3  # suppress ONNX warnings

        self._session = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info(f"HSEmotion loaded ({model_path}, {ONNX_THREADS} threads)")

    def detect(self, face_crop_bgr: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Run emotion detection on a BGR face crop.

        Returns (emotion_label, confidence, all_probabilities).
        """
        x = self._preprocess(face_crop_bgr)
        logits = self._session.run(None, {self._input_name: x})[0][0]
        probs = _softmax(logits)

        best_idx = int(probs.argmax())
        label = EMOTION_LABELS[best_idx]
        confidence = float(probs[best_idx])

        return label, confidence, probs

    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        """BGR face crop → NCHW float32 tensor (1, 3, 224, 224)."""
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        normalized = (resized - MEAN) / STD
        return normalized.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── Mood Fusion ──────────────────────────────────────────────────────────────

class MoodTracker:
    """Fuses face emotion (continuous) + voice emotion (per-utterance) into a mood.

    Face emotion updates every ~100ms from FaceTracker crops.
    Voice emotion updates once per user utterance from emotion2vec.
    Mood is derived from the fused signal and smoothed to avoid flicker.
    """

    def __init__(self):
        self._face_mood: str = DEFAULT_MOOD
        self._face_confidence: float = 0.0
        self._voice_mood: str = DEFAULT_MOOD
        self._voice_confidence: float = 0.0
        self._current_mood: str = DEFAULT_MOOD
        self._mood_hold_until: float = 0.0  # prevent rapid flicker

    MOOD_HOLD_S = 3.0  # hold a mood for at least this long

    @property
    def mood(self) -> str:
        return self._current_mood

    @property
    def temperature(self) -> float:
        return MOOD_TEMPERATURE.get(self._current_mood, 0.7)

    def update_face_emotion(self, emotion: str, confidence: float):
        """Called when face emotion is detected (~10Hz)."""
        mood = EMOTION_TO_MOOD.get(emotion, DEFAULT_MOOD)
        self._face_mood = mood
        self._face_confidence = confidence
        self._update_mood()

    def update_voice_emotion(self, emotion: str, confidence: float):
        """Called when voice emotion is detected (once per utterance)."""
        mood = VOICE_EMOTION_TO_MOOD.get(emotion, DEFAULT_MOOD)
        self._voice_mood = mood
        self._voice_confidence = confidence
        self._update_mood()

    def _update_mood(self):
        """Fuse face + voice mood into a single mood with hold timer."""
        now = time.monotonic()

        # Don't switch mood during hold period (prevents flicker)
        if now < self._mood_hold_until:
            return

        # Voice emotion is episodic and intentional — weight it higher when fresh
        # Face emotion is continuous — use it as baseline
        if self._voice_confidence > 0.5 and self._voice_mood != DEFAULT_MOOD:
            new_mood = self._voice_mood
        elif self._face_confidence > 0.4 and self._face_mood != DEFAULT_MOOD:
            new_mood = self._face_mood
        else:
            new_mood = DEFAULT_MOOD

        if new_mood != self._current_mood:
            logger.info(f"Mood: {self._current_mood} → {new_mood} "
                        f"(face={self._face_mood}/{self._face_confidence:.2f}, "
                        f"voice={self._voice_mood}/{self._voice_confidence:.2f})")
            self._current_mood = new_mood
            self._mood_hold_until = now + self.MOOD_HOLD_S

    def reset(self):
        """Reset to default mood."""
        self._face_mood = DEFAULT_MOOD
        self._face_confidence = 0.0
        self._voice_mood = DEFAULT_MOOD
        self._voice_confidence = 0.0
        self._current_mood = DEFAULT_MOOD
        self._mood_hold_until = 0.0
