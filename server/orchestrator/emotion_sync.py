"""
Emotion Synchronizer

Aligns LLM emotion directives with TTS word timestamps and A2E baseline
to produce timed emotion updates for UE5.

Flow:
  1. Directives extracted during LLM streaming (e.g. <<emotion:happy intensity:0.8>>)
  2. Each sentence goes to TTS → returns audio + word_timestamps
  3. A2E analyzes the TTS audio → baseline NPC emotion
  4. Synchronizer maps directive positions to word timestamp positions
  5. During playback, emotion updates are sent to UE5 at the right moments
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("emotion_sync")


@dataclass
class DirectiveEvent:
    """A directive extracted from the LLM response."""
    type: str           # "emotion" or "gesture"
    value: str          # e.g. "happy", "nod"
    intensity: float    # 0.0-1.0 (for emotions)
    sentence_index: int # which sentence this appeared in
    position: int       # character position within the sentence


@dataclass
class TimedEmotion:
    """An emotion update to send to UE5 at a specific time offset."""
    time_ms: float      # milliseconds from start of sentence audio
    emotion: str        # emotion name
    intensity: float    # 0.0-1.0


@dataclass
class SentenceSync:
    """Synchronization data for one sentence."""
    text: str
    directives: list[DirectiveEvent] = field(default_factory=list)
    word_timestamps: list[dict] = field(default_factory=list)  # from TTS
    a2e_emotion: str = "neutral"
    a2e_confidence: float = 0.0
    duration_ms: float = 0.0
    timed_emotions: list[TimedEmotion] = field(default_factory=list)


class EmotionSynchronizer:
    """Builds timed emotion events from directives + TTS timestamps + A2E."""

    def __init__(self):
        # Accumulate directives as they're extracted from the token stream
        self.pending_directives: list[DirectiveEvent] = []
        self.sentence_count: int = 0

    def add_directive(self, directive_type: str, value: str,
                      intensity: float = 0.5, char_position: int = 0):
        """Called when a directive is extracted from the LLM token stream."""
        self.pending_directives.append(DirectiveEvent(
            type=directive_type,
            value=value,
            intensity=intensity,
            sentence_index=self.sentence_count,
            position=char_position,
        ))

    def start_sentence(self):
        """Called when a new sentence begins."""
        self.sentence_count += 1

    def build_sentence_sync(
        self,
        sentence_text: str,
        word_timestamps: list[dict],
        a2e_emotion: str = "neutral",
        a2e_confidence: float = 0.0,
        duration_ms: float = 0.0,
    ) -> SentenceSync:
        """Build timed emotion events for a sentence.

        Args:
            sentence_text: The clean sentence text sent to TTS
            word_timestamps: List of {"word": str, "start_ms": float, "end_ms": float}
            a2e_emotion: Baseline emotion from A2E analysis of the TTS audio
            a2e_confidence: A2E confidence score
            duration_ms: Total audio duration in ms
        """
        sync = SentenceSync(
            text=sentence_text,
            word_timestamps=word_timestamps,
            a2e_emotion=a2e_emotion,
            a2e_confidence=a2e_confidence,
            duration_ms=duration_ms,
        )

        # Collect directives for this sentence
        current_idx = self.sentence_count
        sync.directives = [
            d for d in self.pending_directives
            if d.sentence_index == current_idx and d.type == "emotion"
        ]

        # Build timed emotions
        sync.timed_emotions = self._map_directives_to_timestamps(
            sync.directives, word_timestamps, a2e_emotion, a2e_confidence, duration_ms
        )

        return sync

    def _map_directives_to_timestamps(
        self,
        directives: list[DirectiveEvent],
        word_timestamps: list[dict],
        a2e_emotion: str,
        a2e_confidence: float,
        duration_ms: float,
    ) -> list[TimedEmotion]:
        """Map directive positions to audio timestamps.

        Strategy:
          - If no word timestamps, spread directives evenly across duration
          - If word timestamps available, map directive character positions
            to the nearest word's start time
          - A2E baseline applies from the start; directives override at their timestamps
        """
        timed = []

        # Start with A2E baseline at time 0
        if a2e_emotion != "neutral" and a2e_confidence > 0.3:
            timed.append(TimedEmotion(
                time_ms=0,
                emotion=a2e_emotion,
                intensity=min(a2e_confidence, 0.6),  # cap baseline intensity
            ))

        if not directives:
            return timed

        if not word_timestamps or duration_ms <= 0:
            # No timestamps — spread directives evenly
            interval = duration_ms / max(len(directives), 1) if duration_ms > 0 else 500
            for i, d in enumerate(directives):
                timed.append(TimedEmotion(
                    time_ms=i * interval,
                    emotion=d.value,
                    intensity=d.intensity,
                ))
        else:
            # Map directives to word timestamps by position
            total_chars = sum(len(w.get("word", "")) + 1 for w in word_timestamps)

            for d in directives:
                # Estimate which word the directive was near based on character position
                # Directives appear before the text they modify, so map to the nearest
                # word start after the directive position
                frac = d.position / max(total_chars, 1)
                target_ms = frac * duration_ms

                # Snap to nearest word boundary
                best_ms = target_ms
                for w in word_timestamps:
                    if w.get("start_ms", 0) >= target_ms * 0.8:
                        best_ms = w["start_ms"]
                        break

                timed.append(TimedEmotion(
                    time_ms=best_ms,
                    emotion=d.value,
                    intensity=d.intensity,
                ))

        # Sort by time
        timed.sort(key=lambda e: e.time_ms)
        return timed

    def clear_sentence_directives(self):
        """Remove directives for completed sentences."""
        self.pending_directives = [
            d for d in self.pending_directives
            if d.sentence_index >= self.sentence_count
        ]

    def reset(self):
        """Reset for a new conversation turn."""
        self.pending_directives.clear()
        self.sentence_count = 0


async def schedule_emotion_updates(
    timed_emotions: list[TimedEmotion],
    send_fn,
    start_time: float,
):
    """Send emotion updates to UE5 at the scheduled timestamps.

    Args:
        timed_emotions: List of TimedEmotion events
        send_fn: Async function to send WebSocket message to UE5
        start_time: time.time() when audio playback started
    """
    for te in timed_emotions:
        # Wait until the right moment
        target_time = start_time + (te.time_ms / 1000.0)
        now = time.time()
        if target_time > now:
            await asyncio.sleep(target_time - now)

        # Send emotion update
        msg = {
            "type": "emotion",
            "value": te.emotion,
            "intensity": te.intensity,
        }
        await send_fn(msg)
        logger.debug(f"Emotion @ {te.time_ms:.0f}ms: {te.emotion} ({te.intensity:.1f})")
