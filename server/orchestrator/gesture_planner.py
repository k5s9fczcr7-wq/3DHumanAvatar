"""
Gestures that fit what she says (rule-based).

Unreal holds the gesture snippets (/Game/Gestures/DA_GestureLibrary) and picks one per request
by intent, emotion fit (energy x tone) and role. This module decides WHEN to ask and WHICH INTENT:

  speaking   at most one gesture per sentence, never closer than talk_gap_s apart (wider in GENTLE and
             DIRECT moods, a little tighter in UPBEAT). The intent comes from her LLM gesture tag when the
             sentence has one (nod -> agree, ...), else from word cues ("yes" -> agree, "but" -> contrast,
             "sorry" -> empathize, ...), else a soft "explain" on longer sentences only. Her voice tags
             ([laugh], [chuckle], [sigh]) are not words: cues ignore them and a tag alone gets no gesture.
  listening  a small nod while the visitor talks: the first after listen_first_s, then every 5-8 s.
  idle       nobody at the kiosk: a look-around now and then (35-70 s apart).

Message to Unreal: {"type": "gesture", "intent": ..., "emotion": ..., "intensity": ..., "role": ...}.
Unreal maps her emotion intensity (0.1-0.8) to ~45-85% strength. Plans are made when a chunk of her
speech starts playing and sent on a timer, so replies are not delayed. [gestures] in kiosk_config.ini
can switch it off (enabled = false) or change the spacing; every key is optional.
"""

import configparser
import logging
import random
import re
from dataclasses import dataclass

from .gaze_calculator import CONFIG_PATH

logger = logging.getLogger("gesture_planner")

# her LLM gesture tags -> intents (persona_prompt.py: nod, wave, tilt_head, thinking_pose, shrug)
TAG_INTENT = {"nod": "agree", "wave": "welcome", "tilt_head": "empathize", "thinking_pose": "unsure", "shrug": "unsure"}
INTENTS = ("agree", "disagree", "emphasize", "explain", "list", "contrast", "unsure", "empathize", "welcome",
           "surprise", "listen", "talk", "idle")

# Word cues, first match wins. Checked on the lower-cased sentence.
CUES = [
    ("welcome", r"^(hi|hello|hey|welcome|good (morning|afternoon|evening))\b|\bnice to (meet|see) you\b"),
    ("empathize", r"\b(sorry|i understand|that must (be|have been)|i hear you|i'm here for you|take care"
                  r"|that sounds (hard|tough|difficult|stressful))\b"),
    ("disagree", r"^(no|nope|not really|not quite)\b|\b(i don't think so|unfortunately|i'm afraid not|that's not)\b"),
    ("unsure", r"\b(i'm not sure|not sure|maybe|perhaps|possibly|it depends|hmm|let me think|i wonder|hard to say)\b"),
    ("agree", r"^(yes|yeah|yep|absolutely|exactly|of course|sure|definitely|right|correct|indeed)\b"
              r"|\b(that's right|you're right|good point)\b"),
    ("surprise", r"^(oh|wow|whoa)\b|\b(no way|amazing|incredible)\b"),
    ("contrast", r"\b(but|however|on the other hand|although|though|whereas|instead|otherwise)\b"),
    ("list", r"\b(first|firstly|second|secondly|third|finally)\b|\b(a few|several|two|three|four) "
             r"(things|ways|options|steps|reasons|kinds)\b"),
    ("explain", r"\b(because|which means|that means|for example|for instance|in other words|the reason"
                r"|here's|the way|basically)\b"),
    ("emphasize", r"\b(really|very|truly|important|key|essential|always|never|every|only)\b|!$"),
]
_CUES = [(intent, re.compile(rx)) for intent, rx in CUES]
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_VOICE_TAG = re.compile(r"\[[a-z][a-z _-]*\]")   # [laugh], [chuckle], [sigh]: passed through to the TTS


@dataclass
class GestureConfig:
    enabled: bool = True
    talk_gap_s: float = 2.5          # minimum time between two gestures while she talks
    min_sentence_s: float = 1.2      # shorter sentences get a gesture only from her own tag
    explain_min_s: float = 2.5       # a sentence with no cue gets a soft "explain" only if at least this long
    listen_first_s: float = 2.5      # first listening nod this long into the visitor's speech
    listen_min_s: float = 5.0        # then one every listen_min_s..listen_max_s
    listen_max_s: float = 8.0
    idle_after_s: float = 20.0       # nobody at the kiosk this long before the first look-around
    idle_min_s: float = 35.0         # then one every idle_min_s..idle_max_s
    idle_max_s: float = 70.0
    lead_s: float = 0.15             # a gesture starts this long after its sentence starts


def load_gesture_config(path: str = CONFIG_PATH) -> GestureConfig:
    """Read the optional [gestures] section of kiosk_config.ini; missing keys keep their defaults."""
    cfg = GestureConfig()
    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8-sig")
    except configparser.Error as e:
        logger.error(f"[GESTURE] could not read {path}: {e} - using defaults")
        return cfg
    if parser.has_section("gestures"):
        sec = parser["gestures"]
        try:
            cfg.enabled = sec.getboolean("enabled", fallback=cfg.enabled)
            for key in ("talk_gap_s", "min_sentence_s", "explain_min_s", "listen_first_s", "listen_min_s",
                        "listen_max_s", "idle_after_s", "idle_min_s", "idle_max_s", "lead_s"):
                setattr(cfg, key, sec.getfloat(key, fallback=getattr(cfg, key)))
        except ValueError as e:
            logger.error(f"[GESTURE] bad value in [gestures]: {e} - using defaults")
            return GestureConfig()
    return cfg


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def tag_intent(tag: str) -> str | None:
    """Her LLM tag (a prompt gesture name or an intent) as an intent, or None if unknown."""
    tag = (tag or "").strip().lower()
    if tag in TAG_INTENT:
        return TAG_INTENT[tag]
    return tag if tag in INTENTS else None


def strip_voice_tags(text: str) -> str:
    return " ".join(_VOICE_TAG.sub(" ", text.lower()).split())


def cue_intent(sentence: str) -> str | None:
    s = strip_voice_tags(sentence)
    for intent, rx in _CUES:
        if rx.search(s):
            return intent
    return None


class GesturePacer:
    """Keeps the timing state; every method takes the current time so it can be tested without a clock."""

    def __init__(self, cfg: GestureConfig, rng: random.Random | None = None):
        self.cfg = cfg
        self.rng = rng or random.Random()
        self.last_talk_at = float("-inf")
        self._listen_for: float | None = None    # start time of the utterance the listen timer belongs to
        self._next_listen_at = 0.0
        self._idle_for: float | None = None
        self._next_idle_at = 0.0

    def plan_speech(self, text: str, duration_s: float, tags: list[str], mood: str | None,
                    started_at: float) -> list[tuple[float, str]]:
        """Gestures for one chunk of her speech: [(seconds after the chunk starts, intent), ...].
        tags = her LLM gesture tags that arrived with this chunk; the first belongs to its first sentence."""
        if not self.cfg.enabled or duration_s <= 0:
            return []
        sentences = split_sentences(text)
        total = sum(len(s) for s in sentences) or 1
        gap = self.cfg.talk_gap_s * {"GENTLE": 1.6, "DIRECT": 1.6, "UPBEAT": 0.85}.get((mood or "").upper(), 1.0)
        plan, pos = [], 0
        for i, sentence in enumerate(sentences):
            start = duration_s * pos / total
            length = duration_s * len(sentence) / total
            pos += len(sentence)            # a voice tag keeps its share of the time: the laugh is heard
            if not strip_voice_tags(sentence):
                continue
            tagged = tag_intent(tags[0]) if (i == 0 and tags) else None
            intent = tagged or cue_intent(sentence)
            if intent is None and length >= self.cfg.explain_min_s:
                intent = "explain"
            if intent is None or (tagged is None and length < self.cfg.min_sentence_s):
                continue
            at = started_at + start + self.cfg.lead_s
            if at - self.last_talk_at < gap:
                continue
            plan.append((round(start + self.cfg.lead_s, 2), intent))
            self.last_talk_at = at
        return plan

    def listen_due(self, now: float, speech_started_at: float) -> bool:
        """True when a listening nod is due during the visitor's current utterance."""
        if not self.cfg.enabled:
            return False
        if self._listen_for != speech_started_at:
            self._listen_for = speech_started_at
            self._next_listen_at = speech_started_at + self.cfg.listen_first_s
        if now >= self._next_listen_at:
            self._next_listen_at = now + self.rng.uniform(self.cfg.listen_min_s, self.cfg.listen_max_s)
            return True
        return False

    def idle_due(self, now: float, idle_since: float) -> bool:
        """True when a look-around is due while nobody is at the kiosk."""
        if not self.cfg.enabled or now - idle_since < self.cfg.idle_after_s:
            return False
        if self._idle_for != idle_since:
            self._idle_for = idle_since
            self._next_idle_at = idle_since + self.cfg.idle_after_s
        if now >= self._next_idle_at:
            self._next_idle_at = now + self.rng.uniform(self.cfg.idle_min_s, self.cfg.idle_max_s)
            return True
        return False
