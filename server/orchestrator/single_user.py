"""
Single-user mode - one person, no identification, one live conversation.

Used only when kiosk_config.ini [identity] mode = single_user; nothing here
runs in multi_user mode.

  Presence          stands in for FaceRecognizer: is someone in front of the
                    camera? Drives the room-audio gate, the lapel pin and the
                    welcome-back. Nobody is ever signed in or out.
  PendingExchanges  every exchange is appended to a JSONL file the moment it
                    completes, so a restart or crash loses nothing. The
                    orchestrator commits the file to memory after
                    COMMIT_IDLE_S of quiet and removes what it committed;
                    leftovers are restored into the conversation at startup.
  prompt builders   the memory block in her system prompt, the welcome-back
                    note, and the two memory-update requests. Those requests
                    run inside the live LLM session, appended after the
                    conversation, so its KV cache survives them.
"""

import json
import logging
import os
import time
from typing import Optional

from .face_recognizer import AUTH_SESSION_GONE_S, FaceRecognizer
from .persona import AVATAR_NAME
from .persona_prompt import strip_directives

logger = logging.getLogger("single_user")

# Memory-store id of the one user. Internal only - never spoken.
SINGLE_USER_ID = "single_user"
# LLM gateway sessions with this prefix never expire (see llm_gateway.py).
SESSION_PREFIX = "single-user-"

PRESENCE_GONE_S = AUTH_SESSION_GONE_S   # no face this long = they've stepped away
WELCOME_BACK_AFTER_S = 15 * 60          # away at least this long -> welcome them back
COMMIT_IDLE_S = 30.0                    # this much quiet -> commit pending exchanges
COMPACT_AT_TOKENS = 240_000             # of the LLM's 262,144 (a KB can take ~200K): past this, rebuild

# Next to faces.db: the backend's own data folder, found relative to this file.
PENDING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "single_user_pending.jsonl"
)

FIRST_GREETING = ("<<emotion:happy intensity:0.6>><<gesture:wave>> "
                  f"Hi! I'm {AVATAR_NAME}. It's nice to meet you. What's on your mind?")

MEMORY_UPDATE_TAG = ("[MEMORY UPDATE - not part of the conversation. Do not reply to "
                     "the person; follow these instructions exactly.]")


class Presence:
    """Single-user stand-in for FaceRecognizer: is someone at the kiosk?

    update() returns "arrived" when a centered face appears while nobody was
    there, "left" once no face has been seen for PRESENCE_GONE_S, otherwise
    None. The event names differ from FaceRecognizer's on purpose, so no
    multi-user handler can fire in this mode. expire_if_gone() exists for the
    orchestrator's shared code paths and never ends anything.
    """

    def __init__(self, face_tracker):
        self._tracker = face_tracker
        self._present = False
        self._last_seen = 0.0                  # monotonic time of the last poll with a face
        self._left_at: Optional[float] = None  # when they last stepped away; None = not since startup
        self.away_s: Optional[float] = None    # at the last arrival: time away (None = first since startup)

    @property
    def has_user(self) -> bool:
        return self._present

    @property
    def session(self):
        return None

    @property
    def user_id(self):
        return None

    @property
    def user_name(self):
        return None

    def expire_if_gone(self) -> bool:
        return False

    def update(self) -> Optional[str]:
        state = self._tracker.state
        now = time.monotonic()
        if self._present:
            if state.present:
                self._last_seen = now
                return None
            if now - self._last_seen < PRESENCE_GONE_S:
                return None
            self._present = False
            self._left_at = self._last_seen
            return "left"
        # As in FaceRecognizer, only a centered face counts as arriving, so
        # someone passing the edge of the frame doesn't get greeted.
        if state.present and FaceRecognizer._is_face_centered(state):
            self._present = True
            self._last_seen = now
            self.away_s = None if self._left_at is None else now - self._left_at
            return "arrived"
        return None


class PendingExchanges:
    """Append-only JSONL file of exchanges not yet committed to memory.

    One line per exchange: {"id": <ns timestamp>, "t": "<local time>",
    "user": ..., "assistant": ...}. Each append is flushed and fsynced, so a
    crash loses at most the exchange being written; a torn last line is
    skipped on read. Methods are synchronous, so within the orchestrator's
    event loop an append can never interleave with remove_through().
    """

    def __init__(self, path: str = PENDING_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, user: str, assistant: str) -> int:
        entry_id = time.time_ns()
        line = json.dumps({"id": entry_id, "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "user": user, "assistant": assistant}, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return entry_id

    def read(self) -> list:
        if not os.path.exists(self.path):
            return []
        entries = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("[SINGLE] Skipping an unreadable pending line (torn write?)")
                    continue
                if isinstance(entry, dict) and "id" in entry:
                    entries.append(entry)
        return entries

    def remove_through(self, last_id: int):
        """Drop committed entries (id <= last_id); keep anything newer."""
        keep = [e for e in self.read() if e["id"] > last_id]
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for e in keep:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


# ── Prompt builders ─────────────────────────────────────────────────────────

def profile_block(state: Optional[dict]) -> str:
    """Her system-prompt addition in single-user mode. Built at startup and
    at compaction only, so the system prompt stays byte-stable and the KV
    cache keeps matching the whole conversation."""
    # Every line must stay true for the whole session, because the block is
    # not rebuilt as the conversation grows (a line like "you haven't talked
    # before" turns false the moment they speak, and she would then deny
    # remembering).
    lines = [
        "[SINGLE-USER MODE]",
        "This kiosk belongs to one person, and whoever is talking to you is that same person.",
        "Never address them by name, even if a name appears in what you remember.",
        "You remember this whole conversation, including what was said before any break.",
    ]
    if state is None:
        lines.append("Your long-term memory could not be loaded just now.")
    else:
        facts = state.get("facts") or []
        summary = (state.get("summary") or "").strip()
        recent = (state.get("recent") or "").strip()
        if facts:
            lines += ["", "What you know about them:"] + [f"- {f}" for f in facts]
        if summary:
            lines += ["", "Summary of your conversations so far:", summary]
        if recent:
            lines += ["", "The last exchanges before this conversation:", recent]
        if not (facts or summary or recent):
            lines.append("Everything you know about them so far is in this conversation.")
    lines.append("[/SINGLE-USER MODE]")
    return "\n".join(lines)


def humanize_duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{max(minutes, 1)} minute{'s' if minutes != 1 else ''}"
    hours = round(seconds / 3600)
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = round(seconds / 86400)
    return f"{days} day{'s' if days != 1 else ''}"


def welcome_back_note(away_s: Optional[float]) -> str:
    """Stand-in 'user' turn that prompts her welcome-back inside the live
    conversation. Never saved to memory."""
    when = ("They've just sat down in front of you." if away_s is None
            else f"They're back after {humanize_duration(away_s)} away.")
    return (f"[{when} Welcome them in a sentence or two and pick up naturally "
            f"from where you left off. Don't use a name.]")


def format_exchanges(exchanges: list) -> str:
    return "\n".join(f"User: {e['user']}\nAssistant: {e['assistant']}" for e in exchanges)


def facts_request(facts: list, exchanges: list) -> str:
    current = "\n".join(f"- {f}" for f in facts) if facts else "(none yet)"
    return (
        f"{MEMORY_UPDATE_TAG}\n"
        f"These are the facts you currently keep about the person:\n{current}\n\n"
        f"The newest exchanges, not yet in your memory:\n{format_exchanges(exchanges)}\n\n"
        "Rewrite the COMPLETE list using everything in this conversation:\n"
        "- Keep facts that are still true.\n"
        "- Add new lasting facts: preferences, family, work, hobbies, plans, commitments.\n"
        "- Conversational preferences and boundaries count. If they asked to stop discussing a "
        "topic, store: \"Topic boundary: <topic> - do not bring it up; they will raise it if they want.\"\n"
        "- Update facts the conversation revises; never keep both versions.\n"
        "- Drop moods, one-day states, fragments and duplicates.\n"
        "- One concise sentence per fact, at least four words. Call them \"the user\", never by name.\n\n"
        "Output ONLY the list, one fact per line starting with \"- \". No tags, no commentary. "
        "If nothing is worth keeping, output \"- No notable facts\"."
    )


def summary_request(summary: str, exchanges: list) -> str:
    return (
        f"{MEMORY_UPDATE_TAG}\n"
        "This is your running summary of all your conversations with the person so far:\n"
        f"{summary.strip() or '(none yet)'}\n\n"
        f"Update it so it also covers these latest exchanges:\n{format_exchanges(exchanges)}\n\n"
        "Keep what still matters from the old summary, fold in what is new, and drop small talk. "
        "Call them \"the user\", never by name. Write one paragraph of at most 200 words. "
        "Output ONLY the summary. No tags, no commentary."
    )


def clean_llm_text(text: str) -> str:
    """Strip <<directive>> tags her persona may add to a memory-update reply."""
    return strip_directives(text or "").strip()


def parse_facts(text: str) -> list:
    facts = []
    for line in clean_llm_text(text).splitlines():
        fact = line.strip().lstrip("-•* ").strip()
        if fact and len(fact.split()) >= 4 and fact.lower() != "no notable facts":
            facts.append(fact)
    return facts
