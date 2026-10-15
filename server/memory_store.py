"""
Hierarchical Memory Store — SQLite-backed user memory with tiered compression.

Stores per-user memory across 5 tiers with a total budget of ~4K tokens.
Directives are stripped from assistant turns before storage (history scrubbing).

Tiers:
  verbatim      — last ~5 turns, lossless (~800 tokens)
  light         — 5-20 turns ago, filler removed (~300 tokens)
  medium        — 20-50 turns ago, topic/intent summaries (~200 tokens)
  facts         — 50+ turns or pinned facts (~150 tokens)
  cross_session — prior session summaries (~200 tokens)

Total budget: ≤4096 tokens regardless of session length.
"""

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

# Simple token estimation: ~4 chars per token for English
_CHARS_PER_TOKEN = 4

# Directive stripping regex (same as persona_prompt.py)
_DIRECTIVE_RE = re.compile(r"<<[^>]*>>")


def estimate_tokens(text: str) -> int:
    """Rough token count estimate. Good enough for budget enforcement."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def strip_directives(text: str) -> str:
    """Remove <<...>> directives from text. Preserves [laugh] etc."""
    return _DIRECTIVE_RE.sub("", text).strip()


# ─── Tier budgets (tokens) ────────────────────────────────────────────────────

TIER_BUDGETS = {
    "verbatim": 800,
    "light": 300,
    "medium": 200,
    "facts": 150,
    "cross_session": 200,
}
TOTAL_BUDGET = 4096  # hard cap


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    tier: str
    content: str
    token_count: int
    pinned: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class UserProfile:
    user_id: str
    user_name: str = ""
    created_at: float = 0.0
    last_seen: float = 0.0
    session_count: int = 0


@dataclass
class AssembledContext:
    user_id: str
    user_name: str
    context_text: str
    token_count: int
    tiers: dict[str, int] = field(default_factory=dict)


# ─── Memory Store ─────────────────────────────────────────────────────────────

class MemoryStore:
    """SQLite-backed hierarchical memory store."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "memory.db"
            )
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def open(self):
        """Open database and create tables if needed."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        print(f"[MemoryStore] Opened {self._db_path}")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                user_name TEXT DEFAULT '',
                created_at REAL NOT NULL,
                last_seen REAL NOT NULL,
                session_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS memory_tiers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                tier TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                pinned INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memory_user_tier
                ON memory_tiers(user_id, tier);

            CREATE TABLE IF NOT EXISTS session_transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                transcript_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                compressed INTEGER DEFAULT 0
            );

            -- Single-user mode: id of the newest exchange committed per user,
            -- so a commit replayed after a crash is not stored twice.
            CREATE TABLE IF NOT EXISTS commit_marks (
                user_id TEXT PRIMARY KEY,
                last_id INTEGER NOT NULL
            );
        """)

    # ─── User management ──────────────────────────────────────────────────

    def get_or_create_user(self, user_id: str, user_name: str = "") -> UserProfile:
        now = time.time()
        row = self._conn.execute(
            "SELECT user_id, user_name, created_at, last_seen, session_count FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if row:
            # Update last_seen
            self._conn.execute(
                "UPDATE users SET last_seen = ? WHERE user_id = ?", (now, user_id)
            )
            self._conn.commit()
            return UserProfile(
                user_id=row[0], user_name=row[1], created_at=row[2],
                last_seen=now, session_count=row[4],
            )

        # Create new user
        self._conn.execute(
            "INSERT INTO users (user_id, user_name, created_at, last_seen, session_count) VALUES (?, ?, ?, ?, 0)",
            (user_id, user_name, now, now),
        )
        self._conn.commit()
        return UserProfile(user_id=user_id, user_name=user_name, created_at=now, last_seen=now)

    def list_users(self) -> list[UserProfile]:
        rows = self._conn.execute(
            "SELECT user_id, user_name, created_at, last_seen, session_count FROM users ORDER BY last_seen DESC"
        ).fetchall()
        return [
            UserProfile(user_id=r[0], user_name=r[1], created_at=r[2], last_seen=r[3], session_count=r[4])
            for r in rows
        ]

    def update_user_name(self, user_id: str, user_name: str):
        self._conn.execute(
            "UPDATE users SET user_name = ? WHERE user_id = ?", (user_name, user_id)
        )
        self._conn.commit()

    def increment_session_count(self, user_id: str):
        self._conn.execute(
            "UPDATE users SET session_count = session_count + 1 WHERE user_id = ?",
            (user_id,),
        )
        self._conn.commit()

    # ─── Memory tier read/write ───────────────────────────────────────────

    def get_tier_entries(self, user_id: str, tier: str) -> list[MemoryEntry]:
        rows = self._conn.execute(
            "SELECT tier, content, token_count, pinned, created_at, updated_at "
            "FROM memory_tiers WHERE user_id = ? AND tier = ? ORDER BY updated_at DESC",
            (user_id, tier),
        ).fetchall()
        return [
            MemoryEntry(tier=r[0], content=r[1], token_count=r[2],
                        pinned=bool(r[3]), created_at=r[4], updated_at=r[5])
            for r in rows
        ]

    def set_tier(self, user_id: str, tier: str, content: str, pinned: bool = False):
        """Replace all non-pinned entries in a tier with new content."""
        now = time.time()
        tokens = estimate_tokens(content)

        if not pinned:
            # Clear non-pinned entries in this tier
            self._conn.execute(
                "DELETE FROM memory_tiers WHERE user_id = ? AND tier = ? AND pinned = 0",
                (user_id, tier),
            )

        self._conn.execute(
            "INSERT INTO memory_tiers (user_id, tier, content, token_count, pinned, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, tier, content, tokens, int(pinned), now, now),
        )
        self._conn.commit()

    def add_fact(self, user_id: str, fact: str, pinned: bool = True):
        """Add a single fact to the facts tier."""
        now = time.time()
        tokens = estimate_tokens(fact)
        self._conn.execute(
            "INSERT INTO memory_tiers (user_id, tier, content, token_count, pinned, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, "facts", fact, tokens, int(pinned), now, now),
        )
        self._conn.commit()

    def clear_tier(self, user_id: str, tier: str, keep_pinned: bool = True):
        """Clear a tier's entries. Optionally keep pinned items."""
        if keep_pinned:
            self._conn.execute(
                "DELETE FROM memory_tiers WHERE user_id = ? AND tier = ? AND pinned = 0",
                (user_id, tier),
            )
        else:
            self._conn.execute(
                "DELETE FROM memory_tiers WHERE user_id = ? AND tier = ?",
                (user_id, tier),
            )
        self._conn.commit()

    # ─── Context assembly ─────────────────────────────────────────────────

    def assemble_context(self, user_id: str) -> AssembledContext:
        """
        Build the full memory context block for LLM injection.
        Concatenates all tiers within budget, prioritizing pinned facts
        and recent verbatim turns.
        """
        user = self.get_or_create_user(user_id)
        sections = []
        tier_tokens = {}
        total_tokens = 0

        # Assembly order: facts first (always included), then verbatim → light → medium → cross_session
        for tier in ["facts", "verbatim", "light", "medium", "cross_session"]:
            budget = TIER_BUDGETS[tier]
            entries = self.get_tier_entries(user_id, tier)

            tier_text_parts = []
            tier_token_sum = 0

            for entry in entries:
                if total_tokens + entry.token_count > TOTAL_BUDGET:
                    break
                if tier_token_sum + entry.token_count > budget and not entry.pinned:
                    continue  # Over tier budget, skip non-pinned
                tier_text_parts.append(entry.content)
                tier_token_sum += entry.token_count
                total_tokens += entry.token_count

            if tier_text_parts:
                sections.append(f"[{tier.upper()}]\n" + "\n".join(tier_text_parts))
                tier_tokens[tier] = tier_token_sum

        if not sections:
            context_text = ""
        else:
            context_text = "[MEMORY]\n" + "\n\n".join(sections) + "\n[/MEMORY]"

        return AssembledContext(
            user_id=user_id,
            user_name=user.user_name,
            context_text=context_text,
            token_count=total_tokens,
            tiers=tier_tokens,
        )

    # ─── Session transcript storage ───────────────────────────────────────

    def store_transcript(self, user_id: str, transcript: list[dict]):
        """
        Store a session transcript for later compression.
        Assistant turns are scrubbed of directives before storage.
        """
        # History scrubbing: strip directives from assistant turns
        scrubbed = []
        for turn in transcript:
            if turn.get("role") == "assistant":
                scrubbed.append({
                    "role": "assistant",
                    "content": strip_directives(turn["content"]),
                })
            else:
                scrubbed.append(turn)

        now = time.time()
        self._conn.execute(
            "INSERT INTO session_transcripts (user_id, transcript_json, created_at) VALUES (?, ?, ?)",
            (user_id, json.dumps(scrubbed), now),
        )
        self.increment_session_count(user_id)
        self._conn.commit()

    def get_uncompressed_transcripts(self, user_id: str) -> list[tuple[int, list[dict]]]:
        """Get transcripts not yet compressed. Returns [(id, transcript), ...]."""
        rows = self._conn.execute(
            "SELECT id, transcript_json FROM session_transcripts "
            "WHERE user_id = ? AND compressed = 0 ORDER BY created_at ASC",
            (user_id,),
        ).fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    def mark_transcript_compressed(self, transcript_id: int):
        self._conn.execute(
            "UPDATE session_transcripts SET compressed = 1 WHERE id = ?",
            (transcript_id,),
        )
        self._conn.commit()

    # ─── Delete user ──────────────────────────────────────────────────────

    def get_all_facts(self, user_id: str) -> list[str]:
        """Return all fact-tier content strings for dedup checking."""
        rows = self._conn.execute(
            "SELECT content FROM memory_tiers WHERE user_id = ? AND tier = 'facts'",
            (user_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def replace_all_facts(self, user_id: str, facts: list[str]):
        """Replace all facts with a consolidated set. Used after LLM dedup."""
        now = time.time()
        self._conn.execute(
            "DELETE FROM memory_tiers WHERE user_id = ? AND tier = 'facts'",
            (user_id,),
        )
        for fact in facts:
            tokens = estimate_tokens(fact)
            self._conn.execute(
                "INSERT INTO memory_tiers (user_id, tier, content, token_count, pinned, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 1, ?, ?)",
                (user_id, "facts", fact, tokens, now, now),
            )
        self._conn.commit()

    # ─── Single-user mode ─────────────────────────────────────────────────

    def get_state(self, user_id: str) -> dict:
        """Facts, running summary and recent exchanges, raw - without the
        per-tier budgets of assemble_context(). Single-user mode builds its
        own memory block from these."""
        summary = self.get_tier_entries(user_id, "cross_session")
        recent = self.get_tier_entries(user_id, "verbatim")
        return {
            "facts": self.get_all_facts(user_id),
            "summary": summary[0].content if summary else "",
            "recent": recent[0].content if recent else "",
        }

    def commit_single_user(self, user_id: str, exchanges: list[dict], facts: list[str],
                           summary: str, last_id: int, recent_n: int = 5) -> dict:
        """Single-user mode: store exchanges together with the facts and
        running summary the orchestrator distilled in its live LLM session,
        in one transaction. Idempotent on last_id - exchanges already
        committed (a commit replayed after a crash) are ignored."""
        mark = self._conn.execute(
            "SELECT last_id FROM commit_marks WHERE user_id = ?", (user_id,)
        ).fetchone()
        done_through = mark[0] if mark else -1
        if last_id <= done_through:
            return {"status": "already_committed", "last_id": done_through}

        self.get_or_create_user(user_id)
        existing = self.get_all_facts(user_id)
        # Same guardrails as MemoryCompressor.reconcile_facts: an empty or
        # sharply shrunken rewrite of an established list is rejected, so one
        # bad LLM reply cannot wipe memory.
        facts_ok = bool(facts) and not (len(existing) > 6 and len(facts) < 0.5 * len(existing))

        turns = []
        for e in exchanges:
            if e.get("id", 0) > done_through:
                turns.append({"role": "user", "content": e["user"]})
                turns.append({"role": "assistant", "content": strip_directives(e["assistant"])})

        now = time.time()
        with self._conn:  # one transaction: all of it or none of it
            self._conn.execute(
                "INSERT INTO session_transcripts (user_id, transcript_json, created_at, compressed) "
                "VALUES (?, ?, ?, 1)",
                (user_id, json.dumps(turns), now),
            )
            if facts_ok:
                self._conn.execute(
                    "DELETE FROM memory_tiers WHERE user_id = ? AND tier = 'facts'", (user_id,)
                )
                self._conn.executemany(
                    "INSERT INTO memory_tiers (user_id, tier, content, token_count, pinned, created_at, updated_at) "
                    "VALUES (?, 'facts', ?, ?, 1, ?, ?)",
                    [(user_id, f, estimate_tokens(f), now, now) for f in facts],
                )
            if summary.strip():
                self._replace_tier_in_txn(user_id, "cross_session", summary.strip(), now)
            recent = self._recent_exchanges_text(user_id, recent_n)
            if recent:
                self._replace_tier_in_txn(user_id, "verbatim", recent, now)
            self._conn.execute(
                "INSERT OR REPLACE INTO commit_marks (user_id, last_id) VALUES (?, ?)",
                (user_id, last_id),
            )
            self._conn.execute("UPDATE users SET last_seen = ? WHERE user_id = ?", (now, user_id))

        return {
            "status": "ok",
            "stored": len(turns) // 2,
            "facts_before": len(existing),
            "facts_after": len(facts) if facts_ok else len(existing),
            # Only a rewrite that would have cut existing facts counts as a
            # rejection; nothing new to remember is just nothing.
            "facts_rejected": not facts_ok and bool(existing),
        }

    def _replace_tier_in_txn(self, user_id: str, tier: str, content: str, now: float):
        """set_tier() without its commit, for use inside a transaction."""
        self._conn.execute(
            "DELETE FROM memory_tiers WHERE user_id = ? AND tier = ? AND pinned = 0",
            (user_id, tier),
        )
        self._conn.execute(
            "INSERT INTO memory_tiers (user_id, tier, content, token_count, pinned, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 0, ?, ?)",
            (user_id, tier, content, estimate_tokens(content), now, now),
        )

    def _recent_exchanges_text(self, user_id: str, n: int) -> str:
        """The last n exchanges across stored transcripts, word for word."""
        rows = self._conn.execute(
            "SELECT transcript_json FROM session_transcripts WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, n),
        ).fetchall()
        turns = []
        for (transcript_json,) in reversed(rows):
            turns.extend(json.loads(transcript_json))
        turns = turns[-2 * n:]
        while turns and turns[0].get("role") != "user":
            turns = turns[1:]
        return "\n".join(
            f"{'User' if t.get('role') == 'user' else 'Assistant'}: {t.get('content', '')}" for t in turns
        )

    def delete_user(self, user_id: str):
        """Delete user and all their memory (CASCADE), plus the single-user
        commit mark, which has no foreign key to cascade from."""
        self._conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        self._conn.execute("DELETE FROM commit_marks WHERE user_id = ?", (user_id,))
        self._conn.commit()
