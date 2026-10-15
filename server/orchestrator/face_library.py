"""Face recognition library.

Stores per-user face embeddings, session history, and topic summaries in SQLite.
Embedding generation is handled by face_recognizer.py (face_recognition/dlib, 128-D).
This module is the storage and matching layer — embedding-format agnostic.
"""

import hashlib
import os
import sqlite3
import time
import threading
import numpy as np
from datetime import datetime
from typing import Optional

# ── Similarity thresholds (cosine similarity, higher = more similar) ──────────
CONF_THRESHOLD = 0.65   # ≥ this → confident match, greet by name
SOFT_THRESHOLD = 0.50   # ≥ this → soft match, ask "Are you X?"
MAX_EMBEDDINGS_PER_USER = 5  # keep the most recent N embeddings per user


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


class FaceLibrary:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ── DB setup ───────────────────────────────────────────────────────────────

    def _connect(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT    NOT NULL,
                    first_seen  TEXT    NOT NULL,
                    last_seen   TEXT    NOT NULL,
                    visit_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL REFERENCES users(id),
                    embedding   BLOB    NOT NULL,
                    created_at  TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL REFERENCES users(id),
                    started_at  TEXT    NOT NULL,
                    ended_at    TEXT,
                    summary     TEXT
                );
            """)
            # Migrate: add columns if not present (safe on existing DBs)
            for col, coltype in [("memory", "TEXT"), ("pin_hash", "TEXT"), ("pin_salt", "TEXT")]:
                try:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {col} {coltype}")
                except Exception:
                    pass  # column already exists

    # ── Core matching ──────────────────────────────────────────────────────────

    def find_match(self, embedding: np.ndarray) -> tuple:
        """Compare embedding against all stored users.

        Returns (user_id, name, similarity, tier) where tier is
        'confident' | 'soft' | 'unknown'.
        """
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT e.user_id, e.embedding, u.name "
                    "FROM embeddings e JOIN users u ON e.user_id = u.id"
                ).fetchall()

        if not rows:
            return (None, None, 0.0, "unknown")

        # Best similarity per user
        best_per_user: dict[int, tuple] = {}
        for row in rows:
            uid = row["user_id"]
            stored = np.frombuffer(row["embedding"], dtype=np.float32)
            sim = cosine_sim(embedding, stored)
            if uid not in best_per_user or sim > best_per_user[uid][0]:
                best_per_user[uid] = (sim, row["name"])

        best_uid = max(best_per_user, key=lambda u: best_per_user[u][0])
        best_sim, best_name = best_per_user[best_uid]

        if best_sim >= CONF_THRESHOLD:
            tier = "confident"
        elif best_sim >= SOFT_THRESHOLD:
            tier = "soft"
        else:
            tier = "unknown"

        return (best_uid, best_name, best_sim, tier)

    # ── User management ────────────────────────────────────────────────────────

    def add_user(self, name: str, embedding: Optional[np.ndarray]) -> int:
        """Create a new user and optionally store their first embedding."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO users (name, first_seen, last_seen, visit_count) "
                    "VALUES (?, ?, ?, 1)",
                    (name, now, now)
                )
                user_id = cur.lastrowid
                if embedding is not None:
                    conn.execute(
                        "INSERT INTO embeddings (user_id, embedding, created_at) VALUES (?, ?, ?)",
                        (user_id, embedding.astype(np.float32).tobytes(), now)
                    )
        return user_id

    def add_embedding(self, user_id: int, embedding: np.ndarray):
        """Add a new embedding for an existing user, keeping only the newest N."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO embeddings (user_id, embedding, created_at) VALUES (?, ?, ?)",
                    (user_id, embedding.astype(np.float32).tobytes(), now)
                )
                # Prune oldest embeddings beyond the limit
                conn.execute(
                    "DELETE FROM embeddings WHERE user_id = ? AND id NOT IN ("
                    "  SELECT id FROM embeddings WHERE user_id = ? "
                    "  ORDER BY created_at DESC LIMIT ?"
                    ")",
                    (user_id, user_id, MAX_EMBEDDINGS_PER_USER)
                )

    def update_last_seen(self, user_id: int):
        now = datetime.utcnow().isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE users SET last_seen = ?, visit_count = visit_count + 1 WHERE id = ?",
                    (now, user_id)
                )

    def get_context(self, user_id: int) -> dict:
        """Return context dict for LLM priming."""
        with self._lock:
            with self._connect() as conn:
                user = conn.execute(
                    "SELECT name, last_seen, visit_count, memory FROM users WHERE id = ?",
                    (user_id,)
                ).fetchone()

        if not user:
            return {}

        try:
            dt = datetime.fromisoformat(user["last_seen"])
            last_seen_str = dt.strftime("%B %d at %I:%M %p")
        except Exception:
            last_seen_str = user["last_seen"]

        return {
            "name":        user["name"],
            "last_seen":   last_seen_str,
            "visit_count": user["visit_count"],
            "memory":      user["memory"],  # accumulated rolling memory (may be None)
        }

    def get_memory(self, user_id: int) -> str:
        """Return the current accumulated memory text for a user (empty string if none)."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT memory FROM users WHERE id = ?", (user_id,)
                ).fetchone()
        return (row["memory"] or "") if row else ""

    def update_memory(self, user_id: int, memory_text: str):
        """Replace the accumulated memory text for a user."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE users SET memory = ? WHERE id = ?",
                    (memory_text, user_id)
                )

    # ── PIN management ─────────────────────────────────────────────────────────

    @staticmethod
    def _hash_pin(pin: str, salt: str) -> str:
        """Hash a 4-digit PIN with a per-user salt. This is a kiosk convenience
        feature, not strong authentication. The per-user salt gives two users
        with the same PIN different hashes, so collisions can't accidentally
        authenticate across accounts."""
        return hashlib.sha256(f"{salt}{pin}".encode()).hexdigest()

    def set_pin(self, user_id: int, pin: str):
        """Store a hashed PIN for a user with a freshly-generated random salt."""
        salt = os.urandom(16).hex()
        pin_hash = self._hash_pin(pin, salt)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE users SET pin_hash = ?, pin_salt = ? WHERE id = ?",
                    (pin_hash, salt, user_id)
                )

    def verify_pin(self, user_id: int, pin: str) -> bool:
        """Check if a PIN matches the stored hash.

        Backward compatibility: a user with no `pin_salt` uses the fixed
        "kiosk-pin-" salt. New registrations always use a random salt.
        """
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT pin_hash, pin_salt FROM users WHERE id = ?", (user_id,)
                ).fetchone()
        if not row or not row["pin_hash"]:
            return False
        salt = row["pin_salt"] if row["pin_salt"] else "kiosk-pin-"
        return row["pin_hash"] == self._hash_pin(pin, salt)

    def has_pin(self, user_id: int) -> bool:
        """Check if a user has a PIN set."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT pin_hash FROM users WHERE id = ?", (user_id,)
                ).fetchone()
        return bool(row and row["pin_hash"])

    # ── Session management ─────────────────────────────────────────────────────

    def start_session(self, user_id: int) -> int:
        now = datetime.utcnow().isoformat()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO sessions (user_id, started_at) VALUES (?, ?)",
                    (user_id, now)
                )
                return cur.lastrowid

    def end_session(self, session_id: int, summary: str):
        now = datetime.utcnow().isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
                    (now, summary, session_id)
                )

    def list_users(self) -> list:
        """Return all users (for debugging)."""
        with self._lock:
            with self._connect() as conn:
                return [dict(r) for r in conn.execute(
                    "SELECT id, name, last_seen, visit_count FROM users ORDER BY last_seen DESC"
                ).fetchall()]
