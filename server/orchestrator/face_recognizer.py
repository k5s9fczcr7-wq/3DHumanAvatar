"""
Face Recognizer

Generates face embeddings from FaceTracker crops using face_recognition (dlib).
Matches against FaceLibrary (SQLite) to identify returning users.

Runs as an async loop polled from the orchestrator. Recognition is triggered:
  - Once when a face first appears (no-face → face transition)
  - Periodically (every RECHECK_INTERVAL_S) while face is present
  - NOT every frame — embedding generation is ~100ms on CPU

Session lifecycle:
  face appears → generate embedding → match against FaceLibrary →
  set current_user on orchestrator → face lost → end session
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import face_recognition
import numpy as np

from .face_library import FaceLibrary
from .face_tracker import FaceTracker

logger = logging.getLogger("face_recognizer")

# ── Configuration ────────────────────────────────────────────────────────────

RECHECK_INTERVAL_S = 10.0     # re-identify while face is present (detect face swaps)
FACE_GONE_TIMEOUT_S = 5.0     # face must be absent this long to end session
MIN_CROP_SIZE = 60            # minimum face crop dimension (pixels) for reliable encoding
FACE_SWAP_THRESHOLD = 0.5    # if recheck embedding sim drops below this vs session, it's a different person
EMBEDDING_REDUNDANT_SIM = 0.95  # above this, new embedding is too similar to stored — skip
EMBEDDING_MIN_SIM = 0.50        # below this, embedding is suspicious — don't store
CENTER_TOLERANCE = 0.30        # face must be within ±30% of frame center to trigger events

# Confirmed-session lifetime is bound to the AUTHENTICATED user's own face, not
# to the presence of any face (a new person in line must not inherit the session).
AUTH_SESSION_GONE_S = 10.0     # end a confirmed session this long after this user's face was last verified-present
AUTH_VERIFY_INTERVAL_S = 2.0   # how often to re-encode + self-compare while a confirmed session is active (dlib ~100ms)
AUTH_MATCH_THRESHOLD = 0.5     # cosine sim to the session embedding that counts as "still the same person"


@dataclass
class UserSession:
    """Active user session state."""
    user_id: Optional[int] = None
    name: Optional[str] = None
    tier: str = "unknown"              # "confident" | "soft" | "unknown"
    similarity: float = 0.0
    visit_count: int = 0
    memory: Optional[str] = None
    session_db_id: Optional[int] = None
    started_at: float = 0.0


class FaceRecognizer:
    """Identifies users via face_recognition embeddings + FaceLibrary matching."""

    def __init__(self, face_tracker: FaceTracker, face_library: FaceLibrary):
        self._tracker = face_tracker
        self._library = face_library

        # State
        self._session: Optional[UserSession] = None
        self._session_embedding: Optional[np.ndarray] = None  # embedding of the tracked face
        self._pin_verified: bool = False  # only add embeddings after PIN confirmation
        self._last_recognition_time: float = 0.0
        self._face_was_present: bool = False
        self._face_lost_time: float = 0.0
        # Authenticated-presence tracking (confirmed sessions only). Session ends
        # AUTH_SESSION_GONE_S after this user's face was last verified-present.
        self._authenticated_last_seen: float = 0.0
        self._last_auth_verify: float = 0.0

    @property
    def session(self) -> Optional[UserSession]:
        """Current active user session (None if no user present)."""
        return self._session

    @property
    def has_user(self) -> bool:
        return self._session is not None

    @property
    def user_name(self) -> Optional[str]:
        return self._session.name if self._session else None

    @property
    def user_id(self) -> Optional[int]:
        return self._session.user_id if self._session else None

    def get_user_context(self) -> Optional[dict]:
        """Return user context dict for LLM priming, or None."""
        if not self._session or not self._session.user_id:
            return None
        return self._library.get_context(self._session.user_id)

    # ── Main update (called from orchestrator async loop) ────────────────

    def expire_if_gone(self) -> bool:
        """Wall-clock expiry for ANY session, safe to call while the
        recognition loop is gated (UE disconnected). While PIE is stopped,
        update() does not run, so without this a session would freeze mid-life
        (a PIN-verified one would resume hours later without re-auth; a
        tentative one would leave the id flow stuck, so a returning face would
        trigger no greeting). Presence truth comes from the tracker thread,
        which runs regardless of UE. Cheap (flag + timestamp compares; no
        encoding). Returns True if it ended a session.
        """
        if self._session is None:
            return False
        now = time.monotonic()
        if self._tracker.state.present:
            self._face_lost_time = 0.0
            if self._pin_verified:
                # Anti-inheritance: a present face that hasn't MATCHED the
                # session embedding for AUTH_SESSION_GONE_S is an impostor —
                # but only judge that while update()'s self-compare has
                # actually been running (verify timestamps fresh). While the
                # loop is gated (UE off) there is no verification, so the
                # session is held rather than falsely expiring the real user
                # standing through a PIE restart; an impostor is then evicted within
                # ~2×AUTH_VERIFY_INTERVAL_S + AUTH_SESSION_GONE_S of reconnect.
                verify_fresh = (now - self._last_auth_verify) <= (2 * AUTH_VERIFY_INTERVAL_S)
                if verify_fresh and (now - self._authenticated_last_seen) >= AUTH_SESSION_GONE_S:
                    logger.info(
                        f"[FACE ID] Present face hasn't matched session for "
                        f">{AUTH_SESSION_GONE_S:.0f}s - ending session"
                    )
                    self._face_was_present = False
                    self._end_session()
                    return True
            return False
        if self._pin_verified:
            if (now - self._authenticated_last_seen) < AUTH_SESSION_GONE_S:
                return False
            logger.info(
                f"[FACE ID] Authenticated user gone >{AUTH_SESSION_GONE_S:.0f}s "
                f"- ending session (next person identifies fresh)"
            )
        else:
            if self._face_lost_time == 0.0:
                self._face_lost_time = now
            if (now - self._face_lost_time) < FACE_GONE_TIMEOUT_S:
                return False
            logger.info(
                f"[FACE ID] Face gone >{FACE_GONE_TIMEOUT_S:.0f}s with "
                f"unverified session - ending it (returning face identifies fresh)"
            )
        # Reset so a person now in frame is recognized promptly next poll.
        self._face_was_present = False
        self._face_lost_time = 0.0
        self._end_session()
        return True

    def update(self) -> Optional[str]:
        """Check face state and run recognition if needed.

        Returns an event string if something happened:
          "user_identified"  — known user recognized (confident or soft)
          "new_face"         — unknown face detected
          "user_left"        — face absent long enough to end session
          None               — no event
        """
        state = self._tracker.state
        now = time.monotonic()

        # Confirmed (PIN-verified) session: lifetime is bound to the AUTHENTICATED
        # user's OWN face, not to the presence of any face. A new person stepping
        # into frame must NOT inherit this session, so the present face is verified
        # against the session embedding (self-compare only; never a library
        # re-match -> no identity flip) and the session ends AUTH_SESSION_GONE_S
        # after this user's face was last seen.
        if self._session is not None and self._pin_verified:
            if state.present and self._is_face_centered(state):
                if (now - self._last_auth_verify) >= AUTH_VERIFY_INTERVAL_S:
                    self._last_auth_verify = now
                    if self._face_matches_session(state.face_crop):
                        self._authenticated_last_seen = now
            if self.expire_if_gone():
                return "user_left"
            return None

        if state.present:
            self._face_lost_time = 0.0

            # Only trigger identification for centered faces (not passers-by)
            if not self._is_face_centered(state):
                return None

            if not self._face_was_present:
                # Transition: no face → face appeared
                self._face_was_present = True
                logger.info("Face appeared (centered) — running recognition")
                return self._recognize(state.face_crop)

            # Periodic re-check: only if session is still unidentified and
            # trying to resolve. Once a user is locked in (tier confident
            # or soft), stay locked until the face leaves the frame for
            # FACE_GONE_TIMEOUT_S seconds. This prevents identity flips mid-
            # session when the library contains multiple high-similarity
            # users.
            needs_recheck = (
                self._session is None
                or self._session.user_id is None
                or self._session.tier == "unknown"
            )
            if needs_recheck and (now - self._last_recognition_time) >= RECHECK_INTERVAL_S:
                logger.debug("Periodic re-recognition (unidentified session)")
                return self._recognize(state.face_crop)

        else:
            # Face not present
            if self._face_was_present:
                if self._face_lost_time == 0.0:
                    self._face_lost_time = now
                elif (now - self._face_lost_time) >= FACE_GONE_TIMEOUT_S:
                    # Face has been gone long enough — end session
                    self._face_was_present = False
                    self._face_lost_time = 0.0
                    return self._end_session()

        return None

    @staticmethod
    def _is_face_centered(state) -> bool:
        """Check if the detected face is roughly centered in the frame.

        Prevents passers-by at frame edges from triggering identification.
        """
        if state.bbox is None or state.full_frame is None:
            return False

        x, y, w, h = state.bbox
        frame_h, frame_w = state.full_frame.shape[:2]

        # Face center as fraction of frame
        face_cx = (x + w / 2) / frame_w
        face_cy = (y + h / 2) / frame_h

        # Must be within ±CENTER_TOLERANCE of frame center (0.5)
        return (abs(face_cx - 0.5) <= CENTER_TOLERANCE and
                abs(face_cy - 0.5) <= CENTER_TOLERANCE)

    def _face_matches_session(self, face_crop) -> bool:
        """True if face_crop encodes to a face matching the CURRENT session
        embedding. Self-comparison only - never a library re-match, so it cannot
        trigger an identity flip. Used to keep a confirmed
        session alive only while THIS authenticated user's face is present."""
        if face_crop is None or self._session_embedding is None:
            return False
        h, w = face_crop.shape[:2]
        if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
            return False
        try:
            from .face_library import cosine_sim
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb, [(0, w, h, 0)])
            if not encs:
                return False
            return cosine_sim(encs[0], self._session_embedding) >= AUTH_MATCH_THRESHOLD
        except Exception:
            return False

    # ── Recognition ──────────────────────────────────────────────────────

    def _recognize(self, face_crop: Optional[np.ndarray]) -> Optional[str]:
        """Generate embedding from face crop and match against library."""
        self._last_recognition_time = time.monotonic()

        if face_crop is None:
            return None

        # Validate crop size
        h, w = face_crop.shape[:2]
        if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
            logger.debug(f"Face crop too small ({w}x{h}), skipping recognition")
            return None

        t0 = time.monotonic()

        # face_recognition expects RGB
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Generate 128-D embedding
        # Use the full crop as the face location (the crop IS the face)
        face_locations = [(0, w, h, 0)]  # (top, right, bottom, left) for face_recognition
        encodings = face_recognition.face_encodings(rgb_crop, known_face_locations=face_locations)

        if not encodings:
            logger.debug("No face encoding generated from crop")
            return None

        embedding = encodings[0]  # 128-D numpy array
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(f"Face encoding generated in {elapsed_ms:.0f}ms")

        # Face-swap detection: compare against the currently tracked embedding
        if self._session_embedding is not None and self._session is not None:
            from .face_library import cosine_sim
            swap_sim = cosine_sim(embedding, self._session_embedding)
            if swap_sim < FACE_SWAP_THRESHOLD:
                logger.info(f"Face swap detected (sim={swap_sim:.3f} < {FACE_SWAP_THRESHOLD}) — ending old session")
                self._end_session()
                self._face_was_present = True  # still a face present, just different
                # Fall through to recognize the new face

        # Match against library
        user_id, name, similarity, tier = self._library.find_match(embedding)

        # Check if this is a state change or just a recheck of the same user
        already_tracking = (
            self._session is not None
            and self._session.user_id == user_id
            and (user_id is not None or self._session.tier == "unknown")
        )

        if tier == "confident":
            logger.info(f"Confident match: {name} (id={user_id}, sim={similarity:.3f})")
            self._start_session(user_id, name, tier, similarity, embedding)
            return None if already_tracking else "user_identified"

        elif tier == "soft":
            logger.info(f"Soft match: {name}? (id={user_id}, sim={similarity:.3f})")
            self._start_session(user_id, name, tier, similarity, embedding)
            return None if already_tracking else "user_identified"

        else:
            logger.info(f"Unknown face (best sim={similarity:.3f})")
            self._start_session(None, None, "unknown", similarity, embedding)
            return None if already_tracking else "new_face"

    def _start_session(
        self,
        user_id: Optional[int],
        name: Optional[str],
        tier: str,
        similarity: float,
        embedding: np.ndarray,
    ):
        """Start or update user session."""
        # If a session exists for a different user, end it first
        if self._session and self._session.user_id != user_id:
            self._end_session()

        if self._session and self._session.user_id == user_id:
            # Same user — update tracking embedding but only store if PIN verified
            self._session_embedding = embedding
            if user_id is not None and self._pin_verified:
                self._add_diverse_embedding(user_id, embedding)
            return

        # New session
        now = time.monotonic()
        session = UserSession(
            user_id=user_id,
            name=name,
            tier=tier,
            similarity=similarity,
            started_at=now,
        )

        if user_id is not None:
            # Known user — don't update last_seen or start DB session until PIN verified
            ctx = self._library.get_context(user_id)
            session.visit_count = ctx.get("visit_count", 0)
            session.memory = ctx.get("memory")
            logger.info(
                f"Tentative match for {name} (id={user_id}, visits={session.visit_count}) — awaiting PIN"
            )
        else:
            logger.info("Session started for unknown user")

        self._session = session
        self._session_embedding = embedding

    def _end_session(self) -> Optional[str]:
        """End the current user session."""
        if not self._session:
            return None

        if self._session.session_db_id is not None:
            self._library.end_session(self._session.session_db_id, summary="")
            logger.info(
                f"Session ended for {self._session.name} "
                f"(id={self._session.user_id}, "
                f"duration={time.monotonic() - self._session.started_at:.0f}s)"
            )
        else:
            logger.info("Session ended for unknown user")

        self._session = None
        self._session_embedding = None
        self._pin_verified = False
        self._last_recognition_time = 0.0
        self._authenticated_last_seen = 0.0
        self._last_auth_verify = 0.0
        return "user_left"

    # ── PIN Confirmation ─────────────────────────────────────────────────

    def confirm_identity(self):
        """Called after PIN verification succeeds. Unlocks embedding storage."""
        self._pin_verified = True
        if self._session and self._session.user_id:
            uid = self._session.user_id
            # NOW update last_seen and start DB session (deferred from _start_session)
            self._library.update_last_seen(uid)
            self._session.session_db_id = self._library.start_session(uid)
            # Store the current session embedding (first confirmed one)
            if self._session_embedding is not None:
                self._add_diverse_embedding(uid, self._session_embedding)
            # Start the authenticated-presence clock. From here, session-end is
            # bound to THIS user's face being verified-present (not any face).
            self._authenticated_last_seen = time.monotonic()
            self._last_auth_verify = time.monotonic()
            logger.info(f"[FACE ID] Identity confirmed for user {uid} — last_seen + embedding updated")

    def _add_diverse_embedding(self, user_id: int, embedding: np.ndarray):
        """Add embedding only if it adds diversity to the stored set.

        Skips if too similar to best existing (redundant) or too dissimilar (suspicious).
        This builds a diverse gallery of confirmed embeddings over time —
        covering different lighting, angles, grooming, makeup, etc.
        """
        from .face_library import cosine_sim

        # Get existing embeddings for this user
        with self._library._lock:
            with self._library._connect() as conn:
                rows = conn.execute(
                    "SELECT embedding FROM embeddings WHERE user_id = ?", (user_id,)
                ).fetchall()

        if not rows:
            # No existing embeddings — always store
            self._library.add_embedding(user_id, embedding)
            logger.debug(f"Stored first embedding for user {user_id}")
            return

        # Compare against all stored embeddings
        best_sim = max(
            cosine_sim(embedding, np.frombuffer(row["embedding"], dtype=np.float32))
            for row in rows
        )

        if best_sim > EMBEDDING_REDUNDANT_SIM:
            logger.debug(f"Embedding too similar (sim={best_sim:.3f} > {EMBEDDING_REDUNDANT_SIM}), skipping")
        elif best_sim < EMBEDDING_MIN_SIM:
            logger.warning(f"Embedding too dissimilar (sim={best_sim:.3f} < {EMBEDDING_MIN_SIM}), skipping — possible wrong face")
        else:
            self._library.add_embedding(user_id, embedding)
            logger.info(f"Stored diverse embedding for user {user_id} (sim={best_sim:.3f}, {len(rows)+1} total)")

    # ── User Registration (called from orchestrator when user gives name) ─

    def register_user(self, name: str) -> int:
        """Register the current unknown face as a named user.

        Call this when the user introduces themselves during conversation.
        Returns the new user_id.
        """
        if not self._session:
            logger.warning("register_user called with no active session")
            # Still register with no embedding
            return self._library.add_user(name, None)

        # Get current face embedding
        state = self._tracker.state
        embedding = None
        if state.present and state.face_crop is not None:
            rgb = cv2.cvtColor(state.face_crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            encodings = face_recognition.face_encodings(rgb, [(0, w, h, 0)])
            if encodings:
                embedding = encodings[0]

        user_id = self._library.add_user(name, embedding)

        # Update session
        self._session.user_id = user_id
        self._session.name = name
        self._session.tier = "confident"
        self._session.session_db_id = self._library.start_session(user_id)
        logger.info(f"Registered new user: {name} (id={user_id})")

        return user_id
