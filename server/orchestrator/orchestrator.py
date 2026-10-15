"""
3DHumanAvatar — Orchestrator

Asyncio service that drives the full conversation loop:
  mic → VAD → ASR → context assembly → LLM → directive strip → TTS → WebSocket to UE5
  + state management

Runs on the same machine as the FastAPI server and llama-server.
"""

import asyncio
import configparser
import glob
import json
import logging
import os
import sys
import tempfile
import time
import uuid
import wave
import re
from collections import deque
from enum import Enum
from typing import Optional

import httpx
import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt

from .persona import AVATAR_NAME
from .persona_prompt import SYSTEM_PROMPT
from .emotion_sync import EmotionSynchronizer, schedule_emotion_updates
from .gesture_planner import GesturePacer, load_gesture_config
from .face_tracker import FaceTracker
from .face_recognizer import FaceRecognizer
from .face_library import FaceLibrary
from .face_emotion import FaceEmotionDetector, MoodTracker
from .user_identification import IdentificationManager, IdState
from .confirm_intent import classify_confirmation, extract_name
from .gaze_calculator import CONFIG_PATH as KIOSK_CONFIG_PATH, load_config as load_kiosk_config, GazeCalculator
from .glance_planner import GlancePlanner, load_glance_config
from .identity_mode import IdentityConfig, load_identity_config
from .knowledge import (KB_TOKEN_BUDGET, KIOSK_PRONUNCIATIONS, PRONUNCIATIONS_FILE, Pronunciations, asr_hint,
                        find_kb_folder, kb_block, kb_names, load_kb_config, read_kb)
from .pronunciation import (AUTO_PRONUNCIATIONS, LLM_BATCH, QUIET_S, SAY_REQUEST, PhonemeWorker, agree,
                            check_enabled, check_word, describe, listening_paused, llm_request, load_cache,
                            lookup_dictionary, parse_llm_table, save_cache, save_current_words, summarize,
                            take_say_request, today, words_to_check, write_auto_file)
from .single_user import (
    COMMIT_IDLE_S, COMPACT_AT_TOKENS, FIRST_GREETING, SESSION_PREFIX, SINGLE_USER_ID,
    WELCOME_BACK_AFTER_S, PendingExchanges, Presence, facts_request, parse_facts,
    clean_llm_text, profile_block, summary_request, welcome_back_note,
)
import websockets

# ── Configuration ───────────────────────────────────────────────────────────────

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8100                # All services on FastAPI: LLM Gateway, TTS, A2E, memory, dashboard
SERVER_BASE = f"http://{SERVER_HOST}:{SERVER_PORT}"

UE_WS_HOST = "127.0.0.1"
UE_WS_PORT = 8765

MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_BLOCKSIZE = 512  # ~32ms at 16kHz
# Match a substring against sounddevice device names; start() prefers the
# WASAPI instance. Falls back to system default (None) if no match.
MIC_DEVICE_NAME_PATTERN = "Jabra Speak"
MIC_DEVICE = None    # resolved at startup in start()

# VAD
VAD_THRESHOLD = 0.3
SILENCE_SPECULATIVE_MS = 250   # speculative LLM fork after this
SILENCE_CONFIRMED_MS = 500     # confirmed end-of-utterance

# Barge-in: always-listening single-pipeline VAD. The Jabra's hardware AEC
# (acoustic echo cancellation) keeps the mic floor near -60 dB during NPC
# playback (measured), so one VAD pipeline keeps listening while she speaks.
# A short sustained-frames filter rejects ultra-brief transients (cough, door
# slam) while keeping barge-in responsiveness near 160 ms.
BARGE_IN_TRIGGER_FRAMES = 5     # ~160 ms at 32 ms blocks; guards against transients
# Some AEC leak still reaches the mic during NPC playback and can push VAD past
# 0.3, which would make her interrupt herself. So DURING playback ONLY: count
# only high-confidence frames, require more of them sustained, after a short
# grace period. VAD_THRESHOLD (0.3) still applies to normal speech detection.
BARGE_IN_VAD_THRESHOLD_PLAYBACK = 0.5   # per-frame bar to count toward barge-in while she speaks
BARGE_IN_TRIGGER_FRAMES_PLAYBACK = 8    # ~256 ms sustained strong speech before honoring barge-in
BARGE_IN_GRACE_S = 0.6                   # ignore barge-in for the first 600 ms of her turn
CHUNK_SEAM_PAUSE_S = 0.7                 # min silence between sentence 1 and the merged remainder
HP_CUTOFF_HZ = 250              # high-pass filter for fan/HVAC hum

# Audio output
TTS_VOLUME_GAIN = 3.0           # full volume — AEC handles echo cancellation

# ASR
ASR_MODEL_SIZE = "large-v3-turbo"  # WER 2-3x lower than small on short utterances and digits
ASR_DEVICE_INDEX = 1               # second GPU (the first is reserved for the LLM + TTS)
# Local model dir — bypasses HuggingFace cache symlink permissions issue on
# Windows (requires admin or Developer Mode otherwise). Pre-downloaded via
# snapshot_download(repo_id="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
# local_dir=ASR_MODEL_PATH).
ASR_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "faster-whisper-large-v3-turbo")
ASR_BEAM_SIZE = 5

# Ring buffer: hold last 30 seconds of audio
RING_BUFFER_SECONDS = 30

# Face recognition database
FACE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faces.db")

# Emotion trajectory: rolling buffer of last 5 turns
EMOTION_TRAJECTORY_SIZE = 5

# Face-presence gate: drop utterances when nobody has been at the kiosk
# within FACE_GATE_S seconds. Prevents ambient room audio (calls, podcasts,
# meetings) from being transcribed and dispatched to the LLM.
FACE_GATE_S = 10.0

# How long to wait for a departed visitor's memory to be saved; the lapel pin
# is purple until then, but only while nobody is at the kiosk.
# A save normally takes ~5 s. The server defers it while the kiosk is busy -
# up to 5 min on a non-stop kiosk (COMPRESS_HARD_CAP_S in server.py) - and
# then makes up to 4 LLM calls, hence the generous ceiling.
SAVE_TIMEOUT_S = 600.0

# How long to stay quiet after telling a visitor the LLM is offline, so a
# person who keeps talking to an offline model hears the apology once, not
# once per utterance.
DEGRADED_REPEAT_S = 25.0

logger = logging.getLogger("orchestrator")

# ── State Machine ───────────────────────────────────────────────────────────────

class ConvState(Enum):
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "talking"


# ── Directive Stripper ──────────────────────────────────────────────────────────

# Directives that get stripped from TTS input and forwarded as events
DIRECTIVE_RE = re.compile(r"<<(\w+)(?::([^\s>]+))?\s*(?:intensity:([0-9.]+))?\s*>>")

def strip_directives(text: str) -> tuple[str, list[dict]]:
    """Remove <<directive>> tags from LLM output.

    Returns (clean_text, list_of_directives).
    Preserves [laugh], [chuckle], [sigh] for Chatterbox TTS.
    """
    directives = []

    def replacer(match):
        name = match.group(1)
        value = match.group(2) or ""
        intensity = match.group(3)
        d = {"name": name, "value": value}
        if intensity:
            d["intensity"] = float(intensity)
        directives.append(d)
        return ""

    clean = DIRECTIVE_RE.sub(replacer, text)
    # Collapse multiple spaces left by removals
    clean = re.sub(r"  +", " ", clean).strip()
    return clean, directives


# ── Orchestrator ────────────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(self):
        # State
        self.state = ConvState.LISTENING
        self.conversation_history: list[dict] = []
        self.emotion_trajectory: deque = deque(maxlen=EMOTION_TRAJECTORY_SIZE)
        self.current_user_emotion: str = "neutral"
        self.current_user_emotion_prob: float = 0.0
        self.current_user_name: Optional[str] = None
        self.current_user_id: Optional[int] = None
        self.current_user_similarity: float = 0.0   # set when confident_match fires
        self.current_face_emotion: str = "neutral"  # raw HSE label, updated ~10 Hz
        self.current_face_emotion_prob: float = 0.0
        self.session_start = time.time()

        # Audio
        self.ring_buffer: deque[tuple[float, np.ndarray]] = deque(
            maxlen=int(RING_BUFFER_SECONDS * MIC_SAMPLE_RATE / MIC_BLOCKSIZE)
        )
        self.speech_buffer: list[np.ndarray] = []
        self.silence_frames = 0
        self.is_speech_active = False
        self._vad_speech_start: float = 0.0       # when VAD first detected speech
        self._was_speaking_at_speech_start = False  # captured at speech-start: True ⇒ this utterance is a barge-in

        # Components (initialized in start())
        self.vad_model = None
        self.asr_model = None
        self.emotion_pipeline = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.ue_websockets: set = set()

        # Barge-in
        self._barge_in_event = asyncio.Event()

        # High-pass filter — designed once, applied per frame
        self._hp_sos = butter(2, HP_CUTOFF_HZ, btype='high', fs=MIC_SAMPLE_RATE, output='sos')

        # WAV cleanup: track speech-WAV paths emitted to UE5; unlink after the
        # next user turn's ASR completes (UE5 has finished playing by then).
        # Also sweep stale kiosk_speech_*.wav files at startup.
        self._pending_wav_unlinks: set[str] = set()

        # Face tracking + gaze + recognition + emotion
        self.face_tracker: Optional[FaceTracker] = None
        self.gaze_calculator: Optional[GazeCalculator] = None
        self.face_library: Optional[FaceLibrary] = None
        self.face_recognizer: Optional[FaceRecognizer | Presence] = None  # Presence in single_user mode
        self.face_emotion: Optional[FaceEmotionDetector] = None
        self.mood_tracker: Optional[MoodTracker] = None
        self.id_manager: Optional[IdentificationManager] = None
        # multi_user (default) or single_user - kiosk_config.ini [identity]
        self.identity = IdentityConfig()
        # Single-user mode only (single_user.py): uncommitted exchanges, when
        # the kiosk was last busy, and whether a memory commit is running.
        self._pending: Optional[PendingExchanges] = None
        self._last_activity_at: float = 0.0
        self._committing = False
        self._single_user_has_memory = False
        self._single_user_memory_loaded = False
        # Knowledge base read into the system prompt at startup (knowledge.py).
        self._kb_block: str = ""
        # Names: respellings for her voice, and the names Whisper listens for.
        self._kb_folder: Optional[str] = None
        self._kb_text: str = ""
        self._pronunciations = Pronunciations()
        self._pronunciation_files: list = []
        self._manual_names: set = set()   # in a hand-written pronunciations.txt
        self._asr_hint: str = ""
        # Background pronunciation check (pronunciation.py): runs only while quiet.
        self._pronunciation_task: Optional[asyncio.Task] = None
        self._last_busy_at: float = 0.0

        # Face-presence gate: timestamp of most recent face-present poll.
        # 0.0 means no face has ever been detected this session.
        self._last_face_seen_at: float = 0.0

        # Identity-flip preemption: set True while _speak_id_line is taking
        # over the speaker (face-poll task). _play_sentence() called from the
        # utterance task checks this and exits, so the utterance loop cannot
        # dispatch additional sentences alongside the id-line. Cleared in finally.
        self._suppress_utterance_playback: bool = False

        # UX mode indicator: the lapel-pin color state currently
        # broadcast to UE5. One of: grey/white/green/amber/red/purple.
        # _saves_pending counts departed visitors whose memory is still being
        # saved; while it is non-zero and nobody is at the kiosk, the pin shows
        # purple instead of grey.
        self._current_indicator: str = "grey"
        self._saves_pending: int = 0
        self._degraded_spoken_at: float = 0.0

        # Flags
        self._running = False
        self._barge_in_requested = False
        self._is_speaking = False
        self._speaking_started_at: float = 0.0   # when her current turn's speech began (barge-in grace timer)
        # Gestures (gesture_planner.py): her LLM gesture tags wait here until the chunk they belong to plays
        self._gesture_cfg = load_gesture_config()
        self._gestures = GesturePacer(self._gesture_cfg)
        self._pending_gesture_tags: list[str] = []
        self._last_emotion: tuple[str, float] = ("neutral", 0.3)   # last emotion sent to UE5
        self._idle_since: float = time.time()                      # nobody at the kiosk since
        self._bargein_strong_frames = 0          # consecutive high-conf VAD frames during her playback
        self._last_vad_prob: float = 0.0         # most recent VAD prob (for [BARGE-IN] logging)
        self._llm_lock = None      # asyncio.Lock, initialized in start()
        self._last_tts_ms: float = 0  # set by _call_tts for telemetry

        # Per-session identity + compressed memory, established ONCE when the user
        # is identified (set from the greeting payload) and injected into the
        # session system prompt — NOT re-asserted per turn. Cleared on
        # session reset. Empty for new/anonymous users.
        self._session_memory_block: str = ""

    # ── Initialization ──────────────────────────────────────────────────────

    async def start(self):
        """Initialize all components and start the main loops."""
        logger.info("Initializing orchestrator...")

        # Resolve mic device by name pattern (e.g. "Jabra Speak"). The same physical
        # device is enumerated once per host API (MME / DirectSound / WASAPI / WDM-KS),
        # so the first name match may not be the WASAPI one, and Windows can rename
        # the device when it re-enumerates it (e.g. "2- Jabra Speak 710"). The WASAPI
        # instance is REQUIRED because _run_mic_capture attaches a PaWasapiStreamInfo
        # (streamCategory=Speech, for barge-in detection) — passing that to a
        # non-WASAPI instance fails with PaErrorCode -9984. Fall back to any name
        # match (the stream then skips the WASAPI extra settings), else system default.
        global MIC_DEVICE
        try:
            wasapi_api = None
            for ha_i, ha in enumerate(sd.query_hostapis()):
                if "WASAPI" in ha["name"]:
                    wasapi_api = ha_i
                    break
            best_idx = None
            fallback_idx = None
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and MIC_DEVICE_NAME_PATTERN in d["name"]:
                    if d["hostapi"] == wasapi_api:
                        best_idx = i  # WASAPI instance — required for streamCategory=Speech
                        break
                    if fallback_idx is None:
                        fallback_idx = i
            if best_idx is None and fallback_idx is not None:
                logger.warning(f"'{MIC_DEVICE_NAME_PATTERN}' found but no WASAPI instance; "
                               f"using [{fallback_idx}] (streamCategory=Speech will be skipped)")
                best_idx = fallback_idx
            if best_idx is not None:
                MIC_DEVICE = best_idx
                d = sd.query_devices()[best_idx]
                api_name = sd.query_hostapis()[d["hostapi"]]["name"]
                logger.info(f"Mic device resolved to [{best_idx}] {d['name']} "
                            f"(api={api_name}, in={d['max_input_channels']}, sr={int(d['default_samplerate'])})")
            else:
                logger.warning(f"No mic device matched '{MIC_DEVICE_NAME_PATTERN}', using system default")
        except Exception as e:
            logger.warning(f"Mic device resolution failed: {e}; using system default")

        # Sweep stale TTS WAV files left over from a previous run / crash.
        # Every match is the kiosk's own (this or another instance) — unlink.
        try:
            stale = glob.glob(os.path.join(tempfile.gettempdir(), "kiosk_speech_*.wav"))
            for p in stale:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            if stale:
                logger.info(f"Cleaned {len(stale)} stale TTS WAV file(s) from {tempfile.gettempdir()}")
        except Exception:
            pass

        # HTTP client for all server services (LLM Gateway, TTS, A2E)
        self.http_client = httpx.AsyncClient(
            base_url=SERVER_BASE,
            timeout=httpx.Timeout(60.0, connect=5.0),
        )

        # Session ID for LLM Gateway (persistent across turns in one conversation)
        self.session_id = str(uuid.uuid4())
        logger.info(f"Session ID: {self.session_id}")

        # Serialize LLM calls — one at a time to prevent gateway conflicts
        self._llm_lock = asyncio.Lock()

        # VAD (Silero)
        logger.info("Loading Silero VAD...")
        import torch
        self.vad_model, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        (self._get_speech_timestamps, _, self._read_audio, _, _) = vad_utils
        self._vad_reset = True
        logger.info("Silero VAD loaded")

        # ASR (faster-whisper)
        logger.info(f"Loading faster-whisper ({ASR_MODEL_SIZE}) on cuda:{ASR_DEVICE_INDEX}...")
        from faster_whisper import WhisperModel
        self.asr_model = WhisperModel(
            ASR_MODEL_PATH,
            device="cuda",
            device_index=ASR_DEVICE_INDEX,
            compute_type="float16",
        )
        logger.info("ASR loaded")

        # emotion2vec (FunASR) — lazy load on first use to speed up startup
        self.emotion_pipeline = None  # loaded on demand

        # Face tracking + gaze + recognition
        logger.info("Starting face tracker + gaze calculator + face recognition...")
        kiosk_geom = load_kiosk_config()
        self.face_tracker = FaceTracker()
        if os.getenv("KIOSK_NO_CAMERA") == "1":       # tests that must leave the room camera off
            logger.info("Camera left off (KIOSK_NO_CAMERA=1)")
        else:
            self.face_tracker.start()
        self.gaze_calculator = GazeCalculator(kiosk_geom)
        self._glances = GlancePlanner(load_glance_config())     # she glances away now and then
        self._sent_voice_delay: Optional[int] = None
        self.face_library = FaceLibrary(FACE_DB_PATH)
        self.identity = load_identity_config()
        if self.identity.single_user:
            await self._start_single_user()
        else:
            self.face_recognizer = FaceRecognizer(self.face_tracker, self.face_library)
        await self._load_knowledge_base()
        self.face_emotion = FaceEmotionDetector()
        self.mood_tracker = MoodTracker()
        self.id_manager = IdentificationManager(avatar_name=AVATAR_NAME)
        logger.info("Face tracker + recognition + emotion + identification started")

        self._running = True
        logger.info("Orchestrator initialized — starting main loops")

        # Start concurrent tasks
        tasks = [
            self._run_websocket_server(),
            self._run_mic_capture(),
            self._run_health_check(),
            self._run_gaze_tracking(),
            self._run_face_recognition(),
            self._run_face_emotion(),
            self._watch_pronunciations(),
            self._run_idle_gestures(),
        ]
        if self.identity.single_user:
            tasks.append(self._run_single_user_commits())
        await self._run_until_stopped(tasks)

    async def _run_until_stopped(self, coros):
        """Run the main loops until a stop is requested or one of them fails. On a stop, each loop gets a moment
        to finish its current step, then the ones still asleep are cancelled: stop() releases the camera only after
        this returns, and _run_health_check alone sleeps 30 s, long enough for stop-stack.ps1 to give up and
        force-kill the orchestrator, which can wedge the C920 webcam."""
        running = [asyncio.ensure_future(c) for c in coros]
        try:
            while self._running:
                done, _ = await asyncio.wait(running, timeout=0.5, return_when=asyncio.FIRST_EXCEPTION)
                for t in done:
                    if not t.cancelled() and t.exception() is not None:
                        raise t.exception()
                if len(done) == len(running):
                    return
        finally:
            pending = [t for t in running if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=2.0)
            for t in running:
                t.cancel()
            await asyncio.gather(*running, return_exceptions=True)

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        if self._pronunciation_task:
            self._pronunciation_task.cancel()
        if self.face_tracker:
            self.face_tracker.stop()
        # End LLM session on the server
        if self.http_client:
            try:
                await self.http_client.post("/llm/end_session", json={"session_id": self.session_id})
            except Exception:
                pass
            await self.http_client.aclose()
        logger.info("Orchestrator stopped")

    # ── WebSocket Server (UE5 connection) ───────────────────────────────────

    async def _run_websocket_server(self):
        """WebSocket server on port 8765 — UE5 connects here."""
        async def handler(ws, path=None):
            # Safety net: expire a logically-dead PIN session BEFORE this
            # client is added and snapshotted. The ungated check in the face
            # recognition loop normally fires first; this covers the case where
            # that loop has stopped, so Play can never resume a stale auth session.
            try:
                if self.face_recognizer and self.face_recognizer.expire_if_gone():
                    logger.warning(
                        "[FACE ID] Stale auth session expired at UE connect "
                        "(ungated loop check missed it - investigate)"
                    )
                    self.current_user_name = None
                    self.current_user_id = None
                    self.current_user_similarity = 0.0
                    if self.id_manager:
                        self.id_manager.on_face_event("user_left")
            except Exception as e:
                logger.warning(f"Connect-time auth expiry failed: {e}")
            self.ue_websockets.add(ws)
            remote = ws.remote_address
            logger.info(f"UE5 connected from {remote}")
            # Push the CURRENT state + indicator snapshot to the new client.
            # State/indicator broadcasts are change-driven, so a UE that
            # (re)connects after they settle would otherwise render its BP
            # defaults (grey pin) until the next transition.
            try:
                await ws.send(json.dumps({"type": "state", "value": self.state.value}))
                await ws.send(json.dumps({"type": "indicator", "value": self._compute_indicator()}))
                self._sent_voice_delay = self._voice_delay_ms()
                await ws.send(json.dumps({"type": "av_sync", "voice_delay_ms": self._sent_voice_delay}))
                logger.info(f"Sent state/indicator snapshot to {remote}")
            except websockets.ConnectionClosed:
                pass
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    if data.get("type") == "ready":
                        logger.info("UE5 reports ready")
                    elif data.get("type") == "eyes_on_screen":
                        # Where UE5 draws her eyes (viewport px): the gaze geometry measures from there
                        if self.gaze_calculator is not None:
                            self.gaze_calculator.set_eyes_on_screen(
                                float(data.get("x", 0)), float(data.get("y", 0)),
                                float(data.get("w", 0)), float(data.get("h", 0)))
                    elif data.get("type") == "barge_in":
                        self._barge_in_requested = True
                        logger.info("Barge-in requested from UE5")
                    elif data.get("type") == "shutdown":
                        # External graceful-shutdown trigger. Releases camera/mic
                        # via face_tracker.stop() in orch.stop()'s finally path.
                        # Used by stop-stack scripts to avoid TerminateProcess wedging
                        # the C920 in MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED.
                        logger.info("Shutdown requested from WS client")
                        self._running = False
                        break
            except websockets.ConnectionClosed:
                pass
            finally:
                self.ue_websockets.discard(ws)
                logger.info(f"UE5 disconnected from {remote}")

        server = await websockets.serve(handler, UE_WS_HOST, UE_WS_PORT)
        logger.info(f"WebSocket server listening on ws://{UE_WS_HOST}:{UE_WS_PORT}")

        try:
            while self._running:
                await asyncio.sleep(1)
        finally:                       # also when cancelled on shutdown
            server.close()
            await server.wait_closed()

    async def _send_to_ue(self, message: dict):
        """Send JSON message to all connected UE5 clients."""
        if message.get("type") == "emotion":
            try:
                self._last_emotion = (str(message.get("value") or "neutral"), float(message.get("intensity", 0.5)))
            except (TypeError, ValueError):
                pass
        if not self.ue_websockets:
            return
        payload = json.dumps(message)
        # Send to all connected clients. Iterate a snapshot — the WS server's
        # connection-close handler removes entries from self.ue_websockets in
        # a separate asyncio task, so iterating the live set races on PIE-stop.
        disconnected = set()
        for ws in list(self.ue_websockets):
            try:
                await ws.send(payload)
            except websockets.ConnectionClosed:
                disconnected.add(ws)
        self.ue_websockets -= disconnected

    async def _set_state(self, new_state: ConvState):
        """Transition avatar state and notify UE5."""
        if self.state != new_state:
            self.state = new_state
            await self._send_to_ue({"type": "state", "value": new_state.value})
            logger.debug(f"State → {new_state.value}")
        # Also refresh the lapel-pin indicator. Even when ConvState is
        # unchanged, face/auth/commit-hold context may have changed.
        await self._refresh_indicator()

    # ── UX Mode Indicator ───────────────────────────────────────────────────

    def _compute_indicator(self) -> str:
        """Derive current lapel-pin state from orchestrator state.

        Priority order:
          1. purple — nobody at the kiosk and a departed visitor's memory is
                      still being saved (replaces grey; never hides a visitor)
          2. grey   — no face has been at the kiosk recently
          3. white  — face present, not yet authenticated or anonymous
          4. red    — ConvState.SPEAKING (audio playing)
          5. amber  — ConvState.PROCESSING (LLM/TTS pipeline running)
          6. green  — listening (default once authenticated/anonymous)
        """
        if not (self.face_recognizer and self.face_recognizer.has_user):
            return "purple" if self._saves_pending else "grey"
        authenticated = self.current_user_id is not None
        if self.id_manager and self.id_manager.is_anonymous:
            authenticated = True
        if not authenticated:
            return "white"
        if self.state == ConvState.SPEAKING:
            return "red"
        if self.state == ConvState.PROCESSING:
            return "amber"
        return "green"

    async def _refresh_indicator(self):
        """Compute current indicator and emit to UE5 if changed."""
        value = self._compute_indicator()
        if value != self._current_indicator:
            self._current_indicator = value
            await self._send_to_ue({"type": "indicator", "value": value})
            logger.debug(f"Indicator → {value}")

    # ── Barge-in ──────────────────────────────────────────────────────────

    async def _handle_barge_in(self):
        """User interrupted NPC mid-speech — stop UE5 audio and let the mic
        loop finish accumulating the user's utterance into speech_buffer.

        Always-listening mode: the speech_buffer is already filling with the
        user's audio by the time this runs (the mic loop captures from the very
        first VAD-positive frame). speech_buffer and is_speech_active are NOT
        cleared — they carry forward into the user turn that this barge-in
        just triggered.
        """
        self._barge_in_event.set()

        try:
            await self._send_to_ue({"type": "stop"})
        except Exception as e:
            logger.warning(f"[BARGE-IN] Stop error: {e}")

        self._is_speaking = False
        self._barge_in_requested = True
        logger.info("[BARGE-IN] Stopped UE5 audio")

    # ── Microphone Capture + VAD Pipeline ───────────────────────────────────

    async def _run_mic_capture(self):
        """Capture mic audio, run VAD, trigger ASR on speech end."""
        import torch

        audio_queue = asyncio.Queue()

        def mic_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Mic status: {status}")
            audio_queue.put_nowait(indata[:, 0].copy())

        # Tell the Jabra firmware this is a speech-recognition stream so it
        # opens the mic during simultaneous playback. Measured:
        # eAudioCategorySpeech raises mic detection from ~30%/39% frame
        # crossings to 71%/81% in the first 2 s of overlap with TTS playback,
        # which is the difference between a missed barge-in and a fired one.
        # sounddevice's WasapiSettings doesn't expose streamCategory, so a
        # duck-typed PaWasapiStreamInfo is built via the CFFI bindings.
        from sounddevice import _ffi as _sd_ffi, _lib as _sd_lib
        _wasapi_speech_streaminfo = _sd_ffi.new('PaWasapiStreamInfo*', dict(
            size=_sd_ffi.sizeof('PaWasapiStreamInfo'),
            hostApiType=_sd_lib.paWASAPI,
            version=1,
            # AutoConvert lets WASAPI shared mode resample its 48 kHz mix format to
            # the 16 kHz request — without it Pa_OpenStream fails with -9997
            # "Invalid sample rate".
            flags=_sd_lib.paWinWasapiAutoConvert,
            streamCategory=_sd_lib.eAudioCategorySpeech,
        ))
        class _SpeechSettings:
            pass
        _speech_settings = _SpeechSettings()
        _speech_settings._streaminfo = _wasapi_speech_streaminfo

        # Only attach the WASAPI-specific settings when the resolved device is
        # actually a WASAPI instance — on any other host API PortAudio rejects the
        # stream info with -9984 and the orchestrator crashes.
        _use_speech_category = False
        try:
            _dev = sd.query_devices(MIC_DEVICE) if MIC_DEVICE is not None else sd.query_devices(kind="input")
            _use_speech_category = "WASAPI" in sd.query_hostapis()[_dev["hostapi"]]["name"]
        except Exception as _e:
            logger.warning(f"Could not determine mic host API ({_e}); skipping streamCategory=Speech")
        if not _use_speech_category:
            logger.warning("Mic is not on WASAPI — streamCategory=Speech disabled "
                           "(barge-in detection during playback will be degraded)")
        logger.info(f"Opening mic stream (streamCategory={'Speech' if _use_speech_category else 'default'})...")
        stream = sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=MIC_CHANNELS,
            blocksize=MIC_BLOCKSIZE,
            dtype="float32",
            device=MIC_DEVICE,
            callback=mic_callback,
            extra_settings=_speech_settings if _use_speech_category else None,
        )
        stream.start()
        logger.info("Mic stream active")

        try:
            while self._running:
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                self.ring_buffer.append((time.time(), chunk))

                # High-pass filter — removes fan hum, HVAC, electrical buzz
                chunk_filtered = sosfilt(self._hp_sos, chunk).astype(np.float32)
                chunk_tensor = torch.from_numpy(chunk_filtered).float()
                # no_grad is REQUIRED, not an optimization. Silero VAD is a
                # stateful RNN whose internal hidden state persists on the model
                # across calls. Without no_grad, every 32ms-frame forward pass
                # chains a fresh autograd-graph node onto that retained state, so
                # ONE graph grows across ~2.7M calls/day and never frees. Those
                # are torch c10 native tensors (invisible to tracemalloc),
                # allocated-once-never-read → paged out cold. Measured without it:
                # orchestrator private bytes reach 82 GB → system swap pressure →
                # UE5 D3D device removed (a GPU TDR crash), while a 24 h tracemalloc
                # diff shows a FLAT Python heap (-0.2 MB) against the native growth.
                with torch.no_grad():
                    speech_prob = self.vad_model(chunk_tensor, MIC_SAMPLE_RATE).item()
                self._last_vad_prob = speech_prob

                # Barge-in confidence gate: while SHE is speaking, only
                # HIGH-confidence frames (>= BARGE_IN_VAD_THRESHOLD_PLAYBACK) count
                # toward a barge-in, so AEC leak in the 0.3-0.5 band can't self-
                # interrupt her. Any weak / non-playback frame resets the run.
                if self._is_speaking and speech_prob >= BARGE_IN_VAD_THRESHOLD_PLAYBACK:
                    self._bargein_strong_frames += 1
                else:
                    self._bargein_strong_frames = 0

                # Always-listening single VAD pipeline. Speech detection below
                # always uses the base 0.3 threshold (VAD_THRESHOLD).
                if speech_prob >= VAD_THRESHOLD:
                    if not self.is_speech_active:
                        self.is_speech_active = True
                        self.speech_buffer.clear()
                        self.silence_frames = 0
                        self._vad_speech_start = time.time()
                        # Capture _is_speaking state at speech-start: True ⇒
                        # this utterance is interrupting her (a barge-in).
                        self._was_speaking_at_speech_start = self._is_speaking
                        logger.debug(
                            f"Speech started (was_speaking={self._was_speaking_at_speech_start})"
                        )

                    self.speech_buffer.append(chunk)
                    self.silence_frames = 0
                    # a small listening nod now and then while the visitor talks (gesture_planner.py)
                    if (not self._is_speaking and self._visitor_present()
                            and self._gestures.listen_due(time.time(), self._vad_speech_start)):
                        asyncio.create_task(self._send_gesture("listen", "listening", 0.3))
                else:
                    if self.is_speech_active:
                        self.speech_buffer.append(chunk)  # keep trailing audio
                        self.silence_frames += 1

                        silence_ms = (self.silence_frames * MIC_BLOCKSIZE / MIC_SAMPLE_RATE) * 1000

                        if silence_ms >= SILENCE_CONFIRMED_MS and len(self.speech_buffer) > 3:
                            audio_data = np.concatenate(self.speech_buffer)
                            vad_start = self._vad_speech_start
                            vad_endpoint = time.time()
                            self.is_speech_active = False
                            self.speech_buffer.clear()
                            self.silence_frames = 0
                            self._was_speaking_at_speech_start = False

                            # Process in background task
                            asyncio.create_task(self._process_utterance(audio_data, vad_start, vad_endpoint))

                            # Reset Silero VAD's internal RNN state so the next
                            # utterance starts clean — stale state from the prior
                            # utterance degrades detection accuracy over time. (The
                            # _vad_reset flag set in start() is not used.)
                            self.vad_model.reset_states()

                # Barge-in trigger — runs every frame while speech is active.
                # Soft speech that intermittently dips below VAD_THRESHOLD still
                # accumulates in speech_buffer (via the else-branch append), so
                # checking here (not just inside the high-VAD branch)
                # ensures barge-in fires whenever sustained speech has been captured
                # during NPC playback. _barge_in_event prevents double-fire.
                if (self.is_speech_active
                        and self._was_speaking_at_speech_start
                        and self._bargein_strong_frames >= BARGE_IN_TRIGGER_FRAMES_PLAYBACK
                        and (time.time() - self._speaking_started_at) >= BARGE_IN_GRACE_S
                        and not self._barge_in_event.is_set()):
                    since_ms = (time.time() - self._speaking_started_at) * 1000
                    logger.warning(
                        f"[BARGE-IN] Triggered: {self._bargein_strong_frames} strong frames "
                        f"(>={BARGE_IN_VAD_THRESHOLD_PLAYBACK}), last_vad={self._last_vad_prob:.2f}, "
                        f"{since_ms:.0f}ms into her turn"
                    )
                    await self._handle_barge_in()
        finally:
            stream.stop()
            stream.close()

    # ── Text Cleaning ─────────────────────────────────────────────────────

    def _clean_for_tts(self, text: str) -> str:
        """Strip directives and markdown from text for TTS."""
        clean, directives = strip_directives(text)
        clean = re.sub(r'\*+', '', clean)
        clean = re.sub(r'#+\s*', '', clean)
        clean = re.sub(r'\[/?[A-Z_]+\]', '', clean)
        # Dashes: TTS guesses where to pause on them ("whenever you are—do you
        # want" can come out as "whenever you... are do you want"). Number ranges
        # read as "to"; every other dash becomes a comma, a pause it handles.
        clean = re.sub(r'(\d)\s*[–—]\s*(\d)', r'\1 to \2', clean)
        clean = re.sub(r'\s*[–—]\s*|\s+-\s+', ', ', clean)
        clean = re.sub(r'\n{2,}', ' ', clean)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        return clean

    # ── Sentence Boundary Detection ─────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on . ! ? boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    # ── Play One Sentence (TTS + lip sync + speaker) ────────────────────────

    # -- Gestures (gesture_planner.py decides when and which intent; Unreal picks the snippet) --

    def _visitor_present(self) -> bool:
        """Someone at the kiosk: face recognition in Kiosk mode, face presence in Companion mode
        (both expose has_user)."""
        if self.face_recognizer is None:
            return True
        return bool(self.face_recognizer.has_user)

    async def _send_gesture(self, intent: str, role: str, intensity: float | None = None):
        emotion, emo_intensity = self._last_emotion
        await self._send_to_ue({"type": "gesture", "intent": intent, "role": role, "emotion": emotion,
                                "intensity": round(emo_intensity if intensity is None else intensity, 2)})

    def _start_gesture_plan(self, text: str, duration: float, started_at: float):
        """Plan this chunk's gestures (her tags + word cues) and send them on a timer while it plays."""
        tags, self._pending_gesture_tags = self._pending_gesture_tags, []
        mood = self.mood_tracker.mood if self.mood_tracker else None
        plan = self._gestures.plan_speech(text, duration, tags, mood, started_at)
        if plan:
            asyncio.create_task(self._run_gesture_plan(plan, started_at, mood))

    async def _run_gesture_plan(self, plan: list, started_at: float, mood: str | None):
        try:
            for offset, intent in plan:
                delay = started_at + offset - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                if self._barge_in_event.is_set() or not self._is_speaking:
                    return
                emotion, intensity = self._last_emotion
                if (mood or "").upper() == "GENTLE":
                    intensity *= 0.8
                await self._send_gesture(intent, "speaking", intensity)
                logger.info(f"[GESTURE] {intent} at +{offset:.1f}s ({emotion} {intensity:.2f})")
        except Exception as e:
            logger.warning(f"[GESTURE] plan stopped: {e}")

    async def _run_idle_gestures(self):
        """Nobody at the kiosk: an occasional look-around (gesture_planner.idle_due)."""
        while self._running:
            await asyncio.sleep(5.0)
            if not self._running:          # stopped during the sleep: don't gesture on the way out
                break
            try:
                now = time.time()
                if self._visitor_present() or self._is_speaking or self.is_speech_active:
                    self._idle_since = now
                    continue
                if self._gestures.idle_due(now, self._idle_since):
                    await self._send_gesture("idle", "idle", 0.3)
                    logger.info("[GESTURE] idle look-around")
            except Exception as e:
                logger.warning(f"[GESTURE] idle loop: {e}")

    async def _play_sentence(self, sentence_text: str, emotion_sync: EmotionSynchronizer = None, force: bool = False):
        """TTS a single sentence, send to UE5 for playback + lip sync.

        Can be interrupted by barge-in (self._barge_in_event).
        force=True bypasses the barge-in and identity-flip-preempt gates;
        used by _speak_id_line so its own playback is not blocked by the same flag
        it sets to silence the utterance task.
        """
        import base64

        # Check barge-in OR identity-flip preempt before starting
        if not force and (self._barge_in_event.is_set() or self._suppress_utterance_playback):
            return

        tts_result = await self._call_tts(sentence_text)
        if not tts_result or (not force and (self._barge_in_event.is_set() or self._suppress_utterance_playback)):
            return

        await self._play_audio(sentence_text, tts_result, emotion_sync, force=force)

    async def _play_audio(self, text: str, tts_result: dict,
                          emotion_sync: EmotionSynchronizer = None,
                          force: bool = False) -> float:
        """Play an already-synthesized TTS result through UE5.

        Separate from _play_sentence so the two-chunk pipeline can prefetch
        synthesis (chunk 2 synthesizes WHILE chunk 1 plays) and hand finished
        audio here. Returns elapsed playback seconds at return time (== full
        duration on normal completion; less when barge-in cut it), so the
        caller can compute what was actually heard via _spoken_prefix.
        """
        import base64

        if not force and (self._barge_in_event.is_set() or self._suppress_utterance_playback):
            return 0.0

        sentence_text = text  # the body below reads sentence_text
        sr = tts_result.get("sample_rate", 24000)
        audio_b64 = tts_result["audio_base64"]
        audio_bytes = base64.b64decode(audio_b64)
        audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio_float) / sr

        logger.info(f"Playing sentence ({duration:.1f}s): {sentence_text[:50]}...")

        # Apply volume gain and write WAV file
        boosted = np.clip(audio_float * TTS_VOLUME_GAIN, -1.0, 1.0)
        boosted_bytes = (boosted * 32767).astype(np.int16).tobytes()
        wav_path = os.path.join(tempfile.gettempdir(), f"kiosk_speech_{int(time.time()*1000)}.wav")
        self._pending_wav_unlinks.add(wav_path)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(boosted_bytes)

        # Enter speaking mode (mic stays live and always-listening pipeline
        # detects barge-in if the user interrupts).
        if not self._is_speaking:
            self._speaking_started_at = time.time()   # turn-level grace timer
        self._is_speaking = True
        self._barge_in_event.clear()

        # Send WAV path to UE5 — triggers lip sync plugin via BP_Avatar → ABP_AvatarFace
        await self._send_to_ue({"type": "speak", "path": wav_path})
        t_playback_started = time.time()
        self._start_gesture_plan(sentence_text, duration, t_playback_started)

        # Send emotion directive if available (overlaps playback)
        if emotion_sync:
            a2e_task = asyncio.create_task(self._call_a2e(audio_b64, sr))
            a2e_emotion, a2e_confidence = "neutral", 0.0
            try:
                a2e_result = await asyncio.wait_for(a2e_task, timeout=1.0)
                if a2e_result:
                    a2e_emotion = a2e_result.get("emotion", "neutral")
                    a2e_confidence = a2e_result.get("confidence", 0.0)
            except (asyncio.TimeoutError, Exception):
                pass
            if a2e_emotion != "neutral":
                logger.info(f"A2E: {a2e_emotion}/{a2e_confidence:.1f}")

        # Wait for audio duration — interruptible by barge-in
        # Wait for the REMAINDER of the audio, measured from when playback
        # actually started - not the full duration after the A2E call (that
        # would add up to 1s of dead air per chunk: the A2E overshoot).
        remaining = duration - (time.time() - t_playback_started)
        try:
            if remaining > 0:
                await asyncio.wait_for(self._barge_in_event.wait(), timeout=remaining)
                # wait_for returned without timing out => barge-in event set
                logger.info("[BARGE-IN] Playback interrupted")
                return time.time() - t_playback_started
            if self._barge_in_event.is_set():
                logger.info("[BARGE-IN] Playback interrupted")
                return time.time() - t_playback_started
        except asyncio.TimeoutError:
            # Normal completion — no barge-in
            pass

        # Done speaking
        self._is_speaking = False
        return duration

    async def _seam_pause(self, t_chunk1_end: float):
        """Hold chunk 2 until CHUNK_SEAM_PAUSE_S after chunk 1 ended.

        Chunk 1 and the merged remainder are synthesized separately, so the
        pause the TTS puts between sentences is lost at this one seam and
        sentence 1 runs straight into sentence 2. Only the shortfall is
        waited: when chunk-2 synthesis is itself slow, the gap already
        exists. Returns early on barge-in; the caller re-checks the event.
        """
        gap = time.time() - t_chunk1_end
        hold = CHUNK_SEAM_PAUSE_S - gap
        if hold <= 0:
            logger.info(f"[SEAM] natural gap {gap:.2f}s (chunk-2 synthesis was the wait)")
            return
        logger.info(f"[SEAM] holding {hold:.2f}s for a sentence pause")
        try:
            await asyncio.wait_for(self._barge_in_event.wait(), timeout=hold)
        except asyncio.TimeoutError:
            pass

    @staticmethod
    def _spoken_prefix(text: str, word_timestamps, elapsed_s: float, duration_s: float) -> str:
        """Best-effort 'what the user actually heard' prefix of a TTS chunk,
        cut at the last word that started before playback stopped. Used for
        barge-in history on merged multi-sentence chunks."""
        if elapsed_s >= duration_s - 0.25:
            return text
        try:
            heard = [w for w in (word_timestamps or [])
                     if float(w.get("start", 1e9)) <= elapsed_s]
            if heard:
                return " ".join(str(w.get("word", "")).strip() for w in heard).strip()
        except Exception:
            pass
        # Fallback: proportional word count
        toks = text.split()
        n = max(0, int(len(toks) * (elapsed_s / max(duration_s, 0.01))))
        return " ".join(toks[:n])

    # ── Utterance Processing Pipeline (streaming) ───────────────────────────

    async def _process_utterance(self, audio: np.ndarray,
                                  vad_start: float = 0.0, vad_endpoint: float = 0.0):
        """Streaming pipeline: ASR → context → stream LLM → per-sentence TTS+play."""
        # Serialize — skip if a previous utterance is still being processed
        if self._llm_lock.locked():
            logger.debug("Skipping utterance — previous still processing")
            return
        # Someone is reading names aloud for the /admin Names page: not for her.
        if listening_paused():
            logger.info("[NAMES] Listening paused by the Names page - utterance ignored")
            return
        async with self._llm_lock:
            await self._process_utterance_inner(audio, vad_start, vad_endpoint)

    async def _process_utterance_inner(self, audio: np.ndarray,
                                        vad_start: float = 0.0, vad_endpoint: float = 0.0):
        """Inner implementation of utterance processing (serialized by _llm_lock)."""
        # Reset barge-in state for this turn
        self._barge_in_event.clear()
        self._barge_in_requested = False

        # Face-presence gate: drop ambient utterances when no face has been
        # at the kiosk recently. Saves ASR + LLM cycles + prevents history
        # pollution from background room audio (meetings, podcasts, calls).
        face_age = time.time() - self._last_face_seen_at
        if face_age > FACE_GATE_S:
            logger.info(f"[FACE-GATE] Dropping utterance — no face for {face_age:.1f}s (threshold {FACE_GATE_S:.0f}s)")
            return

        # ConvState stays LISTENING during ASR — the "Listening" lapel
        # state includes STT running. Transition to PROCESSING below,
        # once the LLM/TTS pipeline actually starts.
        t0 = time.time()

        # ASR and user emotion detection in parallel
        asr_task = asyncio.create_task(self._run_asr(audio))
        emotion_task = asyncio.create_task(self._run_emotion_detection(audio))

        transcript = await asr_task
        if not transcript or len(transcript.strip()) < 2:
            emotion_task.cancel()
            # No need to set LISTENING — never left it
            return

        t_asr = time.time()
        logger.info(f"ASR ({(t_asr - t0)*1000:.0f}ms): {transcript}")

        # Now switch to PROCESSING (amber on the lapel pin): LLM/TTS pipeline begins.
        await self._set_state(ConvState.PROCESSING)

        # Drain previous turn's TTS WAV files now that the user has spoken again
        # (UE5 has finished playing those clips).
        for stale_path in list(self._pending_wav_unlinks):
            try:
                os.unlink(stale_path)
            except OSError:
                pass
            self._pending_wav_unlinks.discard(stale_path)

        # Identification intercept — handle name/PIN collection before LLM
        if self.id_manager and self.id_manager.is_identifying:
            emotion_task.cancel()
            consumed = await self._handle_id_utterance(transcript)
            if consumed:
                await self._set_state(ConvState.LISTENING)
                return

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": transcript})

        # Get user emotion (wait up to 1s, then fall back to neutral)
        try:
            user_emotion = await asyncio.wait_for(emotion_task, timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            user_emotion = "neutral"
        logger.info(f"User emotion: {user_emotion}")

        # Feed voice emotion into mood tracker
        if self.mood_tracker:
            self.mood_tracker.update_voice_emotion(user_emotion, self.current_user_emotion_prob)

        self.emotion_trajectory.append({
            "turn": len(self.conversation_history),
            "time": time.time() - self.session_start,
            "emotion": user_emotion,
        })

        # Assemble context
        messages = self._assemble_context()

        # Stream LLM tokens, accumulate sentences, TTS+play each as it completes
        full_response = ""
        pending_text = ""
        spoken_sentences = []  # tracks sentences actually sent to UE5 (for barge-in history)
        emotion_sync = EmotionSynchronizer()

        # Two-chunk playback: sentence 1 starts playing the moment it completes
        # (no added TTFA) as a BACKGROUND task so token consumption never
        # blocks on playback; every later sentence accumulates and is
        # synthesized as ONE merged chunk while chunk 1 plays. Result: one
        # natural seam per reply instead of dead air at every sentence
        # boundary (serial synthesis adds 0.7-1.5s between sentences).
        first_chunk_task = None   # asyncio.Task playing sentence 1
        first_chunk_text = None
        remainder_parts = []      # cleaned sentences awaiting the merged chunk

        # Telemetry timing
        t_llm_first_token = None
        t_llm_first_sentence = None
        t_first_speak = None
        t_llm_request_sent = None  # when the LLM request is sent
        llm_ttft_ms_server = 0    # from the server's done event
        llm_tokens = 0            # from the server's done event
        llm_tps = 0               # from the server's done event
        llm_total_ms_computed = 0  # derived from tokens/tps
        llm_token_count_local = 0 # counted locally as fallback

        try:
            t_llm_request_sent = time.time()
            async with self.http_client.stream(
                "POST",
                "/llm/generate",
                json={
                    "messages": messages,
                    "session_id": self.session_id,
                    "stream": True,
                    "max_tokens": 256,
                    "temperature": self.mood_tracker.temperature if self.mood_tracker else 0.7,
                },
                timeout=60.0,
            ) as response:
                response.raise_for_status()
                first_token_time = None

                async for line in response.aiter_lines():
                    # Check for barge-in during LLM streaming
                    if self._barge_in_event.is_set():
                        logger.info("[BARGE-IN] Interrupting LLM stream")
                        break

                    if not line or not line.startswith("data: "):
                        continue
                    token_data = line[6:]
                    if token_data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(token_data)
                        # Gateway format: {"token": "...", "done": false}
                        token = chunk.get("token", "")
                        if chunk.get("done", False):
                            # Capture telemetry from the server's done event
                            llm_ttft_ms_server = chunk.get("ttft_ms", 0)
                            llm_tokens = chunk.get("tokens_generated", 0)
                            llm_tps = chunk.get("tps", 0)
                            # Compute total LLM time from server-side metrics (not local
                            # timestamps, which are delayed by sentence playback blocking)
                            llm_total_ms_computed = (llm_tokens / llm_tps * 1000) if llm_tps > 0 else 0
                            logger.info(f"[TELEMETRY] LLM done: ttft={llm_ttft_ms_server}ms "
                                        f"tokens={llm_tokens} tps={llm_tps:.1f} "
                                        f"total={llm_total_ms_computed:.0f}ms")
                            break  # done event is the last — break, don't continue
                        if not token:
                            continue
                    except json.JSONDecodeError:
                        continue

                    if first_token_time is None:
                        first_token_time = time.time()
                        t_llm_first_token = first_token_time
                        logger.info(f"LLM TTFT: {(first_token_time - t_asr)*1000:.0f}ms")

                    full_response += token
                    pending_text += token
                    llm_token_count_local += 1

                    # Extract and forward directives immediately
                    directive_matches = list(DIRECTIVE_RE.finditer(pending_text))
                    if directive_matches:
                        logger.info(f"[DIRECTIVE] Found {len(directive_matches)} directive(s)")
                        for m in directive_matches:
                            name = m.group(1)
                            value = m.group(2) or ""
                            intensity = float(m.group(3)) if m.group(3) else 0.5
                            logger.info(f"[DIRECTIVE] {name}={value} intensity={intensity}")
                            if name == "emotion":
                                value, intensity = self._blend_emotion(value, intensity)
                                logger.info(f"[EMOTION→UE5] Sending mood: {value} intensity={intensity:.2f}")
                                await self._send_to_ue({"type": "emotion", "value": value, "intensity": intensity})
                                emotion_sync.add_directive("emotion", value, intensity, m.start())
                            elif name == "gesture":
                                self._pending_gesture_tags.append(value)   # played with its sentence
                                emotion_sync.add_directive("gesture", value, char_position=m.start())
                        # Strip processed directives so they aren't re-matched on next token
                        pending_text = DIRECTIVE_RE.sub('', pending_text)

                    # Skip sentence boundary check if a directive is still being streamed
                    # (unclosed << without matching >>)
                    open_count = pending_text.count("<<")
                    close_count = pending_text.count(">>")
                    if open_count > close_count:
                        continue  # directive still incomplete, wait for more tokens

                    # Strip all complete directives, then check for sentence boundary
                    text_for_split = DIRECTIVE_RE.sub('', pending_text)
                    # Also strip any remaining << >> fragments
                    text_for_split = re.sub(r'<<[^>]*>>', '', text_for_split)
                    text_for_split = text_for_split.strip()
                    if not text_for_split:
                        continue
                    if re.search(r'[.!?]\s', text_for_split) or re.search(r'[.!?]$', text_for_split.rstrip()):
                        # Track first sentence boundary for telemetry
                        if t_llm_first_sentence is None:
                            t_llm_first_sentence = time.time()

                        # Split the CLEANED text into sentences (avoids a false split at ".8")
                        clean_sentences = self._split_sentences(text_for_split)
                        if len(clean_sentences) > 1:
                            # Dispatch/accumulate all complete sentences
                            for sent in clean_sentences[:-1]:
                                if self._barge_in_event.is_set():
                                    break
                                clean = self._clean_for_tts(sent)
                                if clean and len(clean) > 3:
                                    if first_chunk_task is None:
                                        if self.state != ConvState.SPEAKING:
                                            await self._set_state(ConvState.SPEAKING)
                                        if t_first_speak is None:
                                            t_first_speak = time.time()
                                        first_chunk_text = clean
                                        first_chunk_task = asyncio.create_task(
                                            self._play_sentence(clean, emotion_sync))
                                    else:
                                        remainder_parts.append(clean)
                            # Keep only the raw text after the last sentence boundary
                            # Find where the last complete sentence ends in the original
                            last_clean_sent = clean_sentences[-1]
                            # Reset pending to the remaining cleaned fragment
                            pending_text = last_clean_sent
                        elif re.search(r'[.!?]$', text_for_split.rstrip()):
                            # Single complete sentence
                            clean = self._clean_for_tts(text_for_split)
                            if clean and len(clean) > 3:
                                if first_chunk_task is None:
                                    if self.state != ConvState.SPEAKING:
                                        await self._set_state(ConvState.SPEAKING)
                                    if t_first_speak is None:
                                        t_first_speak = time.time()
                                    first_chunk_text = clean
                                    first_chunk_task = asyncio.create_task(
                                        self._play_sentence(clean, emotion_sync))
                                else:
                                    remainder_parts.append(clean)
                            pending_text = ""

        except httpx.HTTPError as e:
            logger.error(f"LLM streaming failed: {e}")
            await self._set_state(ConvState.LISTENING)
            return

        # Tail fragment joins the remainder (or becomes chunk 1 when the whole
        # reply had no mid-stream sentence boundary).
        if pending_text.strip() and not self._barge_in_event.is_set():
            clean = self._clean_for_tts(pending_text)
            if clean and len(clean) > 3:
                if first_chunk_task is None:
                    if self.state != ConvState.SPEAKING:
                        await self._set_state(ConvState.SPEAKING)
                    if t_first_speak is None:
                        t_first_speak = time.time()
                    first_chunk_text = clean
                    first_chunk_task = asyncio.create_task(
                        self._play_sentence(clean, emotion_sync))
                else:
                    remainder_parts.append(clean)

        # Chunk 2: synthesize the merged remainder WHILE chunk 1 plays, then
        # play it after a CHUNK_SEAM_PAUSE_S sentence pause (see _seam_pause).
        remainder_text = " ".join(remainder_parts).strip()
        remainder_tts_task = None
        if remainder_text and first_chunk_task is not None and not self._barge_in_event.is_set():
            remainder_tts_task = asyncio.create_task(
                self._call_tts(remainder_text, record_telemetry=False))

        if first_chunk_task is not None:
            try:
                await first_chunk_task
            except Exception as e:
                logger.warning(f"[CHUNK1] playback task error: {e!r}")
            if first_chunk_text and not self._barge_in_event.is_set():
                spoken_sentences.append(first_chunk_text)
        t_chunk1_end = time.time()
        if remainder_tts_task is not None and not self._barge_in_event.is_set():
            # Still mid-reply until chunk 2 plays, so speech in the gap (synth
            # wait or seam pause) is a barge-in, not a new turn she talks over.
            self._is_speaking = True

        if remainder_tts_task is not None:
            if self._barge_in_event.is_set():
                remainder_tts_task.cancel()
                try:
                    await remainder_tts_task
                except Exception:
                    pass
            else:
                chunk2 = None
                try:
                    chunk2 = await remainder_tts_task
                except Exception as e:
                    logger.warning(f"[CHUNK2] synth error: {e!r}")
                if chunk2:
                    await self._seam_pause(t_chunk1_end)
                if chunk2 and not self._barge_in_event.is_set():
                    import base64 as _b64
                    dur2 = len(_b64.b64decode(chunk2["audio_base64"])) / 2 / chunk2.get("sample_rate", 24000)
                    elapsed2 = await self._play_audio(remainder_text, chunk2, emotion_sync)
                    if not self._barge_in_event.is_set():
                        spoken_sentences.extend(self._split_sentences(remainder_text))
                    else:
                        # Barged mid-chunk-2: record only what was actually
                        # heard, cut at the word timestamps.
                        heard = self._spoken_prefix(
                            remainder_text, chunk2.get("word_timestamps"),
                            elapsed2, dur2)
                        if heard:
                            spoken_sentences.append(heard)

        # Handle conversation history
        if self._barge_in_event.is_set():
            # Barge-in: PRESERVE the exchange. Keep the user turn and save what
            # she actually finished speaking, falling back to the full generated
            # turn, so user/assistant turns keep alternating and she can recall
            # what she said. No [interrupted] marker — it causes the LLM to
            # apologize for being interrupted.
            #
            # If nothing was generated at all, drop the user turn rather than
            # writing a "..." placeholder: a "..." assistant turn primes the
            # model to emit "..." on subsequent turns. Dropping the user turn
            # keeps history clean and matches the empty-response path below.
            spoken = " ".join(spoken_sentences).strip() if spoken_sentences else full_response.strip()
            if spoken:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": spoken,
                })
                logger.info(f"[BARGE-IN] Kept turn in history "
                            f"({'spoken' if spoken_sentences else 'full-gen'}): {spoken[:80]}...")
            elif self.conversation_history and self.conversation_history[-1].get("role") == "user":
                dropped = self.conversation_history.pop()
                logger.warning(f"[BARGE-IN] No content generated; dropped user turn: {dropped['content'][:60]}...")
        else:
            # Normal completion: save full response to history
            if full_response.strip():
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response,
                })
            else:
                # Empty response — don't add to history (would break alternation).
                # Also remove the user message that triggered it.
                if self.conversation_history and self.conversation_history[-1].get("role") == "user":
                    dropped = self.conversation_history.pop()
                    logger.warning(f"[HISTORY] Dropping orphaned user message: {dropped['content'][:60]}...")

        # Single-user mode: the finished exchange goes to the pending file at
        # once, so nothing is lost before it is committed to memory.
        if self.identity.single_user:
            self._single_user_record_exchange()

        # Debug: log raw LLM output
        logger.info(f"[LLM RAW] {full_response[:200]}")

        # Session recovery (safety net): if LLM returned empty and it wasn't a barge-in,
        # something unexpected happened. Log loudly, clear history, and reset session.
        if not full_response.strip() and not self._barge_in_event.is_set():
            logger.error(f"[SESSION] Empty LLM response — auto-recovering. "
                         f"History had {len(self.conversation_history)} entries.")
            try:
                await self.http_client.post("/llm/end_session", json={"session_id": self.session_id})
            except Exception:
                pass
            self.conversation_history.clear()
            # (single-user keeps its never-expiring session prefix)
            self.session_id = (SESSION_PREFIX if self.identity.single_user else "") + str(uuid.uuid4())
            logger.info(f"[SESSION] New session: {self.session_id}, history cleared")

        total_ms = (time.time() - t0) * 1000
        logger.info(f"Pipeline total: {total_ms:.0f}ms")

        # Telemetry: POST timing data to FastAPI dashboard (fire-and-forget).
        # All times in ms. The row's t=0 anchor is `vad_endpoint` (silence
        # threshold met).
        if full_response.strip():
            speech_duration_ms = ((vad_endpoint - vad_start) * 1000) if vad_start > 0 else 0
            # VAD bar = orchestrator-side dispatch latency between
            # vad_endpoint and start of asyncio _process_utterance_inner work.
            vad_dispatch_ms = ((t0 - vad_endpoint) * 1000) if vad_endpoint > 0 else 0
            asr_processing_ms = (t_asr - t0) * 1000
            llm_first_sentence_ms = ((t_llm_first_sentence - t_llm_first_token) * 1000
                                     if t_llm_first_token and t_llm_first_sentence else 0)

            # LLM TTFT: prefer server-side metric, fall back to local measurement
            if llm_ttft_ms_server > 0:
                llm_ttft_ms = llm_ttft_ms_server
            elif t_llm_first_token and t_llm_request_sent:
                llm_ttft_ms = (t_llm_first_token - t_llm_request_sent) * 1000
            else:
                llm_ttft_ms = 0

            # llm_client_ms = orchestrator-side overhead between sending the
            # /llm/generate POST and receiving the first SSE token, beyond
            # what llama-server reports as TTFT.
            if t_llm_first_token and t_llm_request_sent and llm_ttft_ms_server > 0:
                llm_client_ms = (t_llm_first_token - t_llm_request_sent) * 1000 - llm_ttft_ms_server
            else:
                llm_client_ms = 0
            llm_client_ms = max(0, llm_client_ms)

            # Token count: prefer the server's count, fall back to local count
            final_token_count = llm_tokens if llm_tokens > 0 else llm_token_count_local

            # LLM total: prefer server-side (not delayed by playback), fall back to local
            if llm_tps > 0:
                llm_total_ms = llm_total_ms_computed
            elif t_llm_first_token and final_token_count > 0:
                llm_total_ms = llm_ttft_ms + (final_token_count * 25)
            else:
                llm_total_ms = 0

            tts_ms = self._last_tts_ms

            # llm_tps fallback when gateway done event is not received.
            llm_tps_final = llm_tps
            if llm_tps_final <= 0 and final_token_count > 0 and llm_total_ms > 0:
                llm_tps_final = (final_token_count / llm_total_ms) * 1000

            # TTFA = vad_endpoint → first speak WS message sent.
            # Approximation: doesn't include UE5-side audio load + playback start.
            ttfa_ms = ((t_first_speak - vad_endpoint) * 1000) if (t_first_speak and vad_endpoint > 0) else 0

            # total_pipeline_ms = vad_endpoint → last sentence finished playing.
            total_pipeline_ms = ((time.time() - vad_endpoint) * 1000) if vad_endpoint > 0 else total_ms
            utterance_tail_ms = max(0, total_pipeline_ms - ttfa_ms)

            # Sum named stages and flag any unaccounted gap (instrumentation drift).
            named_sum = (vad_dispatch_ms + asr_processing_ms + llm_client_ms
                         + llm_ttft_ms + llm_first_sentence_ms + tts_ms)
            unaccounted_ms = max(0, ttfa_ms - named_sum)

            # Multimodal context snapshot for the row's Type-C strip.
            ctx = {}
            if self.current_user_name:
                ctx["id"] = self.current_user_name
                ctx["id_sim"] = round(self.current_user_similarity, 3)
            elif self.face_recognizer and self.face_recognizer.has_user:
                # Fallback for active session not yet PIN-confirmed (returning user awaiting PIN)
                ctx["id"] = self.face_recognizer.user_name or "user"
                ctx["id_sim"] = round(self.current_user_similarity, 3)
            ctx["face_emo"] = self.current_face_emotion           # raw HSE label
            ctx["voice_emo"] = self.current_user_emotion          # raw e2v label
            if self.mood_tracker:
                ctx["mood_to"] = self.mood_tracker._current_mood
            # avatar_emo + avatar_int are not populated yet

            turn_timings = {
                "turn_id": f"turn-{int(t0)}",
                "session_id": self.session_id,
                "timestamp": t0,
                "transcript": transcript,
                "response": full_response[:500],
                "ctx": ctx,
                "timings": {
                    "speech_duration_ms": round(speech_duration_ms),
                    "vad_dispatch_ms": round(vad_dispatch_ms),
                    "asr_processing_ms": round(asr_processing_ms),
                    "llm_client_ms": round(llm_client_ms),
                    "llm_ttft_ms": round(llm_ttft_ms),
                    "llm_first_sentence_ms": round(llm_first_sentence_ms),
                    "llm_total_ms": round(llm_total_ms),
                    "llm_tokens": final_token_count,
                    "llm_tps": round(llm_tps_final, 1) if llm_tps_final > 0 else 0,
                    "tts_ms": round(tts_ms),
                    "ws_speak_send_ms": 0,  # localhost WS — sub-ms, not separately measured yet
                    "ttfa_ms": round(ttfa_ms),
                    "total_pipeline_ms": round(total_pipeline_ms),
                    "utterance_tail_ms": round(utterance_tail_ms),
                    "unaccounted_ms": round(unaccounted_ms),
                },
            }
            asyncio.create_task(self._post_telemetry(turn_timings))

        # No per-turn fact extraction: memory is extracted at session end
        # only; see _end_memory_session and FastAPI's deferred-compress on
        # /memory/{user}/session_end. Per-turn extraction would use the same
        # llama-server slot 0 as the conversation, evicting the KV cache on
        # every turn.

        # Ensure clean state
        self._is_speaking = False
        await self._set_state(ConvState.LISTENING)

    async def _post_telemetry(self, data: dict):
        """Fire-and-forget telemetry POST to FastAPI dashboard."""
        try:
            await self.http_client.post("/telemetry/turn", json=data, timeout=5.0)
        except Exception as e:
            logger.debug(f"Telemetry POST failed (non-blocking): {e}")

    def _emit_event(self, component: str, kind: str, text: str = "", **data):
        """Fire-and-forget identity / multimodal event POST for dashboard
        timeline. Schedules the HTTP call as a background task — caller
        does not need to await."""
        body = {
            "session_id": self.session_id,
            "component": component,
            "kind": kind,
            "text": text,
            "data": data,
        }
        async def _send():
            try:
                await self.http_client.post("/telemetry/event", json=body, timeout=2.0)
            except Exception as e:
                logger.debug(f"Telemetry event POST failed (non-blocking): {e}")
        try:
            asyncio.create_task(_send())
        except RuntimeError:
            pass  # no running event loop (shouldn't happen but safe)

    # ── Emotion Blending ──────────────────────────────────────────────────

    # Map HSEmotion labels (8) to the avatar's directive labels (12)
    _FACE_TO_DIRECTIVE = {
        "anger": "angry", "contempt": "disgust", "disgust": "disgust",
        "fear": "fearful", "happiness": "happy", "neutral": "neutral",
        "sadness": "sad", "surprise": "surprised",
    }

    # Intensity scale factor — LLM tends to overshoot
    INTENSITY_SCALE = 0.7

    # Blend weight: 0.0 = pure LLM directive, 1.0 = pure user mirror
    MIRROR_WEIGHT = 0.3

    def _blend_emotion(self, llm_emotion: str, llm_intensity: float) -> tuple[str, float]:
        """Blend LLM directive emotion with user's face emotion for natural mirroring.

        - Scales down intensity (LLM overshoots)
        - Mixes user face emotion at MIRROR_WEIGHT to add empathetic mirroring
        - If user is smiling and LLM says sad, the blend softens the sadness
        - If face emotion is unavailable, just scale the LLM directive
        """
        # Scale LLM intensity
        scaled = llm_intensity * self.INTENSITY_SCALE

        # Get current face emotion if available
        if not (self.face_emotion and self.face_tracker):
            return llm_emotion, round(scaled, 2)

        state = self.face_tracker.state
        if not state.present or state.face_crop is None:
            return llm_emotion, round(scaled, 2)

        try:
            face_label, face_conf, _ = self.face_emotion.detect(state.face_crop)
        except Exception:
            return llm_emotion, round(scaled, 2)

        face_directive = self._FACE_TO_DIRECTIVE.get(face_label, "neutral")

        # If face and LLM agree on emotion category, boost slightly (confidence)
        if face_directive == llm_emotion:
            blended_intensity = scaled * 1.1
            return llm_emotion, round(min(blended_intensity, 0.8), 2)

        # If face is neutral/low confidence, just use scaled LLM
        if face_directive == "neutral" or face_conf < 0.3:
            return llm_emotion, round(scaled, 2)

        # Blend: LLM emotion wins for the label (contextually correct),
        # but intensity is dampened when user's face disagrees
        dampened = scaled * (1.0 - self.MIRROR_WEIGHT * face_conf)
        return llm_emotion, round(max(dampened, 0.1), 2)

    # ── Memory Commitment ───────────────────────────────────────────────

    @staticmethod
    def _strip_context_tags(text: str) -> str:
        """Remove [MOOD: X], [CONTEXT: X], and [PLAYER_STATE]…[/PLAYER_STATE]
        tags. Used before fact extraction and memory persistence so the stored
        transcript is the user's actual words without the internal context
        prefix."""
        # Remove the multi-line [PLAYER_STATE]…[/PLAYER_STATE] block first
        text = re.sub(r'\[PLAYER_STATE\].*?\[/PLAYER_STATE\]\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'\[MOOD:\s*\w+\]\s*', '', text)
        text = re.sub(r'\[CONTEXT:\s*\w+\]\s*', '', text)
        return text.strip()

    async def _end_memory_session(self, user_id: str, transcript: list[dict]):
        """Send the full transcript to the server and wait until it is saved
        (compressed into memory). Runs as a background task. The caller counted
        this save in _saves_pending, which keeps the lapel pin purple while
        nobody is at the kiosk; it is released here however the save ends."""
        t0 = time.monotonic()
        try:
            # Clean the transcript: strip tags from user messages, directives from assistant
            clean_transcript = []
            for msg in transcript:
                if msg["role"] == "user":
                    clean_transcript.append({
                        "role": "user",
                        "content": self._strip_context_tags(msg["content"]),
                    })
                elif msg["role"] == "assistant":
                    clean_transcript.append({
                        "role": "assistant",
                        "content": self._clean_for_tts(msg["content"]),
                    })

            if not clean_transcript:
                return

            resp = await self.http_client.post(
                f"/memory/{user_id}/session_end",
                json={"transcript": clean_transcript, "wait": True},
                timeout=SAVE_TIMEOUT_S,
            )
            resp.raise_for_status()
            logger.info(f"[MEMORY] Session saved for user {user_id} in {time.monotonic() - t0:.1f}s "
                        f"({len(clean_transcript)} messages)")
        except Exception as e:
            logger.warning(f"[MEMORY] Session save failed after {time.monotonic() - t0:.1f}s: {e!r}")
        finally:
            self._saves_pending -= 1
            await self._refresh_indicator()

    # ── ASR ─────────────────────────────────────────────────────────────────

    async def _run_asr(self, audio: np.ndarray) -> str:
        """Run faster-whisper on audio chunk. Returns transcript string."""
        loop = asyncio.get_event_loop()

        hint = self._asr_hint

        def _transcribe():
            segments, info = self.asr_model.transcribe(
                audio,
                beam_size=ASR_BEAM_SIZE,
                language="en",
                vad_filter=True,
                hotwords=hint or None,  # the KB's names (knowledge.py)
            )
            return " ".join(seg.text.strip() for seg in segments)

        return await loop.run_in_executor(None, _transcribe)

    # ── Emotion Detection ───────────────────────────────────────────────────

    # emotion2vec returns Chinese labels — map to English
    # emotion2vec labels look like "<Chinese>/angry": the English part after "/" is looked up here.
    EMOTION_LABEL_MAP = {
        "angry": "angry",
        "disgusted": "disgusted",
        "fearful": "fearful",
        "happy": "happy",
        "neutral": "neutral",
        "other": "neutral",
        "sad": "sad",
        "surprised": "surprised",
        "unknown": "neutral",
    }

    async def _run_emotion_detection(self, audio: np.ndarray) -> str:
        """Run emotion2vec on user audio. Returns English emotion label."""
        loop = asyncio.get_event_loop()

        def _detect():
            try:
                if self.emotion_pipeline is None:
                    from funasr import AutoModel
                    self.emotion_pipeline = AutoModel(
                        model="iic/emotion2vec_plus_base",
                        hub="hf",
                        device="cpu",
                    )

                # emotion2vec expects 16kHz audio
                # Take a 2-second clip from the middle of the utterance
                clip_samples = min(len(audio), MIC_SAMPLE_RATE * 2)
                start = max(0, (len(audio) - clip_samples) // 2)
                clip = audio[start:start + clip_samples]

                result = self.emotion_pipeline.generate(clip, granularity="utterance", extract_embedding=False)
                if result and len(result) > 0:
                    labels = result[0].get("labels", [])
                    scores = result[0].get("scores", [])
                    if labels and scores:
                        best_idx = scores.index(max(scores))
                        raw_label = labels[best_idx]
                        english_label = self.EMOTION_LABEL_MAP.get(raw_label.rsplit("/", 1)[-1].strip().lower(), "neutral")
                        self.current_user_emotion = english_label
                        self.current_user_emotion_prob = scores[best_idx]
                        return english_label
            except Exception as e:
                logger.warning(f"Emotion detection failed: {e}")

            return "neutral"

        return await loop.run_in_executor(None, _detect)

    # ── Context Assembly ────────────────────────────────────────────────────

    def _assemble_context(self) -> list[dict]:
        """Build the full message list for the LLM.

        Structure:
          1. System prompt (with personality + rules)
          2. [PLAYER_STATE] block (current user emotion + trajectory)
          3. Conversation history
        """
        # Per-turn PLAYER_STATE carries ONLY volatile state — current emotion
        # (mood is added as a [MOOD: X] prefix in the baking step below). Identity
        # and compressed memory are NOT re-injected per turn; they live once in
        # the session system prompt (self._session_memory_block, set at identity
        # confirmation). Re-asserting "returning visitor + memory" every turn makes
        # the LLM re-greet mid-conversation and adds noise that degrades clean,
        # fast generation.
        player_state = (
            f"[PLAYER_STATE]\n"
            f"Emotion: {self.current_user_emotion} ({self.current_user_emotion_prob:.2f})\n"
            f"[/PLAYER_STATE]"
        )

        # System prompt is the bare constant — byte-stable across turns.
        # Dynamic context (player_state + mood + anonymous tag) is BAKED into
        # the latest user message in conversation_history at first read, then
        # marked with a _prefixed flag so subsequent calls don't re-prefix.
        # This keeps past user messages byte-identical to what llama-server
        # cached on prior turns — required for KV-cache reuse. Measured: when
        # player_state sits in the system prompt, or user_N in turn N+1 is not
        # byte-identical to user_N as sent in turn N, TTFT stays at ~1100 ms
        # on every turn.
        if (self.conversation_history
                and self.conversation_history[-1].get("role") == "user"
                and not self.conversation_history[-1].get("_prefixed")):
            prefix = ""
            if self.mood_tracker:
                prefix += f"[MOOD: {self.mood_tracker.mood}] "
            if self.id_manager and self.id_manager.is_anonymous:
                prefix += "[CONTEXT: ANONYMOUS] "
            self.conversation_history[-1]["content"] = (
                f"{player_state}\n{prefix}{self.conversation_history[-1]['content']}"
            )
            self.conversation_history[-1]["_prefixed"] = True

        # Identity + compressed memory live ONCE in the session system prompt:
        # byte-stable within a session, so it sits in the warm-cache prefix
        # (better cache reuse than baking memory into every turn), and the
        # gateway erases cache on user switch anyway. Empty for new/anonymous.
        # Order matters for the cache: persona, then the knowledge base (same
        # for every visitor), then this session's memory block.
        system_content = SYSTEM_PROMPT
        if self._kb_block:
            system_content = f"{system_content}\n\n{self._kb_block}"
        if self._session_memory_block:
            system_content = f"{system_content}\n\n{self._session_memory_block}"
        messages = [{"role": "system", "content": system_content}]
        # Multi-user sends the last 20 messages. Single-user sends the whole
        # live conversation: it only ever grows, so llama's prompt cache
        # matches everything but the newest turn (compaction bounds it).
        history = (self.conversation_history if self.identity.single_user
                   else self.conversation_history[-20:])
        # Strip the _prefixed flag from the dicts sent to the LLM (the flag
        # is internal bookkeeping; the OpenAI/llama API rejects unknown keys).
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        return messages

    # ── Server API calls ────────────────────────────────────────────────────

    # NOTE: LLM streaming is inline in _process_utterance for per-sentence TTS pipelining.

    async def _call_tts(self, text: str, record_telemetry: bool = True, respell: bool = True) -> Optional[dict]:
        """Call the server's /tts/synthesize. Returns dict with audio_base64, sample_rate, word_timestamps.

        record_telemetry=False for the prefetched chunk-2 synth so the
        dashboard's tts_ms keeps meaning "first-chunk synthesis" (the TTFA
        contributor), not whichever call finished last. respell=False for the
        pronunciation check, which must hear the exact spelling it's testing.
        """
        if not text.strip():
            return None

        try:
            t_start = time.time()
            response = await self.http_client.post(
                "/tts/synthesize",
                json={
                    # Respellings reach the voice only; history and UE keep the real text.
                    "text": self._pronunciations.apply(text) if respell else text,
                    "emotion_exaggeration": 0.5,
                },
                timeout=15.0,
            )
            response.raise_for_status()
            if record_telemetry:
                self._last_tts_ms = (time.time() - t_start) * 1000
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"TTS call failed: {e}")
            return None

    async def _call_a2e(self, audio_base64: str, sample_rate: int) -> Optional[dict]:
        """Call the server's /a2e/detect — NPC self-emotion from TTS audio."""
        try:
            response = await self.http_client.post(
                "/a2e/detect",
                json={
                    "audio_base64": audio_base64,
                    "sample_rate": sample_rate,
                },
                timeout=5.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.debug(f"A2E call failed: {e}")
            return None

    # ── Health Check ────────────────────────────────────────────────────────

    async def _run_health_check(self):
        """Periodic server connectivity check."""
        while self._running:
            try:
                resp = await self.http_client.get("/health", timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    logger.debug(f"Server health: OK, VRAM free: {data.get('vram_free_gb', '?')}GB")
                else:
                    logger.warning(f"Server health check returned {resp.status_code}")
            except Exception as e:
                logger.warning(f"Server unreachable: {e}")

            await asyncio.sleep(30)

    # ── Gaze Tracking Loop ─────────────────────────────────────────────────

    def _glance_mode(self) -> str:
        """What the conversation is doing, for the glance planner: she talks (steady eye contact), she is preparing
        an answer (a glance up and aside is natural), the visitor talks, or nobody talks."""
        if self._is_speaking:
            return "speaking"
        if self.state == ConvState.PROCESSING:
            return "thinking"
        if self.is_speech_active:
            return "listening"
        return "quiet"

    def _voice_delay_ms(self) -> int:
        """kiosk_config.ini [avatar] voice_delay_ms: Unreal holds her voice this long on its way to the speakers so it
        lines up with her lips (the lip sync hears the sound first, then needs ~0.1 s to move her mouth)."""
        cfg = configparser.ConfigParser()
        try:
            cfg.read(KIOSK_CONFIG_PATH, encoding="utf-8")
            return max(0, min(400, int(float(cfg.get("avatar", "voice_delay_ms", fallback="0")))))
        except (configparser.Error, ValueError, OSError):
            return 0

    async def _run_gaze_tracking(self):
        """Poll the face tracker, compute the gaze offset, send it to UE5 at ~10 fps.

        UE5 aims her eyes at the camera that films her and turns them by this offset toward the visitor; with
        "present": false her eyes rest in her head."""
        interval = 1.0 / self.face_tracker._target_fps
        last_present = False
        next_report = 0.0

        while self._running:
            now = time.monotonic()
            if self.gaze_calculator.maybe_reload(now):
                delay = self._voice_delay_ms()          # kiosk_config.ini edited: her voice delay may have changed
                if delay != self._sent_voice_delay:
                    self._sent_voice_delay = delay
                    await self._send_to_ue({"type": "av_sync", "voice_delay_ms": delay})
                    logger.info(f"Her voice delay is now {delay} ms (lip sync)")
            self._glances.maybe_reload(now)
            state = self.face_tracker.state
            result = self.gaze_calculator.update(state.landmarks if state.present else None)

            if result.valid:
                glance_yaw, glance_pitch = self._glances.offset(now, self._glance_mode())
                await self._send_to_ue({
                    "type": "gaze",
                    "present": True,
                    "yaw": round(result.yaw_deg + glance_yaw, 2),
                    "pitch": round(result.pitch_deg + glance_pitch, 2),
                    "distance_in": result.distance_in,
                    "focus_in": result.focus_in,      # her two eyes converge on the visitor at this distance
                })
                if not last_present or now >= next_report:
                    next_report = now + 10.0
                    logger.info(
                        f"{'Face acquired' if not last_present else 'Gaze'} — distance {result.distance_in:.0f}\" "
                        f"({result.focus_in:.0f}\" from her eyes), "
                        f"seen from her eyes at {result.angle_yaw_deg:+.1f} deg (+ = her left), "
                        f"{result.angle_pitch_deg:+.1f} deg (+ = up); eyes turn {result.yaw_deg:+.1f}/{result.pitch_deg:+.1f}"
                    )
                last_present = True
            else:
                self._glances.offset(now, "away")
                if last_present:
                    # User left - her eyes rest in her head
                    self.gaze_calculator.reset()
                    await self._send_to_ue({"type": "gaze", "present": False, "yaw": 0.0, "pitch": 0.0})
                    logger.info("Face lost — eyes rest")
                last_present = False

            await asyncio.sleep(interval)

    # ── Face Recognition Loop ──────────────────────────────────────────────

    async def _run_face_recognition(self):
        """Poll face recognizer at ~2Hz. Recognition runs in executor to avoid blocking."""
        loop = asyncio.get_event_loop()

        while self._running:
            if self.identity.single_user:
                # Single-user mode: presence only - no recognition, no sessions.
                await self._single_user_presence_poll()
                await asyncio.sleep(0.5)
                continue
            try:
                # Wall-clock session expiry, UNGATED: with UE disconnected,
                # update() below never runs, so without this ANY session would
                # freeze mid-life — PIN-verified ones would resume hours later
                # without re-auth; tentative ones would leave the id flow stuck
                # so a returning face gets no greeting. Expire on real elapsed
                # time (tracker presence as truth) and route through the normal
                # "user_left" branch so side effects (memory digest, LLM/id
                # reset, indicator) land on time.
                if self.face_recognizer.expire_if_gone():
                    event = "user_left"
                elif not self.ue_websockets:
                    # Don't trigger identification until UE5 is connected (Play mode)
                    await asyncio.sleep(0.5)
                    continue
                else:
                    event = await loop.run_in_executor(None, self.face_recognizer.update)

                # Face-presence gate: keep timestamp fresh while a face is
                # being tracked. _process_utterance_inner uses this to drop ambient
                # audio when nobody has been at the kiosk for FACE_GATE_S seconds.
                if self.face_recognizer.has_user:
                    self._last_face_seen_at = time.time()

                # Refresh the lapel-pin indicator on every poll
                # so face_acquired → white and face_lost → grey transition
                # promptly without waiting for a ConvState transition.
                await self._refresh_indicator()

                # During identification or anonymous sessions,
                # only process "user_left" — ignore rechecks that would disrupt the flow
                if self.id_manager and event and event != "user_left":
                    if self.id_manager.is_identifying or self.id_manager.is_anonymous:
                        continue

                if event == "user_identified":
                    session = self.face_recognizer.session
                    if session:
                        logger.info(
                            f"[FACE ID] Match: {session.name} "
                            f"(tier={session.tier}, sim={session.similarity:.3f}) "
                            f"— awaiting PIN verification"
                        )
                        self.current_user_similarity = float(session.similarity)
                        self._emit_event("recog", "confident_match",
                                         f"{session.name} sim={session.similarity:.3f}",
                                         name=session.name,
                                         user_id=session.user_id,
                                         sim=float(session.similarity),
                                         tier=session.tier)
                        # Don't set current_user_name yet — wait for PIN
                        self.id_manager.on_face_event(
                            "user_identified",
                            user_name=session.name,
                            user_id=session.user_id,
                            similarity=session.similarity,
                        )
                        await self._speak_id_line()

                elif event == "new_face":
                    self.current_user_name = None
                    self.current_user_id = None
                    logger.info("[FACE ID] New face detected — unknown user")
                    self._emit_event("camera", "face_appeared",
                                     "New face detected — unknown user")
                    self.id_manager.on_face_event("new_face")
                    await self._speak_id_line()

                elif event == "user_left":
                    await self._visitor_left()

            except Exception as e:
                logger.warning(f"Face recognition error: {e}")

            await asyncio.sleep(0.5)  # ~2Hz polling

    # ── Identification Helpers ─────────────────────────────────────────────

    async def _visitor_left(self):
        """Kiosk mode: the visitor walked away. An identified visitor's conversation is saved to their memory;
        a guest's is not (zero memory). Either way her conversation is wiped for the next person."""
        logger.info("[FACE ID] User left — ending session")
        self._emit_event("userid", "session_end",
                         "User left — session ended")
        # Commit memory before clearing (identified users only).
        # Use orchestrator-tracked current_user_id rather than the
        # face_recognizer fields: by the time poll() returns "user_left",
        # face_recognizer._end_session() has already nulled its session,
        # so .has_user and .user_id are False/None here.
        user_id = self.current_user_id
        if (user_id is not None
                and self.id_manager and not self.id_manager.is_anonymous
                and self.conversation_history):
            transcript = list(self.conversation_history)
            # Counted before the task starts so the refresh below
            # already shows purple; the task releases it when saved.
            self._saves_pending += 1
            asyncio.create_task(self._end_memory_session(str(user_id), transcript))

        self.current_user_name = None
        self.current_user_id = None
        self.current_user_similarity = 0.0
        self.id_manager.on_face_event("user_left")
        # Stop any playing audio. Sent unconditionally — UE5 plugin tail
        # can outlast the orchestrator's _is_speaking window; gating here
        # would let user A's audio bleed into user B's identification prompt.
        await self._send_to_ue({"type": "stop"})
        self._is_speaking = False
        # Back to her neutral resting face: a sad or serious last reply would
        # otherwise keep her faint resting smile faded out until the next visitor.
        await self._send_to_ue({"type": "emotion", "value": "neutral", "intensity": 0.3})
        # Reset LLM session so next user gets clean context
        await self._reset_llm_session()

        # Lapel pin: purple while this visitor's memory is being
        # saved (see _saves_pending), otherwise straight to grey.
        await self._refresh_indicator()

    async def _reset_llm_session(self):
        """End the current LLM session on the server and start fresh.

        Called when a user leaves so the next user gets clean context.
        """
        try:
            await self.http_client.post(
                "/llm/end_session",
                json={"session_id": self.session_id},
                timeout=5.0,
            )
        except Exception:
            pass
        self.conversation_history.clear()
        self.session_id = str(uuid.uuid4())
        self.emotion_trajectory.clear()
        self._session_memory_block = ""
        if self.mood_tracker:
            self.mood_tracker.reset()
        logger.info(f"[SESSION] Reset — new session: {self.session_id}")

    async def _speak_degraded_mode(self):
        """Tell the visitor the truth when the LLM is unreachable.

        TTS lives in a different process from the LLM, so she can almost
        always still say this. Rate-limited: a visitor who keeps talking to an
        offline model should not trigger the same apology on every utterance.
        """
        logger.error("[DEGRADED] LLM unreachable during identification — "
                     "informed the visitor; id state left intact for resume")
        now = time.time()
        if (now - self._degraded_spoken_at) < DEGRADED_REPEAT_S:
            return
        self._degraded_spoken_at = now
        await self._send_to_ue({"type": "emotion", "value": "confused", "intensity": 0.5})
        await self._play_sentence(
            "I'm sorry - my thinking has gone offline, so I can't sign you in "
            "right now. Please try again in a moment.",
            force=True,
        )

    async def _speak_id_line(self):
        """Speak the identification manager's pending avatar line via TTS."""
        line = self.id_manager.avatar_line
        if not line:
            return

        # Strip directives for TTS, but send directives to UE5 first
        clean_text, directives = strip_directives(line)
        for d in directives:
            if d["name"] == "emotion":
                await self._send_to_ue({
                    "type": "emotion",
                    "value": d.get("value", ""),
                    "intensity": d.get("intensity", 0.5),
                })
            elif d["name"] == "gesture":
                self._pending_gesture_tags.append(d.get("value", ""))   # played with its sentence

        clean_text = self._clean_for_tts(clean_text)
        if clean_text and len(clean_text) > 3:
            # Preempt any in-flight avatar audio so identity-flip prompts don't overlap.
            # Two-layer preempt:
            #   1. _suppress_utterance_playback gates utterance-task _play_sentence calls
            #      (subsequent sentences in the LLM stream exit on entry).
            #   2. WS stop + brief settle clears UE5's active AudioComponent so the
            #      next speak does not overlap with the prior audio.
            # The _play_sentence call below passes force=True to bypass the gate.
            self._suppress_utterance_playback = True
            try:
                await self._send_to_ue({"type": "stop"})
                await asyncio.sleep(0.1)
                await self._set_state(ConvState.SPEAKING)
                await self._play_sentence(clean_text, force=True)
                await self._set_state(ConvState.LISTENING)
            finally:
                self._suppress_utterance_playback = False

    async def _handle_id_utterance(self, transcript: str) -> bool:
        """Process an utterance through the identification manager.

        Returns True if the utterance was consumed (bypass LLM pipeline).
        Returns False if identification is complete and normal pipeline should run.
        """
        if not self.id_manager.is_identifying:
            return False

        # LLM-first understanding of the reply: rely on the LLM, not
        # predetermined answers. Grammar-forced yes/no/unclear
        # classification for confirm states; LLM name extraction for name
        # states (stops ASR garble like "Hi" being enrolled as a name). Both
        # run concurrently when a state accepts either (~75 ms each, warm).
        #
        # There is deliberately NO word-list fallback when the LLM cannot be
        # reached. The kiosk cannot converse without it, so a word-list login
        # only admits the visitor to a system that cannot answer — and it
        # disguises an outage as unreliable understanding (a word list accepts
        # or rejects replies like "that's me" or "certainly am" purely by which
        # words it happens to contain). Say so out loud instead.
        intent = None
        name_guess = None
        llm_down = False
        expected = self.id_manager.expected_input
        if expected in ("yesno", "yesno_name", "name"):
            calls = []
            if expected in ("yesno", "yesno_name"):
                calls.append(classify_confirmation(self.id_manager.pending_question, transcript))
            if expected in ("name", "yesno_name"):
                calls.append(extract_name(transcript))
            try:
                results = await asyncio.gather(*calls)
                for r in results:
                    if r.get("source") != "llm":
                        llm_down = True
                    elif "intent" in r:
                        intent = r["intent"]
                    elif "name" in r:
                        name_guess = r["name"]
                total_ms = sum(r["latency_ms"] for r in results)
                logger.info(f"[CONFIRM] expected={expected} intent={intent} "
                            f"name={name_guess!r} ({total_ms:.0f}ms)")
            except Exception as e:
                logger.warning(f"[CONFIRM] resolver error ({e!r})")
                llm_down = True

        if llm_down:
            # Leave the identification state untouched so the flow resumes
            # where it paused once the model is back.
            await self._speak_degraded_mode()
            return True

        action = self.id_manager.process_utterance(transcript, intent=intent,
                                                   name_guess=name_guess)

        if action is None:
            # Still identifying — speak her next line
            await self._speak_id_line()
            return True

        if action.startswith("check_db:"):
            # check_db:NAME — look up name in face library
            name = action.split(":", 1)[1]
            matched, uid, uname, sim = self._lookup_user_by_name(name)
            self.id_manager.on_db_result(matched, uid, uname, sim)
            await self._speak_id_line()
            return True

        if action.startswith("register:"):
            # register:NAME:PIN
            parts = action.split(":", 2)
            name, pin = parts[1], parts[2]
            user_id = self.face_recognizer.register_user(name)
            self.face_library.set_pin(user_id, pin)
            self.face_recognizer.confirm_identity()
            self.current_user_name = name
            self.current_user_id = user_id
            logger.info(f"[ID] Registered {name} (id={user_id}) with PIN")
            asyncio.create_task(self._create_server_memory(str(user_id), name))
            await self._speak_id_line()
            return True

        if action.startswith("check_pin:"):
            # check_pin:USER_ID:PIN
            parts = action.split(":", 2)
            user_id, pin = int(parts[1]), parts[2]
            verified = self.face_library.verify_pin(user_id, pin)
            self.id_manager.on_pin_result(verified)
            if verified:
                self.current_user_name = self.id_manager._pending_user_name or self.id_manager._pending_name
                self.current_user_id = user_id
                self.face_recognizer.confirm_identity()
                logger.info(f"[ID] PIN verified for user {user_id}")
                self._emit_event("userid", "pin_verified",
                                 f"PIN verified for {self.current_user_name}",
                                 user_id=user_id, name=self.current_user_name)
                await self._fetch_and_speak_greeting(str(user_id))
            else:
                await self._speak_id_line()
            return True

        if action == "anonymous":
            self.current_user_name = self.id_manager._pending_name  # may have a name but no persistence
            self.current_user_id = None  # anonymous: no DB profile, no facts/transcripts persisted
            # Clear face recognizer session — no face data retained for anonymous users
            self.face_recognizer._end_session()
            self.face_recognizer._face_was_present = True  # still present, just not tracked
            logger.info(f"[ID] Anonymous mode — name={self.current_user_name}, face session cleared")
            await self._speak_id_line()
            return True

        return True

    def _lookup_user_by_name(self, name: str) -> tuple:
        """Check if a user with this name exists in the face library.

        Returns (matched, user_id, user_name, similarity).
        """
        users = self.face_library.list_users()
        for u in users:
            if u["name"].lower() == name.lower():
                return (True, u["id"], u["name"], 0.0)
        return (False, None, None, 0.0)

    async def _create_server_memory(self, user_id: str, name: str):
        """Create a memory profile on the server for a new user (fire-and-forget)."""
        try:
            resp = await self.http_client.put(
                f"/memory/{user_id}/facts",
                json={"fact": f"User's name is {name}.", "pinned": True},
                timeout=5.0,
            )
            if resp.status_code == 200:
                logger.info(f"[MEMORY] Created server profile for {name} (id={user_id})")
            else:
                logger.warning(f"[MEMORY] server profile creation returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to create server profile: {e}")

    async def _fetch_and_speak_greeting(self, user_id: str):
        """Fetch personalized greeting from the server and speak it."""
        try:
            resp = await self.http_client.post(
                f"/memory/{user_id}/greeting",
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                greeting = data.get("greeting", "")
                new_session_id = data.get("session_id")
                memory_context = data.get("memory_context", "")
                user_name = data.get("user_name", "")

                if greeting:
                    # Establish identity + compressed memory ONCE for the session,
                    # in the system prompt (read by _assemble_context), instead of
                    # re-injecting per turn. The "already greeted" note stops the
                    # LLM re-greeting mid-conversation.
                    block = "[USER PROFILE]\n"
                    if user_name:
                        block += f"You are talking with {user_name}.\n"
                    if memory_context:
                        block += f"{memory_context}\n"
                    block += (
                        "You have ALREADY greeted them at the start of this session. "
                        "Do NOT greet again or re-introduce yourself; just continue "
                        "the conversation naturally.\n"
                        "[/USER PROFILE]"
                    )
                    self._session_memory_block = block

                    # Use the server's session_id (KV cache primed with user memory)
                    if new_session_id:
                        self.session_id = new_session_id
                        logger.info(f"[MEMORY] Switched to greeting session: {new_session_id}")

                    # Speak the greeting (includes directives)
                    clean_text, directives = strip_directives(greeting)
                    for d in directives:
                        if d["name"] == "emotion":
                            await self._send_to_ue({
                                "type": "emotion",
                                "value": d.get("value", ""),
                                "intensity": d.get("intensity", 0.5),
                            })
                        elif d["name"] == "gesture":
                            self._pending_gesture_tags.append(d.get("value", ""))   # played with its sentence

                    clean_text = self._clean_for_tts(clean_text)
                    if clean_text and len(clean_text) > 3:
                        await self._set_state(ConvState.SPEAKING)
                        await self._play_sentence(clean_text)
                        await self._set_state(ConvState.LISTENING)

                    logger.info(f"[MEMORY] Greeting spoken for user {user_id}")
                    return

            logger.warning(f"[MEMORY] Greeting endpoint returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"[MEMORY] Greeting fetch failed: {e}")

        # Fallback: use identification manager's line
        await self._speak_id_line()

    # -- Knowledge base (kiosk_config.ini [kb]; see knowledge.py) -------------

    async def _load_knowledge_base(self):
        """Whole-KB-in-context: read the KB into her system prompt at startup,
        then pre-read it into the LLM cache so the first question doesn't
        wait for it."""
        self._load_names()  # the kiosk's own respellings, KB or not
        enabled, configured = load_kb_config()
        if not enabled:
            return
        folder = find_kb_folder(configured)
        if not folder:
            logger.info("[KB] No knowledge base found "
                        + (f"at {configured}" if configured else "on a removable drive"))
            return
        text, loaded, skipped = read_kb(folder)
        if not text:
            logger.warning(f"[KB] {folder} has no readable text files")
            return
        name = os.path.basename(os.path.normpath(folder))
        self._load_names(folder, text)
        block = kb_block(name, text, self._pronunciations.table)
        try:
            resp = await self.http_client.post("http://127.0.0.1:8080/tokenize",
                                               json={"content": block}, timeout=120.0)
            tokens = len(resp.json()["tokens"])
        except Exception as e:
            tokens = len(block) // 3  # deliberately high estimate
            logger.warning(f"[KB] Token count unavailable ({e!r}); estimating {tokens:,}")
        if tokens > KB_TOKEN_BUDGET:
            logger.error(f"[KB] '{name}' is {tokens:,} tokens, over the {KB_TOKEN_BUDGET:,} budget - not loaded")
            self._load_names()  # back to the kiosk's own respellings
            return
        self._kb_block = block
        self._kb_folder, self._kb_text = folder, text
        logger.info(f"[KB] Loaded '{name}' from {folder}: {loaded} files ({skipped} skipped), {tokens:,} tokens"
                    + (f"; pronunciation guide: {len(self._pronunciations.table)} names" if self._pronunciations.table else ""))
        # Reading a large KB takes tens of seconds; do it in the background so
        # startup isn't held up. A question asked meanwhile waits for it.
        asyncio.create_task(self._prewarm_kb(text))

    def _load_names(self, folder: Optional[str] = None, kb_text: str = ""):
        """Respellings for her voice - the pronunciation check's, then the
        kiosk's pronunciations.txt, then the drive's; later files win - and
        the names Whisper listens for: hand-written respelled names first,
        then the KB's most widespread hard-to-hear names."""
        manual_files = [KIOSK_PRONUNCIATIONS] + ([os.path.join(folder, PRONUNCIATIONS_FILE)] if folder else [])
        manual = Pronunciations.load(*manual_files)
        self._manual_names = {n.lower() for n in manual.names}
        self._pronunciation_files = [AUTO_PRONUNCIATIONS] + manual_files
        self._pronunciations = Pronunciations.load(*self._pronunciation_files)
        tokenizer = self.asr_model.hf_tokenizer
        count = lambda s: len(tokenizer.encode(" " + s, add_special_tokens=False).ids)
        names = manual.names + (kb_names(kb_text, count) if kb_text else [])
        self._asr_hint = asr_hint(names, count)
        if self._pronunciations.names or self._asr_hint:
            logger.info(f"[NAMES] {len(self._pronunciations.names)} respellings "
                        f"(hand-written: {', '.join(manual.names) or 'none'}); "
                        f"listening for {len(self._asr_hint.split(', ')) if self._asr_hint else 0} "
                        f"names ({count(self._asr_hint)} tokens): {self._asr_hint}")

    def _pronunciation_mtimes(self) -> tuple:
        stamps = []
        for path in self._pronunciation_files:
            try:
                stamps.append(os.path.getmtime(path))
            except OSError:
                stamps.append(None)
        return tuple(stamps)

    async def _watch_pronunciations(self):
        """Apply a changed pronunciations file within seconds - an Accept on
        the /admin Names page, a hand edit, or the check's own fixes - without
        a restart. Her in-prompt pronunciation guide keeps its startup
        version (changing it would move the cached prompt); it catches up at
        the next restart."""
        seen = self._pronunciation_mtimes()
        while self._running:
            await asyncio.sleep(5.0)
            now = self._pronunciation_mtimes()
            if now != seen:
                seen = now
                self._load_names(self._kb_folder, self._kb_text)
                logger.info("[NAMES] A pronunciations file changed on disk - reloaded")
            if os.path.exists(SAY_REQUEST) and not self._kiosk_busy():
                await self._say_requested()

    async def _say_requested(self):
        """Say the lines queued by the /admin Names page, one at a time,
        through her real voice path - respellings applied, lip sync in UE -
        and log exactly what her voice was given."""
        lines = take_say_request()
        if not lines:
            return
        await self._set_state(ConvState.SPEAKING)
        try:
            for line in lines:
                spoken = self._pronunciations.apply(line)
                logger.info(f"[NAMES] Saying {line!r}" + (f" as {spoken!r}" if spoken != line else " as spelled"))
                await self._play_sentence(line)
        finally:
            await self._set_state(ConvState.LISTENING)

    async def _prewarm_kb(self, kb_text: str):
        t0 = time.monotonic()
        if await self._prewarm_llm():
            logger.info(f"[KB] Pre-read into the LLM cache in {time.monotonic() - t0:.1f}s")
        if check_enabled():
            self._pronunciation_task = asyncio.create_task(self._run_pronunciation_check(kb_text))

    async def _prewarm_llm(self) -> bool:
        """Have the LLM read the current prompt now (a 1-token reply), so the
        next real turn finds it cached."""
        try:
            resp = await self.http_client.post("/llm/generate", json={
                "messages": self._assemble_context() + [{"role": "user", "content": "[Reply with OK.]"}],
                "session_id": self.session_id, "stream": False, "max_tokens": 1, "temperature": 0.0,
            }, timeout=300.0)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"[KB] Pre-read failed ({e!r}); the first reply will read it instead")
            return False

    # -- Pronunciation check (pronunciation.py; [kb] pronunciation_check) ----
    # Background work for either mode: runs only while nobody is at the kiosk
    # and nothing is in flight, and yields the moment that changes.

    def _kiosk_busy(self) -> bool:
        """A visitor talking, her thinking or talking, a turn in flight, or a
        memory save running."""
        return (self.is_speech_active or self._is_speaking or self._llm_lock.locked()
                or self.state != ConvState.LISTENING or self._committing or self._saves_pending > 0)

    async def _wait_for_quiet(self):
        """Return once nobody has been at the kiosk, and nothing has been in
        flight, for QUIET_S."""
        while True:
            now = time.time()
            if self._kiosk_busy():
                self._last_busy_at = now
            if now - max(self._last_busy_at, self._last_face_seen_at) >= QUIET_S:
                return
            await asyncio.sleep(1.0)

    async def _background_llm(self, messages: list[dict], max_tokens: int,
                              temperature: float = 0.2) -> Optional[str]:
        """A streamed call in the live session for background work. It stops
        as soon as the kiosk gets busy, and a reply that starts meanwhile makes
        the gateway close it. None if stopped or failed."""
        parts = []
        try:
            async with self.http_client.stream(
                "POST", "/llm/generate",
                json={"messages": messages, "session_id": self.session_id, "stream": True,
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=120.0,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if self._kiosk_busy():
                        return None
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if data.get("error") or (data.get("done") and data.get("cancelled")):
                        return None
                    if data.get("done"):
                        break
                    parts.append(data.get("token", ""))
        except Exception as e:
            logger.warning(f"[PRONOUNCE] LLM call failed: {e!r}")
            return None
        return "".join(parts).strip() or None

    async def _pronunciation_targets(self, items: list) -> dict:
        """The LLM's own IPA and respellings for a batch of [(word, name)],
        asked after the live conversation (appended, never stored in it) so
        the cached KB is reused rather than evicted."""
        for _ in range(3):
            await self._wait_for_quiet()
            messages = self._assemble_context()
            if messages[-1]["role"] == "user":   # a turn mid-way: keep roles alternating
                await asyncio.sleep(5.0)
                continue
            messages.append({"role": "user", "content": llm_request(items)})
            text = await self._background_llm(messages, max_tokens=80 * len(items))
            if text:
                return parse_llm_table(text, [w for w, _ in items])
        return {}

    async def _run_pronunciation_check(self, kb_text: str):
        """Check how her voice says the rare words in the KB's names, a step at
        a time while the kiosk is quiet; each fix applies as it's found. Words
        checked before are skipped, so after a KB's first run this ends at once."""
        tokenizer = self.asr_model.hf_tokenizer
        count = lambda s: len(tokenizer.encode(" " + s, add_special_tokens=False).ids)
        cache = load_cache()
        everything = words_to_check(kb_names(kb_text, count, limit=5000), count, set())
        save_current_words(w for w, _ in everything)       # the Names page shows only these
        skip = set(cache) | self._manual_names
        todo = [(w, name) for w, name in everything if w.lower() not in skip]
        if not todo:
            return
        known = lookup_dictionary([w for w, _ in todo])   # trusted targets
        logger.info(f"[PRONOUNCE] Checking how she says {len(todo)} words from the KB's names "
                    f"({len(known)} in the pronouncing dictionary; {len(cache)} checked before), "
                    f"only while nobody is at the kiosk")
        worker = PhonemeWorker(device=f"cuda:{ASR_DEVICE_INDEX}")

        async def say_and_hear(text):
            await self._wait_for_quiet()
            tts = await self._call_tts(text, record_telemetry=False, respell=False)
            if not tts or not tts.get("audio_base64"):
                return None
            return await worker.hear(tts["audio_base64"], tts.get("sample_rate", 24000))

        done = 0
        try:
            for start in range(0, len(todo), LLM_BATCH):
                batch = todo[start:start + LLM_BATCH]
                targets = await self._pronunciation_targets(batch)
                for word, _name in batch:
                    done += 1
                    variants = known.get(word.lower()) or []
                    llm_ipa, respellings = targets.get(word, (None, []))
                    if not variants and not llm_ipa:
                        logger.info(f"[PRONOUNCE] {done}/{len(todo)} {word}: no pronunciation from the LLM; next start")
                        continue
                    # Only two independent sources agreeing earn an automatic fix
                    # (a dictionary alone can misread an unusual product name).
                    both = bool(variants and llm_ipa and agree(variants, llm_ipa))
                    ipa, alternatives = (variants[0], variants[1:]) if variants else (llm_ipa, [])
                    result = await check_word(word, ipa, respellings, say_and_hear,
                                              trusted=both, alternatives=alternatives)
                    result["rank"] = done
                    result["source"] = ("the dictionary and the LLM" if both
                                        else "the dictionary" if variants else "the LLM")
                    if variants and llm_ipa and not both:
                        result["other_ipa"] = llm_ipa
                    if result["status"] == "error":
                        continue    # nothing heard; tried again next start
                    result["checked"] = today()
                    cache[word.lower()] = result
                    save_cache(cache)
                    write_auto_file(cache)
                    if result["status"] == "fixed":
                        self._pronunciations = Pronunciations.load(*self._pronunciation_files)
                    logger.info(f"[PRONOUNCE] {done}/{len(todo)} {word} /{ipa}/: {describe(result)}")
            logger.info(f"[PRONOUNCE] Done: {summarize(cache)}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[PRONOUNCE] Check stopped ({e!r}); it resumes at the next start")
        finally:
            await worker.close()

    # -- Single-user mode (kiosk_config.ini [identity] mode = single_user) ----
    # One person, no identification, one live conversation that is never
    # reset; see single_user.py. Nothing below runs in multi_user mode.

    async def _start_single_user(self):
        """Presence instead of recognition, one never-expiring LLM session, and
        the conversation restored from exchanges not yet committed to memory,
        so a restart loses nothing."""
        self.face_recognizer = Presence(self.face_tracker)
        self.current_user_id = SINGLE_USER_ID
        self.session_id = SESSION_PREFIX + str(uuid.uuid4())
        self._pending = PendingExchanges()
        leftovers = self._pending.read()
        for e in leftovers:
            self.conversation_history.append({"role": "user", "content": e["user"], "_recorded": True})
            self.conversation_history.append({"role": "assistant", "content": e["assistant"], "_recorded": True})
        await self._load_single_user_memory()
        self._last_activity_at = time.time()
        logger.info(f"[SINGLE] single_user mode - presence only, never signed in or out; "
                    f"session {self.session_id}; restored {len(leftovers)} uncommitted exchange(s)")

    async def _load_single_user_memory(self) -> bool:
        """(Re)build her memory block. Only at startup, at compaction, or on
        a retry after a failed load - the system prompt must stay byte-stable
        for the KV cache to keep matching the conversation."""
        state = None
        try:
            resp = await self.http_client.get(f"/memory/{SINGLE_USER_ID}/state", timeout=10.0)
            resp.raise_for_status()
            state = resp.json()
        except Exception as e:
            logger.warning(f"[SINGLE] Could not load memory: {e!r}")
        self._session_memory_block = profile_block(state)
        self._single_user_memory_loaded = state is not None
        if state is not None:
            self._single_user_has_memory = bool(state.get("facts") or state.get("summary"))
        return state is not None

    async def _single_user_presence_poll(self):
        """Presence drives the room-audio gate, the pin and the welcome-back;
        arriving or leaving never touches the conversation."""
        try:
            event = self.face_recognizer.update()
            if self.face_recognizer.has_user:
                self._last_face_seen_at = time.time()
            await self._refresh_indicator()
            if event == "arrived":
                away = self.face_recognizer.away_s
                logger.info("[SINGLE] Someone arrived ("
                            + ("first time since startup" if away is None else f"away {away:.0f}s") + ")")
                if self.ue_websockets and (away is None or away >= WELCOME_BACK_AFTER_S):
                    # A task, so presence keeps updating (face gate, pin) while she speaks.
                    asyncio.create_task(self._single_user_welcome(away))
            elif event == "left":
                logger.info("[SINGLE] Nobody in front of the camera")
        except Exception as e:
            logger.warning(f"[SINGLE] Presence error: {e!r}")

    async def _single_user_welcome(self, away_s: Optional[float]):
        """Greet on arrival: a fixed first hello while there is nothing to
        remember, otherwise a note in the live conversation so she welcomes
        them back knowing everything said before. Neither goes to memory."""
        if self._llm_lock.locked():
            return
        async with self._llm_lock:
            if not self._single_user_memory_loaded:
                await self._load_single_user_memory()
            if not self.conversation_history and not self._single_user_has_memory:
                await self._speak_directive_line(FIRST_GREETING)
                return
            self.conversation_history.append(
                {"role": "user", "content": welcome_back_note(away_s), "_note": True})
            await self._set_state(ConvState.PROCESSING)
            text = await self._single_user_llm(self._assemble_context(), max_tokens=120, temperature=0.8)
            if not text:
                self.conversation_history.pop()  # keep user/assistant alternation
                await self._set_state(ConvState.LISTENING)
                return
            self.conversation_history.append({"role": "assistant", "content": text, "_recorded": True})
            await self._speak_directive_line(text)

    async def _speak_directive_line(self, line: str):
        """Speak one line of hers: its emotion/gesture directives go to UE5,
        the rest to TTS."""
        clean_text, directives = strip_directives(line)
        for d in directives:
            if d["name"] == "emotion":
                await self._send_to_ue({"type": "emotion", "value": d.get("value", ""),
                                        "intensity": d.get("intensity", 0.5)})
            elif d["name"] == "gesture":
                self._pending_gesture_tags.append(d.get("value", ""))   # played with its sentence
        clean_text = self._clean_for_tts(clean_text)
        if clean_text and len(clean_text) > 3:
            await self._set_state(ConvState.SPEAKING)
            await self._play_sentence(clean_text)
        await self._set_state(ConvState.LISTENING)

    def _single_user_busy(self) -> bool:
        """Someone talking, her thinking or talking, or a turn in progress."""
        return (self.is_speech_active or self._is_speaking or self._llm_lock.locked()
                or self.state != ConvState.LISTENING)

    async def _single_user_llm(self, messages: list[dict], max_tokens: int = 600,
                               temperature: float = 0.3, abort_on_speech: bool = False) -> Optional[str]:
        """One streamed call in the live session. Returns the text, or None if
        it was cancelled or failed. Streaming keeps it preemptible: a reply that
        starts meanwhile makes the gateway close this stream, and with
        abort_on_speech it also stops as soon as someone starts talking."""
        parts = []
        try:
            async with self.http_client.stream(
                "POST", "/llm/generate",
                json={"messages": messages, "session_id": self.session_id, "stream": True,
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=120.0,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if abort_on_speech and (self.is_speech_active or self._llm_lock.locked()):
                        return None
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    if data.get("error"):
                        logger.warning(f"[SINGLE] LLM error: {str(data.get('detail', data['error']))[:200]}")
                        return None
                    if data.get("done"):
                        return None if data.get("cancelled") else ("".join(parts).strip() or None)
                    parts.append(data.get("token", ""))
        except Exception as e:
            logger.warning(f"[SINGLE] LLM call failed: {e!r}")
            return None
        return "".join(parts).strip() or None

    async def _run_single_user_commits(self):
        """Commit the pending exchanges to memory after COMMIT_IDLE_S of quiet."""
        while self._running:
            await asyncio.sleep(1.0)
            try:
                now = time.time()
                if self._single_user_busy():
                    self._last_activity_at = now
                    continue
                if self._committing or now - self._last_activity_at < COMMIT_IDLE_S:
                    continue
                entries = self._pending.read()
                if entries:
                    await self._single_user_commit(entries)
                    # A paused or failed commit retries after another quiet spell.
                    self._last_activity_at = time.time()
            except Exception as e:
                logger.warning(f"[SINGLE] Commit loop error: {e!r}")

    async def _single_user_commit(self, entries: list[dict]):
        """Distil the pending exchanges into memory inside the live LLM session
        - two requests appended after the conversation, so its KV cache
        survives - then store them. All or nothing: if either request is
        cancelled or anything fails, the pending file is left as it is and a
        later quiet spell retries."""
        self._committing = True
        self._saves_pending += 1
        await self._refresh_indicator()
        t0 = time.monotonic()
        try:
            resp = await self.http_client.get(f"/memory/{SINGLE_USER_ID}/state", timeout=10.0)
            resp.raise_for_status()
            state = resp.json()
            prefix = self._assemble_context()
            facts_text = await self._single_user_llm(
                prefix + [{"role": "user", "content": facts_request(state.get("facts") or [], entries)}],
                max_tokens=800, abort_on_speech=True)
            summary_text = None
            if facts_text is not None:
                summary_text = await self._single_user_llm(
                    prefix + [{"role": "user", "content": summary_request(state.get("summary") or "", entries)}],
                    max_tokens=400, abort_on_speech=True)
            if facts_text is None or summary_text is None:
                logger.info(f"[SINGLE] Commit of {len(entries)} exchange(s) paused - the kiosk got busy")
                return
            resp = await self.http_client.post(f"/memory/{SINGLE_USER_ID}/commit", json={
                "exchanges": entries,
                "facts": parse_facts(facts_text),
                "summary": clean_llm_text(summary_text),
                "last_id": entries[-1]["id"],
            }, timeout=30.0)
            resp.raise_for_status()
            result = resp.json()
            self._pending.remove_through(entries[-1]["id"])
            self._single_user_has_memory = True
            logger.info(f"[SINGLE] Committed {len(entries)} exchange(s) to memory in "
                        f"{time.monotonic() - t0:.1f}s: {result}")
            await self._single_user_compact_if_needed()
        except Exception as e:
            logger.warning(f"[SINGLE] Commit failed after {time.monotonic() - t0:.1f}s: {e!r}")
        finally:
            self._committing = False
            self._saves_pending -= 1
            await self._refresh_indicator()

    def _single_user_record_exchange(self):
        """Append the exchange that just completed to the pending file."""
        h = self.conversation_history
        if len(h) < 2 or h[-1].get("role") != "assistant" or h[-1].get("_recorded"):
            return
        h[-1]["_recorded"] = True
        if h[-2].get("role") != "user" or h[-2].get("_note"):
            return
        user = self._strip_context_tags(h[-2]["content"]).strip()
        assistant = self._clean_for_tts(h[-1]["content"])
        if user and assistant:
            self._pending.append(user, assistant)

    async def _single_user_compact_if_needed(self):
        """Past COMPACT_AT_TOKENS, rebuild the live conversation from memory
        (everything is committed by now) plus the last 20 messages, then have
        the LLM read it during this quiet spell so no reply waits on it."""
        chars = (len(SYSTEM_PROMPT) + len(self._kb_block) + len(self._session_memory_block)
                 + sum(len(m["content"]) for m in self.conversation_history))
        if chars / 3.5 < COMPACT_AT_TOKENS or self._pending.read():
            return
        if not await self._load_single_user_memory():
            return
        before = len(self.conversation_history)
        tail = self.conversation_history[-20:]
        while tail and tail[0].get("role") != "user":
            tail = tail[1:]
        self.conversation_history = tail
        await self._single_user_llm(
            self._assemble_context() + [{"role": "user", "content": "[Reply with OK.]"}],
            max_tokens=1, abort_on_speech=True)
        logger.info(f"[SINGLE] Compacted the live conversation: {before} -> {len(tail)} messages")

    # ── Face Emotion Loop ──────────────────────────────────────────────────

    async def _run_face_emotion(self):
        """Run HSEmotion on face crops at ~10Hz, feed into MoodTracker."""
        loop = asyncio.get_event_loop()
        interval = 1.0 / self.face_tracker._target_fps

        def _detect():
            state = self.face_tracker.state
            if state.present and state.face_crop is not None:
                label, conf, _ = self.face_emotion.detect(state.face_crop)
                return label, conf
            return None, 0.0

        while self._running:
            try:
                label, conf = await loop.run_in_executor(None, _detect)
                if label:
                    self.current_face_emotion = label
                    self.current_face_emotion_prob = conf
                    self.mood_tracker.update_face_emotion(label, conf)
            except Exception as e:
                logger.warning(f"Face emotion error: {e}")

            await asyncio.sleep(interval)


# ── Entry Point ─────────────────────────────────────────────────────────────────

async def _tracemalloc_snapshot_loop(interval_s: int = 300):
    """Diagnostic-only: write a tracemalloc snapshot to server/diag/ every
    `interval_s` seconds. Gated on TRACEMALLOC env var (see main()). Used to
    find memory leaks: diff two snapshots with tracemalloc's Snapshot.compare_to()."""
    import pickle
    import tracemalloc
    from pathlib import Path

    diag_dir = Path(__file__).resolve().parent.parent / "diag"
    diag_dir.mkdir(exist_ok=True)
    t0 = time.monotonic()
    n = 0
    while True:
        await asyncio.sleep(interval_s)
        n += 1
        elapsed_min = int((time.monotonic() - t0) / 60)
        snap = tracemalloc.take_snapshot()
        path = diag_dir / f"snapshot_t{elapsed_min:04d}min.pkl"
        with open(path, "wb") as f:
            pickle.dump(snap, f)
        logger.info(f"[TRACEMALLOC] Snapshot #{n} written: {path}")


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if os.environ.get("TRACEMALLOC"):
        import tracemalloc
        tracemalloc.start(25)
        logger.info("[TRACEMALLOC] Enabled (25 frames per traceback) — periodic snapshots every 5 min")
        asyncio.create_task(_tracemalloc_snapshot_loop())

    orch = Orchestrator()
    code = 0
    try:
        await orch.start()
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Orchestrator failed")
        code = 1
    finally:
        await orch.stop()
        # Leave here, before asyncio winds down its worker threads: once speech
        # recognition has run, that wind-down ends in an uncaught C++ exception
        # (Windows Application log: KERNELBASE 0xe06d7363, then ucrtbase
        # 0xc0000409) and a "python.exe stopped working" box on the kiosk
        # screen. Everything is stopped by now.
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    import traceback
    try:
        asyncio.run(main())        # main() itself leaves via os._exit once stopped
    except BaseException:
        traceback.print_exc()      # failed before main() could stop and leave
        logging.shutdown()
        sys.stderr.flush()
        os._exit(1)
