"""
Inference server — FastAPI app serving the LLM, text-to-speech and audio emotion on this machine.

Lip sync runs inside Unreal (Runtime MetaHuman Lip Sync plugin).

Endpoints:
  POST /llm/generate    — streaming LLM proxy (disconnect-aware, barge-in safe)
  POST /llm/cancel      — abort in-flight LLM stream (safety net, cache preserved)
  POST /llm/end_session — explicitly end session and erase KV cache
  GET  /llm/session     — current session state (debug/health)
  POST /tts/synthesize  — Chatterbox Turbo voice-cloned audio + word timestamps
  POST /tts/cancel      — discard in-flight TTS result (barge-in)
  POST /a2e/detect      — emotion label + confidence from audio
  GET  /health          — liveness + service status + GPU stats + session info
  POST /telemetry/turn  — accept per-turn latency breakdown from the orchestrator
  GET  /telemetry/recent— last N turns as JSON
  GET  /dashboard       — web-based latency monitor
  Memory API            — /memory/{user_id}, /memory/users, etc.
"""

import asyncio
import base64
import collections
import configparser
import io
import json
import os
import subprocess
import re
import statistics
import time
import wave
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from llm_gateway import LLMGateway
from memory_compressor import MemoryCompressor
from memory_store import MemoryStore
from orchestrator.identity_mode import (
    IDENTITY_MODES, load_identity_config, running_mode_from_log, write_identity_mode,
)
from orchestrator.gaze_calculator import CONFIG_PATH as KIOSK_CONFIG
from orchestrator import pron_match
from orchestrator import pronunciation as pron
from orchestrator.knowledge import (
    KB_EXTENSIONS, KIOSK_PRONUNCIATIONS, PRONUNCIATIONS_FILE, Pronunciations, find_kb_folder, load_kb_config, read_kb,
    word_mentions,
)


# ─── Request / Response Models ────────────────────────────────────────────────

class LLMRequest(BaseModel):
    messages: list[dict] = Field(..., description="OpenAI-style messages array")
    session_id: str = Field(default="default", description="User session ID — cache stays warm within a session")
    max_tokens: int = Field(default=256)
    temperature: float = Field(default=0.7)
    stream: bool = Field(default=True)


class EndSessionRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Session to end (None = end any active session)")


class LLMResponse(BaseModel):
    text: str
    tokens_generated: int
    ttft_ms: float
    tps: float


class TTSRequest(BaseModel):
    text: str
    emotion_exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    voice_path: str | None = None  # override REFERENCE_VOICE for this call only


class WordTimestamp(BaseModel):
    word: str
    start_ms: int
    end_ms: int


class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int = 24000
    duration_ms: int
    word_timestamps: list[WordTimestamp]


class A2ERequest(BaseModel):
    audio_base64: str
    sample_rate: int = 22050


class A2EResponse(BaseModel):
    emotion: str
    confidence: float


class ServiceHealth(BaseModel):
    llm: bool = True
    tts: bool = True
    a2e: bool = True


class HealthResponse(BaseModel):
    status: str = "ok"
    services: ServiceHealth = ServiceHealth()
    gpu_memory_used_gb: float = 0.0
    gpu_memory_free_gb: float = 0.0
    uptime_s: float = 0.0
    llm_session: dict = {}


# ─── App Lifecycle ────────────────────────────────────────────────────────────

# ─── Telemetry ring buffer ────────────────────────────────────────────────────

_telemetry_turns: collections.deque = collections.deque(maxlen=50)
_conversation_turns: collections.deque = collections.deque(maxlen=100)
_telemetry_events: collections.deque = collections.deque(maxlen=400)

_start_time: float = 0.0
_tts_model = None
_a2e_model = None
_llm_gateway: LLMGateway | None = None
_memory_store: MemoryStore | None = None
_memory_compressor: MemoryCompressor | None = None
_tts_cancelled = False  # Barge-in flag: discard TTS result when set
_tts_voice: str | None = None  # the reference voice file the TTS conditioning was prepared from


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time, _tts_model, _a2e_model, _llm_gateway, _memory_store, _memory_compressor
    _start_time = time.monotonic()

    # Open memory store + compressor
    _memory_store = MemoryStore()
    _memory_store.open()
    _memory_compressor = MemoryCompressor(_memory_store)

    # Backfill user_name from pinned facts for any user with empty user_name
    _backfill_user_names()

    # Start LLM Gateway (session-aware proxy to llama-server)
    _llm_gateway = LLMGateway(
        llama_url=settings.LLAMA_SERVER_URL,
        idle_timeout_s=settings.LLM_SESSION_IDLE_TIMEOUT_S,
    )
    await _llm_gateway.start()

    # Load Chatterbox Turbo TTS
    print("Loading Chatterbox Turbo TTS...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    _tts_model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    _synthesize("warmup", settings.REFERENCE_VOICE)  # also prepares her voice once
    print("Chatterbox Turbo ready")

    # Load emotion2vec for NPC self-emotion detection
    print("Loading emotion2vec (A2E)...")
    from funasr import AutoModel
    _a2e_model = AutoModel(model="iic/emotion2vec_plus_base", hub="hf", device="cpu")
    print("emotion2vec ready")

    _names_queue_load()                    # names waiting from before a restart carry on
    if _names_waiting():
        _names_queue_start()
    print(f"Server starting on {settings.HOST}:{settings.PORT}")
    yield
    if _names_worker is not None:
        await _names_worker.close()
    if _memory_compressor:
        _memory_compressor.close()
    if _memory_store:
        _memory_store.close()
    if _llm_gateway:
        await _llm_gateway.stop()
    _tts_model = None
    _a2e_model = None
    print("Server shutting down")


app = FastAPI(
    title="Inference server",
    version="0.2.0",
    lifespan=lifespan,
)


# ─── GET /health ──────────────────────────────────────────────────────────────

def _gpu_stats() -> tuple[float, float]:
    """Query nvidia-smi for VRAM usage. Returns (used_gb, free_gb)."""
    try:
        out = subprocess.check_output(
            [settings.NVIDIA_SMI, "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        used_mb, free_mb = (float(x) for x in out.split(","))
        return round(used_mb / 1024, 2), round(free_mb / 1024, 2)
    except Exception:
        return 0.0, 0.0


@app.get("/health", response_model=HealthResponse)
async def health():
    used, free = _gpu_stats()
    return HealthResponse(
        gpu_memory_used_gb=used,
        gpu_memory_free_gb=free,
        uptime_s=round(time.monotonic() - _start_time, 1),
        llm_session=_llm_gateway.session_info() if _llm_gateway else {},
    )


# ─── POST /llm/generate ──────────────────────────────────────────────────────

@app.post("/llm/generate")
async def llm_generate(req: LLMRequest, request: Request):
    """
    Session-aware LLM proxy to llama-server.

    KV cache stays warm across all requests with the same session_id.
    Barge-in: client disconnect is detected automatically — generator
    exits cleanly, lock releases, cache preserved.
    """
    result = await _llm_gateway.generate(
        messages=req.messages,
        session_id=req.session_id,
        request=request,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=req.stream,
    )

    if req.stream:
        # Extract last user message (strip [MOOD:] and [CONTEXT ...] lines)
        user_text = ""
        for msg in reversed(req.messages):
            if msg.get("role") == "user":
                lines = msg["content"].split("\n")
                user_text = "\n".join(
                    l for l in lines
                    if not l.startswith("[MOOD:") and not l.startswith("[CONTEXT ")
                ).strip()
                break

        async def _capture_and_forward():
            """Wrap the SSE stream to capture assistant text for conversation log."""
            tokens = []
            meta = {}
            async for chunk in result:
                yield chunk
                # Parse SSE data lines to collect tokens
                for line in chunk.split("\n"):
                    if line.startswith("data: "):
                        try:
                            d = json.loads(line[6:])
                            if d.get("done"):
                                meta = d
                            elif "token" in d:
                                tokens.append(d["token"])
                        except (json.JSONDecodeError, KeyError):
                            pass
            # Log the completed turn
            if tokens:
                raw_text = "".join(tokens)
                from orchestrator.persona_prompt import strip_directives, _ANY_TAG_RE
                directives = _ANY_TAG_RE.findall(raw_text)
                spoken = strip_directives(raw_text)
                _conversation_turns.appendleft({
                    "timestamp": time.time(),
                    "session_id": req.session_id,
                    "user": user_text,
                    "assistant_raw": raw_text,
                    "assistant_spoken": spoken,
                    "directives": directives,
                    "tokens": meta.get("tokens_generated", len(tokens)),
                    "ttft_ms": meta.get("ttft_ms", 0),
                    "tps": meta.get("tps", 0),
                })

        return StreamingResponse(_capture_and_forward(), media_type="text/event-stream")
    return result


# ─── POST /llm/end_session ───────────────────────────────────────────────────

@app.post("/llm/end_session")
async def llm_end_session(req: EndSessionRequest = EndSessionRequest()):
    """
    Explicitly end the active LLM session and erase KV cache from VRAM.
    Call this when the user walks away or conversation ends.
    No cache is saved to disk — the memory store handles persistence.
    """
    return await _llm_gateway.end_session(req.session_id)


# ─── POST /llm/cancel ───────────────────────────────────────────────────────

@app.post("/llm/cancel")
async def llm_cancel():
    """
    Cancel in-flight LLM generation (barge-in).

    KV cache is preserved — the session stays warm for the next turn.
    The streaming response will emit a final SSE event with cancelled=true.
    Returns immediately (<1ms).
    """
    return await _llm_gateway.cancel()


# ─── GET /llm/session ────────────────────────────────────────────────────────

@app.get("/llm/session")
async def llm_session():
    """Current LLM session state — for debugging and dashboard."""
    return _llm_gateway.session_info()


# ─── POST /tts/synthesize ────────────────────────────────────────────────────

@app.post("/tts/cancel")
async def tts_cancel():
    """
    Signal TTS to discard its result on completion (barge-in).

    Since TTS runs synchronously on CUDA, the GPU computation finishes
    regardless. But the caller (the orchestrator) won't be blocked — it can close
    the HTTP connection. This flag ensures the server-side logs the discard.
    """
    global _tts_cancelled
    _tts_cancelled = True
    print("[TTS] Cancel flag set — next result will be discarded")
    return {"status": "cancelled"}


def _synthesize(text: str, voice: str):
    """One Chatterbox Turbo call. Her voice's conditioning is prepared once per
    reference file, not per call, and nothing records autograd history.
    Passing audio_prompt_path on every call re-runs prepare_conditionals (load,
    resample and embed the reference WAV) with gradients on. Measured: +6.7 MB
    of memory per sentence (+2.0 GB over 300 calls)."""
    global _tts_voice
    with torch.inference_mode():
        if voice != _tts_voice:
            # exaggeration/norm_loudness: the defaults generate() passes
            _tts_model.prepare_conditionals(voice, exaggeration=0.0, norm_loudness=True)
            _tts_voice = voice
        return _tts_model.generate(text=text)


@app.post("/tts/synthesize", response_model=TTSResponse)
async def tts_synthesize(req: TTSRequest):
    """
    Chatterbox Turbo TTS with voice clone from ReferenceVoice.wav.
    Returns 24kHz 16-bit mono PCM (base64) + estimated word timestamps.
    """
    global _tts_cancelled
    _tts_cancelled = False  # Clear flag at start of new synthesis
    t0 = time.monotonic()

    # Mark kiosk as active so deferred memory compression doesn't fire while
    # she is talking — covers identification flow (which uses TTS but not LLM).
    if _llm_gateway:
        _llm_gateway.bump_activity()

    # Run synthesis in thread pool to avoid blocking the event loop
    voice = req.voice_path or settings.REFERENCE_VOICE
    try:
        wav = await asyncio.get_event_loop().run_in_executor(None, lambda: _synthesize(req.text, voice))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

    # Check barge-in flag — discard result if cancelled during synthesis
    if _tts_cancelled:
        _tts_cancelled = False
        elapsed = time.monotonic() - t0
        print(f"[TTS] Result discarded (barge-in) after {elapsed*1000:.0f}ms")
        return TTSResponse(
            audio_base64="",
            sample_rate=24000,
            duration_ms=0,
            word_timestamps=[],
        )

    # Convert to 16-bit PCM bytes
    audio_np = wav.squeeze().cpu().numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()

    # Audio duration
    sample_rate = 24000  # Chatterbox output rate
    duration_ms = int(len(audio_np) / sample_rate * 1000)

    # Estimated word timestamps (evenly spaced, not aligned to the audio)
    words = req.text.split()
    timestamps = []
    if words:
        ms_per_word = duration_ms // len(words)
        cursor = 0
        for w in words:
            end = min(cursor + ms_per_word, duration_ms)
            timestamps.append(WordTimestamp(word=w, start_ms=cursor, end_ms=end))
            cursor += ms_per_word

    elapsed = time.monotonic() - t0
    print(f"[TTS] {duration_ms}ms audio in {elapsed*1000:.0f}ms (RTF: {elapsed/(duration_ms/1000):.2f}x)")

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
        word_timestamps=timestamps,
    )


# ─── POST /a2e/detect ────────────────────────────────────────────────────────

# emotion2vec label order (emotion2vec_plus_base outputs 9 classes in this order)
_A2E_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]


@app.post("/a2e/detect", response_model=A2EResponse)
async def a2e_detect(req: A2ERequest):
    """
    emotion2vec+ NPC self-emotion detection from TTS audio output.
    Classifies what emotion the NPC's synthesized speech conveys.
    """
    t0 = time.monotonic()

    # Decode 16-bit PCM to float32 numpy array
    audio_bytes = base64.b64decode(req.audio_base64)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    try:
        # Pass numpy array directly (avoids torchaudio/ffmpeg file loading issues)
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _a2e_model.generate(
                audio_f32,
                granularity="utterance",
                extract_embedding=False,
                fs=req.sample_rate,
            ),
        )

        scores = result[0]["scores"]
        top_idx = scores.index(max(scores))
        emotion = _A2E_LABELS[top_idx]
        confidence = round(scores[top_idx], 3)

        elapsed = time.monotonic() - t0
        print(f"[A2E] {emotion} ({confidence}) in {elapsed*1000:.0f}ms")

        return A2EResponse(emotion=emotion, confidence=confidence)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return A2EResponse(emotion="neutral", confidence=0.5)


# ─── Memory API ──────────────────────────────────────────────────────────────


class FactRequest(BaseModel):
    fact: str
    pinned: bool = True
    user_name: str | None = None  # Optional: set user display name alongside fact


class ExtractFactsRequest(BaseModel):
    text: str = Field(..., description="Latest user message")
    response: str = Field(..., description="Latest assistant response")


class SingleUserCommitRequest(BaseModel):
    exchanges: list[dict]   # pending-file entries: {"id", "t", "user", "assistant"}
    facts: list[str]        # the complete rewritten fact list
    summary: str = ""       # the updated running summary
    last_id: int            # id of the newest exchange in this commit


class SessionEndRequest(BaseModel):
    transcript: list[dict]
    # True = respond only once the session is compressed into memory (the
    # orchestrator keeps the lapel pin purple until then). False = respond as
    # soon as the transcript is stored and compress in the background.
    wait: bool = False


_NAME_IN_FACT_RE = re.compile(
    r"(?:user(?:'s)?|their)\s+name\s+is\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+)?)",
    re.IGNORECASE,
)


def _maybe_set_name_from_fact(user_id: str, fact: str) -> str | None:
    """If the fact matches 'User's name is X' and user_name is empty, set it.

    Returns the name extracted, or None if no update made.
    """
    m = _NAME_IN_FACT_RE.search(fact)
    if not m:
        return None
    user = _memory_store.get_or_create_user(user_id)
    if user.user_name:  # Already set, don't overwrite
        return None
    name = m.group(1).strip().rstrip(".,;")
    _memory_store.update_user_name(user_id, name)
    print(f"[Name] Auto-set user_name='{name}' for user {user_id} from fact")
    return name


def _backfill_user_names() -> int:
    """Populate user_name field for users whose name is only in pinned facts.

    Pinned facts often contain phrases like 'User's name is Alex.' — this scans those
    and updates the users.user_name column accordingly. Runs once on server startup.
    """
    updated = 0
    try:
        for u in _memory_store.list_users():
            if u.user_name:
                continue
            for fact in _memory_store.get_all_facts(u.user_id):
                m = _NAME_IN_FACT_RE.search(fact)
                if m:
                    name = m.group(1).strip().rstrip(".,;")
                    _memory_store.update_user_name(u.user_id, name)
                    updated += 1
                    print(f"[Backfill] Set user_name='{name}' for user {u.user_id}")
                    break
    except Exception as e:
        print(f"[Backfill] Failed: {e}")
    if updated:
        print(f"[Backfill] Updated {updated} user names from pinned facts")
    return updated


@app.post("/memory/backfill_names")
async def memory_backfill_names():
    """Manually trigger user_name backfill from pinned facts."""
    return {"updated": _backfill_user_names()}


@app.get("/memory/users")
async def memory_list_users():
    """List all known users."""
    users = _memory_store.list_users()
    return [
        {
            "user_id": u.user_id,
            "user_name": u.user_name,
            "session_count": u.session_count,
            "last_seen": u.last_seen,
        }
        for u in users
    ]


@app.get("/memory/{user_id}")
async def memory_get(user_id: str):
    """Assemble memory context for a user. Returns all tiers within 4K token budget."""
    ctx = _memory_store.assemble_context(user_id)
    return {
        "user_id": ctx.user_id,
        "user_name": ctx.user_name,
        "context_text": ctx.context_text,
        "token_count": ctx.token_count,
        "tiers": ctx.tiers,
    }


@app.get("/memory/{user_id}/detail")
async def memory_detail(user_id: str):
    """Full memory detail for dashboard — all tiers with individual entries."""
    user = _memory_store.get_or_create_user(user_id)
    tier_order = ["facts", "verbatim", "light", "medium", "cross_session"]
    tiers = {}
    total_tokens = 0
    for tier in tier_order:
        entries = _memory_store.get_tier_entries(user_id, tier)
        tiers[tier] = [
            {
                "content": e.content,
                "token_count": e.token_count,
                "pinned": e.pinned,
                "created_at": e.created_at,
                "updated_at": e.updated_at,
            }
            for e in entries
        ]
        total_tokens += sum(e.token_count for e in entries)
    return {
        "user_id": user_id,
        "user_name": user.user_name,
        "session_count": user.session_count,
        "tiers": tiers,
        "total_tokens": total_tokens,
    }


@app.delete("/memory/{user_id}")
async def memory_delete_user(user_id: str):
    """Delete a user and all their memory. Cannot be undone."""
    _memory_store.delete_user(user_id)
    print(f"[Memory] Deleted user '{user_id}' and all memory")
    return {"status": "ok", "deleted": user_id}


@app.put("/memory/{user_id}/facts")
async def memory_add_fact(user_id: str, req: FactRequest):
    """Add a pinned fact for a user (e.g., name, preference)."""
    _memory_store.get_or_create_user(user_id)
    if req.user_name:
        _memory_store.update_user_name(user_id, req.user_name)
    _memory_store.add_fact(user_id, req.fact, req.pinned)
    # Auto-populate user_name from the fact if it matches the name pattern
    # and user_name wasn't set explicitly
    if not req.user_name:
        _maybe_set_name_from_fact(user_id, req.fact)
    return {"status": "ok", "fact": req.fact, "pinned": req.pinned, "user_name": req.user_name}


@app.post("/memory/{user_id}/extract_facts")
async def memory_extract_facts(user_id: str, req: ExtractFactsRequest):
    """
    Extract pinned facts from the latest conversation turn.

    Call this after each completed turn. Runs LLM extraction in background
    and pins any new facts (names, preferences, commitments) to the memory
    store. Deduplicates against existing facts automatically.

    Zero pipeline latency — fire-and-forget from orchestrator.
    """
    _memory_store.get_or_create_user(user_id)

    async def _extract():
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _memory_compressor.extract_facts_from_turn(
                user_id, req.text, req.response,
            ),
        )
        if result.get("extracted", 0) > 0:
            print(f"[Memory] Extracted {result['extracted']} facts for '{user_id}': {result.get('facts', [])}")
            # If any extracted fact is a name fact, auto-populate user_name
            for fact in result.get("facts", []):
                if _maybe_set_name_from_fact(user_id, fact):
                    break

    asyncio.create_task(_extract())

    return {"status": "ok", "extraction": "running_async", "user_id": user_id}


# Deferred-compression timing. Compression makes up to 4 sequential LLM calls
# (light, medium, fact reconcile, session summary; ~5 s in total on Gemma 4)
# through slot 0 — the same slot the conversation uses — so it waits until
# the kiosk has been idle (no LLM calls, no TTS) for COMPRESS_IDLE_S
# before firing. HARD_CAP prevents indefinite deferral on a busy kiosk.
COMPRESS_IDLE_S = 15.0
COMPRESS_HARD_CAP_S = 5 * 60.0
COMPRESS_POLL_S = 2.0


async def _deferred_compress(user_id: str):
    """Wait for kiosk idle, then compress the user's stored transcripts.
    Returns the compressor's result (None if the gateway is down)."""
    deadline = time.monotonic() + COMPRESS_HARD_CAP_S
    while True:
        if _llm_gateway is None:
            return
        idle_for = _llm_gateway.seconds_since_activity()
        if idle_for >= COMPRESS_IDLE_S:
            break
        if time.monotonic() > deadline:
            print(f"[Memory] Deferred-compress hard-cap expired for '{user_id}' "
                  f"(idle_for={idle_for:.1f}s); firing despite activity")
            break
        # Sleep until either the threshold could be met or POLL_S, whichever shorter
        sleep_for = min(COMPRESS_IDLE_S - idle_for, COMPRESS_POLL_S)
        await asyncio.sleep(max(sleep_for, 0.5))

    print(f"[Memory] Firing deferred compress for '{user_id}' "
          f"(idle_for={_llm_gateway.seconds_since_activity():.1f}s)")
    t0 = time.monotonic()
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _memory_compressor.compress_session(user_id)
    )
    print(f"[Memory] Compressed session for '{user_id}' in {time.monotonic() - t0:.1f}s: {result}")
    return result


@app.post("/memory/{user_id}/session_end")
async def memory_session_end(user_id: str, req: SessionEndRequest):
    """
    Store session transcript and schedule deferred compression.

    Transcript storage is synchronous and durable (SQLite write completes
    before this endpoint returns). Compression is deferred until the
    kiosk has been idle for COMPRESS_IDLE_S so its LLM calls don't
    contend with the next user's conversation on slot 0. With
    req.wait the response comes back only once compression is done.

    Directives are scrubbed from assistant turns automatically.
    """
    _memory_store.get_or_create_user(user_id)
    _memory_store.store_transcript(user_id, req.transcript)

    task = asyncio.create_task(_deferred_compress(user_id))
    if req.wait:
        # shield(): the save finishes even if the caller gives up or
        # disconnects while waiting.
        result = await asyncio.shield(task)
        return {
            "status": "ok",
            "turns_stored": len(req.transcript),
            "compression": "done",
            "result": result,
            "user_id": user_id,
        }

    return {
        "status": "ok",
        "turns_stored": len(req.transcript),
        "compression": "deferred_pending_idle",
        "compress_idle_threshold_s": COMPRESS_IDLE_S,
        "user_id": user_id,
    }


@app.get("/memory/{user_id}/state")
async def memory_state(user_id: str):
    """Raw memory for single-user mode: facts, running summary, recent exchanges."""
    return {"user_id": user_id, **_memory_store.get_state(user_id)}


@app.post("/memory/{user_id}/commit")
async def memory_commit(user_id: str, req: SingleUserCommitRequest):
    """Single-user mode: store exchanges plus the facts and running summary
    the orchestrator distilled in its live LLM session - no LLM calls here.
    Multi-user never calls this. Idempotent on last_id."""
    result = _memory_store.commit_single_user(
        user_id, req.exchanges, req.facts, req.summary, req.last_id
    )
    print(f"[Memory] Single-user commit for '{user_id}': {result}")
    return result


@app.post("/memory/{user_id}/compress")
async def memory_compress(user_id: str):
    """Manually trigger compression for a user's uncompressed transcripts."""
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _memory_compressor.compress_session(user_id)
    )
    return result


@app.post("/memory/{user_id}/consolidate_facts")
async def memory_consolidate_facts(user_id: str):
    """Manually trigger LLM-based fact dedup/consolidation for a user."""
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _memory_compressor.consolidate_facts(user_id)
    )
    return result


@app.post("/memory/{user_id}/greeting")
async def memory_greeting(user_id: str, request: Request):
    """
    Generate a personalized greeting for a returning user.

    Loads user memory, sends it as context to the LLM, and returns
    the avatar's greeting ready for TTS.

    For new users (no memory), returns a generic greeting.
    """
    from orchestrator.persona_prompt import SYSTEM_PROMPT

    # Load memory context
    ctx = _memory_store.assemble_context(user_id)
    user = _memory_store.get_or_create_user(user_id)

    if not ctx.context_text:
        # New user — no memory, return generic greeting
        return {
            "user_id": user_id,
            "is_returning": False,
            "greeting": "<<emotion:happy intensity:0.8>><<gesture:wave>> Hey there! Welcome! <<emotion:curious intensity:0.5>><<gesture:tilt_head>> What can I do for you today?",
            "session_id": None,
        }

    # Returning user — build briefing prompt with memory
    briefing = (
        f"A returning visitor has arrived. Here is what you remember about them:\n\n"
        f"{ctx.context_text}\n\n"
        f"Greet them warmly by name (if known) and reference something specific "
        f"from your shared history. Keep it to 1-2 sentences. Be natural — "
        f"don't list facts, just weave one detail into a warm welcome.\n"
        f"HARD RULES: If any remembered fact is a topic boundary (the user asked "
        f"to stop discussing something or said it's settled), NEVER mention or "
        f"ask about that topic — they'll raise it themselves if they want. And "
        f"don't open with the same remembered detail every visit; a fresh "
        f"question about their day beats recycling one interest."
    )

    # Generate greeting via LLM Gateway
    session_id = f"greeting-{user_id}-{int(time.time())}"
    result = await _llm_gateway.generate(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": briefing},
        ],
        session_id=session_id,
        request=request,
        max_tokens=100,
        temperature=0.8,
        stream=False,
    )

    greeting_text = result.get("text", "")
    _memory_store.increment_session_count(user_id)

    return {
        "user_id": user_id,
        "is_returning": True,
        "user_name": user.user_name,
        "greeting": greeting_text,
        "session_id": session_id,
        # Compressed memory text so the orchestrator can establish it ONCE in the
        # session system prompt instead of re-injecting it per turn.
        "memory_context": ctx.context_text,
        "memory_tokens": ctx.token_count,
        "ttft_ms": result.get("ttft_ms", 0),
    }


# ─── Telemetry / Dashboard ────────────────────────────────────────────────────


class TurnTelemetry(BaseModel):
    """Per-turn pipeline timings. timings is a flexible dict so the dashboard
    schema can iterate without requiring server-side changes for each new
    field. Known keys:
        speech_duration_ms, vad_dispatch_ms, asr_processing_ms,
        llm_client_ms, llm_ttft_ms, llm_first_sentence_ms, llm_total_ms,
        llm_tokens, llm_tps, tts_ms, ws_speak_send_ms,
        ttfa_ms, total_pipeline_ms, utterance_tail_ms, unaccounted_ms
    Legacy keys (still accepted):
        vad_endpoint_ms, asr_ms, network_to_server_ms,
        audio_load_ms, total_e2e_ms
    """
    model_config = {"extra": "allow"}
    turn_id: str = ""
    session_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    timings: dict[str, Any] = Field(default_factory=dict)
    transcript: str = ""           # optional: user utterance ASR
    response: str = ""             # optional: avatar response (raw or cleaned)
    ctx: dict[str, Any] = Field(default_factory=dict)
    # ctx may include: id (str), id_sim (float), gaze ([float]), face_emo ([str]),
    #                  voice_emo (str), mood_from (str), mood_to (str),
    #                  avatar_emo (str), avatar_int (float)


class EventTelemetry(BaseModel):
    """Single event in a session timeline. component ∈ {camera, recog, userid,
    ttsprompt, asrpin, mood, ...}. kind is component-specific."""
    model_config = {"extra": "allow"}
    event_id: str = ""
    session_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    component: str
    kind: str
    text: str = ""                  # human-readable event description
    data: dict[str, Any] = Field(default_factory=dict)


@app.post("/telemetry/turn")
async def telemetry_report_turn(turn: TurnTelemetry):
    """Per-turn pipeline timings. Schema is flexible — see TurnTelemetry."""
    _telemetry_turns.appendleft(turn.model_dump())
    return {"status": "ok", "turns_stored": len(_telemetry_turns)}


@app.post("/telemetry/event")
async def telemetry_report_event(event: EventTelemetry):
    """Identity-flow + multimodal events outside the per-turn pipeline.
    Used by the dashboard to render Type-B identity rows + Type-C context strips."""
    _telemetry_events.appendleft(event.model_dump())
    return {"status": "ok", "events_stored": len(_telemetry_events)}


@app.get("/telemetry/recent")
async def telemetry_recent(limit: int = 50):
    """Return the most recent turns for the dashboard."""
    turns = list(_telemetry_turns)[:limit]
    return {"turns": turns, "count": len(turns)}


@app.get("/telemetry/events/recent")
async def telemetry_events_recent(limit: int = 200):
    """Return recent identity/multimodal events for the dashboard."""
    events = list(_telemetry_events)[:limit]
    return {"events": events, "count": len(events)}


@app.delete("/telemetry/clear")
async def telemetry_clear():
    """Clear all stored telemetry data (turns + events)."""
    _telemetry_turns.clear()
    _telemetry_events.clear()
    return {"status": "ok", "turns_stored": 0, "events_stored": 0}


@app.get("/conversation/recent")
async def conversation_recent(limit: int = 50):
    """Return recent conversation turns for the dashboard."""
    turns = list(_conversation_turns)[:limit]
    return {"turns": turns, "count": len(turns)}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the latency monitoring dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# ─── Admin / Ops Dashboard ────────────────────────────────────────────────────

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the system-health / ops admin dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "admin.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


async def _probe_tcp(host: str, port: int, timeout_s: float = 1.0) -> bool:
    """Probe whether something is listening on host:port via TCP connect."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout_s,
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False


async def _probe_http(url: str, timeout_s: float = 1.0) -> bool:
    """Probe whether an HTTP endpoint returns 200."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as c:
            r = await c.get(url)
            return r.status_code == 200
    except Exception:
        return False


def _pid_listening_on(port: int) -> int | None:
    """Get the PID of the process listening on the given port, or None.
    Best-effort — falls back gracefully if psutil is unavailable."""
    try:
        import psutil
        for c in psutil.net_connections(kind='inet'):
            if c.status == 'LISTEN' and c.laddr.port == port:
                return c.pid
    except Exception:
        pass
    return None


@app.get("/admin/status")
async def admin_status():
    """Per-component state for the admin dashboard. Probes run in parallel."""
    llama_task = asyncio.create_task(_probe_http("http://127.0.0.1:8080/health"))
    llama_up = await llama_task
    # Orchestrator liveness via psutil listen-check (no socket connect). A bare TCP
    # probe (_probe_tcp) makes the orchestrator's websockets server log a benign
    # "handshake failed" traceback on every poll. Reuse the pid below.
    orch_pid = _pid_listening_on(8765)
    orch_up = orch_pid is not None

    components = [
        {
            "name": "llama-server",
            "key": "llama",
            "port": 8080,
            "state": "up" if llama_up else "down",
            "pid": _pid_listening_on(8080) if llama_up else None,
        },
        {
            "name": "FastAPI",
            "key": "fastapi",
            "port": 8100,
            "state": "up",  # this server is answering the request, so it is up
            "pid": _pid_listening_on(8100),
        },
        {
            "name": "orchestrator",
            "key": "orchestrator",
            "port": 8765,
            "state": "up" if orch_up else "down",
            "pid": orch_pid,
        },
    ]
    ue_proc = _ue_process()
    components.append({
        "name": "UE editor",
        "key": "ue",
        "port": None,
        "state": "up" if ue_proc else "down",
        "pid": ue_proc.pid if ue_proc else None,
    })

    # Devices: enumeration only, never opens a stream, so it cannot disturb the
    # orchestrator's live mic.
    audio = _audio_bindings()
    cam = _camera_state()
    devices = [
        {
            "name": "Jabra (mic + speaker)",
            "state": "up" if audio["ok"] else "warn",
            "detail": audio["detail"] or f"in: {audio['input']} / out: {audio['output']}",
        },
        {"name": "Camera (C920)", "state": cam["state"], "detail": cam["detail"]},
    ]

    return {
        "components": components,
        "devices": devices,
        "identity": _identity_status(orch_up),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _identity_status(orch_up: bool) -> dict:
    """Single-/multi-user mode: what kiosk_config.ini asks for, and what the
    running orchestrator actually started in (from the head of its log,
    which the launcher truncates at every start). running is None while the
    orchestrator is down or still starting."""
    running = None
    if orch_up:
        path = os.path.join(os.path.dirname(__file__), "logs", "orchestrator.err.log")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                head = [f.readline() for _ in range(80)]
            running = running_mode_from_log(head)
        except OSError:
            pass
    return {"configured": load_identity_config().mode, "running": running}


def _spawn_detached_bat(bat_name: str) -> dict:
    """Spawn a .bat from server/scripts/ as a fully detached process so it
    survives FastAPI itself being killed (which the bat may do).

    Spawning is done via PowerShell's Start-Process -WindowStyle Hidden so
    the .bat runs in a hidden console; child PowerShell calls inside the
    .bat inherit that console (no per-call window pop-ups). Component
    windows launched explicitly via `start "title" /MIN` inside start.bat
    still appear as minimized — and only those — which is the intended UX.
    """
    bat_path = os.path.join(_SCRIPTS_DIR, bat_name)
    if not os.path.isfile(bat_path):
        return {"status": "error", "action": "not_spawned",
                "detail": f"missing: {bat_path}"}
    try:
        # The wrapper PowerShell process exits immediately after Start-Process
        # spawns the .bat. CREATE_NEW_PROCESS_GROUP keeps the spawned tree from
        # being torn down by Ctrl+C signals to FastAPI. STARTUPINFO hides the
        # wrapper's own console too, so nothing flashes on screen.
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
        subprocess.Popen(
            [
                "powershell.exe", "-NoProfile", "-Command",
                f"Start-Process -FilePath '{bat_path}' -WindowStyle Hidden",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=0x00000200,  # CREATE_NEW_PROCESS_GROUP
            startupinfo=startupinfo,
            close_fds=True,
        )
        return {"status": "ok", "action": f"{bat_name}_spawned"}
    except Exception as e:
        return {"status": "error", "action": "spawn_failed", "detail": str(e)}


@app.post("/admin/start_all")
async def admin_start_all():
    """Spawn start.bat detached. Re-launches any down components; no-op for up ones."""
    return _spawn_detached_bat("start.bat")


@app.post("/admin/stop_all")
async def admin_stop_all():
    """Spawn stop.bat detached. Stops in order: orchestrator (graceful WS) ->
    llama -> 3s delay (dashboard observability) -> FastAPI itself."""
    return _spawn_detached_bat("stop.bat")


@app.post("/admin/restart_all")
async def admin_restart_all():
    """Spawn restart.bat detached — stop.bat + 2s buffer + start.bat."""
    return _spawn_detached_bat("restart.bat")


# ─── Per-component control ────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UPROJECT = os.path.join(_PROJECT_ROOT, "unreal", "KioskAvatar.uproject")


def _resolve_ue_editor() -> str | None:
    """Locate UnrealEditor.exe WITHOUT hardcoding an engine version.

    UE ships multiple releases a year, so a literal ...\\UE_5.7\\... path breaks
    on the next upgrade. Resolution order:
      1. UE_EDITOR_EXE environment variable, if set
      2. the project's own EngineAssociation -> registry InstalledDirectory
      3. newest C:\\Program Files\\Epic Games\\UE_* on disk
    """
    override = os.environ.get("UE_EDITOR_EXE", "").strip()
    if override and os.path.isfile(override):
        return override

    rel = os.path.join("Engine", "Binaries", "Win64", "UnrealEditor.exe")

    version = None
    try:
        with open(_UPROJECT, "r", encoding="utf-8") as f:
            version = (json.load(f).get("EngineAssociation") or "").strip()
    except Exception:
        pass

    if version:
        try:
            import winreg
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    key = winreg.OpenKey(
                        hive, rf"SOFTWARE\EpicGames\Unreal Engine\{version}")
                    root, _ = winreg.QueryValueEx(key, "InstalledDirectory")
                    cand = os.path.join(root, rel)
                    if os.path.isfile(cand):
                        return cand
                except OSError:
                    continue
        except Exception:
            pass

    import glob
    cands = sorted(glob.glob(r"C:\Program Files\Epic Games\UE_*"), reverse=True)
    if version:
        cands.sort(key=lambda p: (version not in p, p), reverse=False)
    for root in cands:
        cand = os.path.join(root, rel)
        if os.path.isfile(cand):
            return cand
    return None


def _ue_process():
    """The UE process running THIS project, matched by .uproject in the command
    line — robust to Epic renaming the executable, and it will not touch a
    different project the user happens to have open."""
    try:
        import psutil
        target = os.path.normcase(_UPROJECT)
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or [])
                if target in os.path.normcase(cmd):
                    return p
            except Exception:
                continue
    except Exception:
        pass
    return None


def _audio_bindings() -> dict:
    """Is the Jabra actually bound for BOTH capture and playback?

    If the Jabra is unpowered at boot, the orchestrator silently falls back to
    another mic and Windows moves default playback elsewhere. That disables the
    hardware echo cancellation, and the avatar barges in on her own voice on
    every line.
    """
    want = "jabra"
    out = {"input": None, "output": None, "ok": False, "detail": ""}
    try:
        import sounddevice as sd
        din, dout = sd.default.device[0], sd.default.device[1]
        devs = sd.query_devices()
        in_name = devs[din]["name"] if din is not None and din >= 0 else ""
        out_name = devs[dout]["name"] if dout is not None and dout >= 0 else ""
        out["input"], out["output"] = in_name, out_name
        in_ok = want in in_name.lower()
        out_ok = want in out_name.lower()
        out["ok"] = in_ok and out_ok
        if not in_ok and not out_ok:
            out["detail"] = "Jabra bound for neither capture nor playback"
        elif not in_ok:
            out["detail"] = "capture is not the Jabra"
        elif not out_ok:
            out["detail"] = "playback is not the Jabra — echo cancellation is OFF"
    except Exception as e:
        out["detail"] = f"probe failed: {e}"
    return out


def _camera_state() -> dict:
    """Last camera lifecycle event from the orchestrator log.

    Deliberately log-derived: actually opening the device to test it would
    fight the orchestrator's live capture and can wedge the C920 in Media
    Foundation. A true frames-flowing signal would need an orchestrator-side
    telemetry field.
    """
    out = {"state": "unknown", "detail": "no orchestrator log"}
    path = os.path.join(os.path.dirname(__file__), "logs", "orchestrator.err.log")
    try:
        size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(max(0, size - 400_000))
            tail = f.read()
        last = None
        for line in tail.splitlines():
            if "Camera opened" in line or "Camera released" in line or "camera" in line.lower() and "error" in line.lower():
                last = line
        if last is None:
            out["detail"] = "no camera events in recent log"
        elif "Camera opened" in last:
            out["state"] = "up"
            out["detail"] = last.strip()[:120]
        else:
            out["state"] = "down"
            out["detail"] = last.strip()[:120]
    except Exception as e:
        out["detail"] = f"log read failed: {e}"
    return out


def _kill_process(proc, timeout_s: float = 8.0) -> str:
    try:
        proc.terminate()
        proc.wait(timeout=timeout_s)
        return "terminated"
    except Exception:
        try:
            proc.kill()
            return "killed"
        except Exception as e:
            return f"kill failed: {e}"


_COMPONENT_BATS = {
    "llama": "start-llama.bat",
    "orchestrator": "start-orchestrator.bat",
    "fastapi": "start-fastapi.bat",
}


@app.post("/admin/start/{component}")
async def admin_start_component(component: str):
    component = component.lower()
    if component == "ue":
        exe = _resolve_ue_editor()
        if not exe:
            return {"status": "error", "detail": "UnrealEditor.exe not found"}
        if _ue_process():
            return {"status": "ok", "action": "already_running"}
        try:
            subprocess.Popen([exe, _UPROJECT], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=0x00000208, close_fds=True)
            return {"status": "ok", "action": "ue_launched", "detail": exe}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    bat = _COMPONENT_BATS.get(component)
    if not bat:
        return {"status": "error", "detail": f"unknown component '{component}'"}
    return _spawn_detached_bat(bat)


@app.post("/admin/stop/{component}")
async def admin_stop_component(component: str):
    component = component.lower()
    if component == "ue":
        proc = _ue_process()
        if not proc:
            return {"status": "ok", "action": "not_running"}
        return {"status": "ok", "action": _kill_process(proc)}

    if component == "orchestrator":
        # Graceful WS shutdown FIRST — force-killing while the orchestrator
        # holds the C920 wedges Windows Media Foundation until a USB replug.
        # "Stopped" means the PROCESS is gone, not the port: the port frees in
        # under a second, but the camera is released ~10 s later and the
        # process exits ~20 s later (measured). A restart that waits only for
        # the port opens a camera the exiting process still holds.
        pid = _pid_listening_on(8765)
        if pid is None:
            return {"status": "ok", "action": "not_running"}
        try:
            import websockets
            async with websockets.connect("ws://127.0.0.1:8765", open_timeout=3) as ws:
                await ws.send(json.dumps({"type": "shutdown"}))
        except Exception:
            pass
        try:
            import psutil
            proc = psutil.Process(pid)
        except Exception:
            return {"status": "ok", "action": "graceful_shutdown"}  # already exited
        for _ in range(90):  # up to 45 s for the graceful exit
            if not proc.is_running():
                return {"status": "ok", "action": "graceful_shutdown"}
            await asyncio.sleep(0.5)
        return {"status": "ok", "action": _kill_process(proc)}

    port = {"llama": 8080, "fastapi": 8100}.get(component)
    if port is None:
        return {"status": "error", "detail": f"unknown component '{component}'"}
    if component == "fastapi":
        return {"status": "error",
                "detail": "FastAPI serves this page — use stop_all instead"}
    pid = _pid_listening_on(port)
    if pid is None:
        return {"status": "ok", "action": "not_running"}
    try:
        import psutil
        return {"status": "ok", "action": _kill_process(psutil.Process(pid))}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/admin/restart/{component}")
async def admin_restart_component(component: str):
    stopped = await admin_stop_component(component)
    if stopped.get("status") == "error":
        return stopped
    await asyncio.sleep(3.0)
    started = await admin_start_component(component)
    return {"status": started.get("status", "ok"),
            "stop": stopped.get("action"), "start": started.get("action"),
            "detail": started.get("detail", "")}


class IdentityModeRequest(BaseModel):
    mode: str   # "multi_user" | "single_user"


@app.post("/admin/identity_mode")
async def admin_identity_mode(req: IdentityModeRequest):
    """Switch single-/multi-user mode from the admin page: set [identity]
    mode in kiosk_config.ini, then restart the orchestrator, which reads it
    at startup. If the orchestrator is down, the new mode applies at its
    next start."""
    mode = req.mode.strip().lower()
    if mode not in IDENTITY_MODES:
        return {"status": "error", "detail": f"unknown mode '{req.mode}'"}
    try:
        write_identity_mode(mode)
    except Exception as e:
        return {"status": "error", "detail": f"could not update kiosk_config.ini: {e}"}
    if _pid_listening_on(8765) is None:
        return {"status": "ok", "mode": mode, "action": "config_updated",
                "detail": "orchestrator is down; the new mode applies when it starts"}
    restart = await admin_restart_component("orchestrator")
    return {"status": restart.get("status", "ok"), "mode": mode, "action": "orchestrator_restarted",
            "detail": restart.get("detail", "")}


# ─── Names: settle pronunciations by hand (/admin/names) ─────────────────────
# The pronunciation check (orchestrator/pronunciation.py) fixes a name on its
# own only when the dictionary and the LLM agree; the rest waits here for a
# person: hear her say it, accept a proposal, type a respelling, or say the
# name - the recording becomes the target her voice is fitted to. No LLM
# calls from here: an unrelated prompt can trip llama-server's LRU abort,
# so respellings for a recording come from its sounds.

NAMES_WORKER_IDLE_S = 300           # the recognizer exits after this long unused
_names_worker = None                # pron.PhonemeWorker, started on first use
_names_worker_last = 0.0
_names_lock = asyncio.Lock()        # one recognizer job at a time


def _kb_folder() -> str | None:
    enabled, configured = load_kb_config()
    return find_kb_folder(configured) if enabled else None


def _names_edit_target() -> str:
    """Where accepted respellings go: the KB's own pronunciations.txt when
    its drive is there and writable (it travels with the KB), else the
    kiosk's."""
    folder = _kb_folder()
    if folder and os.access(folder, os.W_OK):
        return os.path.join(folder, PRONUNCIATIONS_FILE)
    return KIOSK_PRONUNCIATIONS


def _names_settled() -> dict:
    """{word (lowercase): (name, respelling, file)} from the hand-written files."""
    folder = _kb_folder()
    out = {}
    for path in [KIOSK_PRONUNCIATIONS] + ([os.path.join(folder, PRONUNCIATIONS_FILE)] if folder else []):
        for name, say in Pronunciations.load(path).table.items():
            out[name.lower()] = (name, say, path)
    return out


def _pcm16(wav) -> np.ndarray:
    return (wav.squeeze().cpu().numpy() * 32767).clip(-32768, 32767).astype(np.int16)


def _wav_file_bytes(pcm: np.ndarray, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _uploaded_wav(data: bytes) -> tuple:
    """(base64 16-bit mono PCM, sample rate) from the page's WAV recording."""
    with wave.open(io.BytesIO(data)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("expected a 16-bit WAV")
    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode(), rate


async def _names_hear(audio_b64: str, rate: int) -> list:
    """Phonemes in a clip, from the recognizer subprocess (started on first
    use, closed after NAMES_WORKER_IDLE_S unused)."""
    global _names_worker, _names_worker_last
    if _names_worker is None:
        _names_worker = pron.PhonemeWorker(
            device="cuda:1", log_path=os.path.join(os.path.dirname(__file__), "logs", "pronunciation_worker_admin.log"))
        asyncio.create_task(_names_worker_reaper())
    _names_worker_last = time.monotonic()
    try:
        return await _names_worker.hear(audio_b64, rate)
    except Exception:
        worker, _names_worker = _names_worker, None     # start afresh next time
        await worker.close()
        raise


async def _names_worker_reaper():
    global _names_worker
    while _names_worker is not None:
        await asyncio.sleep(30)
        if time.monotonic() - _names_worker_last > NAMES_WORKER_IDLE_S and not _names_lock.locked():
            worker, _names_worker = _names_worker, None
            await worker.close()


async def _names_say_and_hear(text: str) -> list:
    """Her voice says the text; the recognizer writes down the sounds."""
    wav = await asyncio.get_event_loop().run_in_executor(None, lambda: _synthesize(text, settings.REFERENCE_VOICE))
    return await _names_hear(base64.b64encode(_pcm16(wav).tobytes()).decode(), 24000)


def _names_result(spelling: str, rated) -> dict:
    stats, _, renderings = rated
    return {"spelling": spelling, "right": (stats or {}).get("right", 0), "score": (stats or {}).get("score"),
            "says": list(dict.fromkeys(pron.respell_ipa(r.split()) for r in renderings if r))}


NAMES_RECORDINGS = os.path.join(os.path.dirname(__file__), "data", "names_recordings")


def _names_recording_path(word: str) -> str:
    return os.path.join(NAMES_RECORDINGS, re.sub(r"[^A-Za-z0-9_-]", "_", word) + ".wav")


async def _names_calibrate(word: str, wav: bytes) -> dict:
    """A person said the word: what the mic caught comes back at once, and a job joins the queue to find the spelling
    whose takes in her voice sound closest to the recording (_names_stick_run, kind "voice"). The recording is kept
    (one per word) so they can hear it, and typed respellings are judged against its sounds."""
    audio_b64, rate = _uploaded_wav(wav)
    os.makedirs(NAMES_RECORDINGS, exist_ok=True)
    with open(_names_recording_path(word), "wb") as fh:
        fh.write(wav)
    async with _names_lock:
        heard = pron.fold_heard(await _names_hear(audio_b64, rate))
    if not heard:
        return {"status": "error", "detail": "Nothing was heard in the recording - try again, closer to the mic."}
    previous = pron.load_reviewed().get(word.lower(), {}).get("decision")
    pron.mark_reviewed(word, previous if previous in ("set", "fine") else "recorded", recorded=" ".join(heard))
    job = _names_enqueue(word, voice_wav=wav)
    job.update(you_said=pron.respell_ipa(heard), you_heard=" ".join(heard))
    _names_job_save(job)
    return {"status": "ok", "word": word, "heard": " ".join(heard), "sounds_like": pron.respell_ipa(heard),
            "queued": _names_position(job), "id": job["id"]}


async def _names_try(word: str, spelling: str) -> dict:
    """Her voice says a typed respelling, judged against the person's
    recording of the word if there is one, else the checker's target."""
    reviewed = pron.load_reviewed().get(word.lower(), {})
    if reviewed.get("recorded"):
        targets, against = [pron.target_from_recording(reviewed["recorded"].split())], "your recording"
    else:
        entry = pron.load_cache().get(word.lower())
        if not entry:
            return {"status": "error", "detail": "No target for this word yet - say it first."}
        targets, against = [pron.parse_ipa(entry["ipa"])], f"/{entry['ipa']}/"
    async with _names_lock:
        result = _names_result(spelling, await pron.rate_spelling(spelling, targets, _names_say_and_hear))
    return {"status": "ok", "takes": len(pron.TAKES), "against": against, **result}


@app.get("/admin/names", response_class=HTMLResponse)
async def names_page():
    with open(os.path.join(os.path.dirname(__file__), "names.html"), encoding="utf-8") as f:
        return f.read()


NAMES_MIN_MENTIONS = 3     # the list shows the words the KB uses at least this often, most used first; rarer ones
                           # aren't worth a person's review
_names_mentions = {"key": None, "counts": None}


def _kb_mentions(folder: str | None):
    """word_mentions of the KB's text; read again only when its files change."""
    if not folder:
        return None
    stamp, own = [], os.path.join(folder, PRONUNCIATIONS_FILE).lower()   # her voice's file isn't KB text
    for dirpath, _dirs, files in os.walk(folder):
        for f in files:
            path = os.path.join(dirpath, f)
            if f.lower().endswith(KB_EXTENSIONS) and path.lower() != own:
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                stamp.append((path, st.st_size, st.st_mtime_ns))
    key = (folder, tuple(sorted(stamp)))
    if _names_mentions["key"] != key:
        _names_mentions.update(key=key, counts=word_mentions(read_kb(folder)[0]))
    return _names_mentions["counts"]


@app.get("/admin/names/list")
async def names_list():
    current = pron.load_current_words()
    cache = {k: r for k, r in pron.load_cache().items() if current is None or k in current}
    settled = _names_settled()
    folder = _kb_folder()
    counts = {s: sum(1 for r in cache.values() if r.get("status") == s) for s in ("ok", "fixed", "review", "unresolved")}
    mentions = await asyncio.get_event_loop().run_in_executor(None, _kb_mentions, folder)
    items, rare = pron.by_mentions(pron.review_items(cache, set(settled), pron.load_reviewed()), mentions,
                                   NAMES_MIN_MENTIONS)
    return {
        "kb": os.path.basename(os.path.normpath(folder)) if folder else None,
        "writes_to": _names_edit_target(),
        "checker": {"enabled": pron.check_enabled(), "checked": len(cache), **counts},
        "listening_paused": pron.listening_paused(),
        "takes": len(pron.TAKES), "majority": pron.MAJORITY,
        "items": items, "rare": rare, "min_mentions": NAMES_MIN_MENTIONS,
        "settled": [{"word": n, "respelling": s, "file": f} for n, s, f in settled.values()],
        "confirmed": _names_confirmed(),
    }


class NameSpeak(BaseModel):
    text: str
    word: str | None = None      # the name being reviewed: its playback is kept for "She's fine"
    spelling: str | None = None  # what stood for the name in text (a typed respelling); default: the word


NAMES_RENDER_KEEP_S = 900       # a take you heard can be kept to for 15 minutes
_names_renders: dict = {}       # word (lowercase) -> {"pcm", "spelling", "before", "after", "at"}


def _names_keep_render(word: str, text: str, pcm: np.ndarray, spelling: str = None):
    """Keep the last take played for a name - as spelled or a typed respelling - with the sentence around it,
    so "She's fine" can keep her to that take; older ones are let go."""
    now = time.monotonic()
    for key in [k for k, r in _names_renders.items() if now - r["at"] > NAMES_RENDER_KEEP_S]:
        del _names_renders[key]
    spelling = spelling or word
    at = text.lower().find(spelling.lower())
    if at >= 0:
        _names_renders[word.lower()] = {"pcm": pcm, "spelling": text[at:at + len(spelling)], "before": text[:at],
                                        "after": text[at + len(spelling):], "at": now}


@app.post("/admin/names/speak")
async def names_speak(req: NameSpeak):
    """Her voice saying the text, as a WAV for the page to play. A name's
    playback is kept, so "She's fine" can keep her to that take."""
    text = req.text[:300]
    wav = await asyncio.get_event_loop().run_in_executor(None, lambda: _synthesize(text, settings.REFERENCE_VOICE))
    pcm = _pcm16(wav)
    if req.word:
        _names_keep_render(req.word, text, pcm, req.spelling)
    return Response(content=_wav_file_bytes(pcm), media_type="audio/wav")


# "She's fine" after a take you liked: keep her to THAT take. Her voice re-performs every take, so the page looks for
# a spelling she repeats it with: the one you played, then spellings built from the take's sounds, each tried in 8
# sentences and judged by how the name SOUNDS against your take (orchestrator/pron_match.py: sound, pacing,
# stress), not by a transcription.
STICK_SENTENCES = list(pron.TAKES) + ["Have you heard of {} before?", "{} is part of the company now.",
                                      "We visited {} last spring."]
STICK_SAVE = 6           # the best spelling is saved at once when this many of its 8 takes sound like yours: a
                         # take of the same pronunciation passes the strict match line only ~60-70% of the time, while a
                         # different one passes ~10% - 6 of 8 by chance is ~1 in 25,000 for a wrong spelling
STICK_EARLY_OUT = 4      # a spelling with no matching take in its first 4 is dropped
STICK_MAX = 16           # spellings tried per round of proposals
NAMES_QUEUE_DIR = os.path.join(os.path.dirname(__file__), "data", "names_queue")   # <id>.json + <id>.wav, outside git
NAMES_QUIET_S = 20       # her own conversation goes first: work waits until her voice and LLM have been quiet this long
NAMES_BUSY_POLL_S = 5    # how often a waiting job looks again
NAMES_KEEP_DONE_S = 7 * 24 * 3600
_names_jobs: dict = {}   # id -> job (progress and result, polled by the page); kept on disk, so a restart resumes
_names_queue_task = None


async def _names_features(pcm: np.ndarray, rate: int = 24000) -> dict:
    """The recognizer's sounds, frames and features for a take of hers (24 kHz) or a person's recording."""
    global _names_worker, _names_worker_last
    if _names_worker is None:
        _names_worker = pron.PhonemeWorker(
            device="cuda:1", log_path=os.path.join(os.path.dirname(__file__), "logs", "pronunciation_worker_admin.log"))
        asyncio.create_task(_names_worker_reaper())
    _names_worker_last = time.monotonic()
    return await _names_worker.hear(base64.b64encode(pcm.tobytes()).decode(), rate, features=True)


def _names_target(word: str) -> list:
    entry = pron.load_cache().get(word.lower()) or {}
    return pron.parse_ipa(entry["ipa"]) if entry.get("ipa") else []


async def _names_take(spelling: str, k: int, target: list):
    """Her voice says the spelling in sentence k; the name's sounds (with a margin), or None if not found."""
    before, after = STICK_SENTENCES[k % len(STICK_SENTENCES)].split("{}")
    text = before + spelling + after
    wav = await asyncio.get_event_loop().run_in_executor(None, lambda: _synthesize(text, settings.REFERENCE_VOICE))
    heard = await _names_features(_pcm16(wav))
    b, a = pron.sentence_sounds(before), pron.sentence_sounds(after)
    frames = pron_match.name_frames(heard, b, target, a) if b is not None and a is not None else None
    return pron_match.segment(heard, frames, pron_match.MARGIN) if frames else None


async def _names_try_spelling(spelling: str, approved: dict, target: list, job: dict, match, best=None) -> dict:
    """8 takes of the spelling, each judged against the approved take. match: the line for "sounds the same" (None:
    a voice job before calibration - then a spelling clearly farther than the best so far is dropped early)."""
    dists = []
    for k in range(len(STICK_SENTENCES)):
        job["take"] = k + 1
        seg = await _names_take(spelling, k, target)
        dists.append(pron_match.distance(approved, seg, open_ends=True)["total"] if seg else None)
        if k + 1 == STICK_EARLY_OUT:
            found = [d for d in dists if d is not None]
            if match is not None and not any(d <= match for d in found):
                break
            if match is None and best is not None and (not found or statistics.median(found) > best + VOICE_EARLY_OUT):
                break
    found = [d for d in dists if d is not None]
    return {"spelling": spelling, "right": sum(d <= match for d in found) if match is not None else None,
            "takes": len(dists), "distance": round(statistics.median(found), 3) if found else None,
            "dists": [round(d, 3) if d is not None else None for d in dists]}


def _names_rank(results: list) -> list:
    if any(r["right"] is None for r in results):
        return sorted(results, key=lambda r: (r["distance"] if r["distance"] is not None else 9.0, -r["takes"]))
    return sorted(results, key=lambda r: (-r["right"] / max(1, r["takes"]), -r["right"],
                                          r["distance"] if r["distance"] is not None else 9.0))


VOICE_EARLY_OUT = 0.08   # a voice job drops a spelling whose first 4 takes are this much farther than the best so far
VOICE_PICKS = "voice_picks.jsonl"


def _names_voice_match():
    """The "sounds like your voice" line ([names] voice_match), set once ~10 picks calibrate it; None until then."""
    cfg = configparser.ConfigParser()
    cfg.read(KIOSK_CONFIG, encoding="utf-8")
    try:
        return cfg.getfloat("names", "voice_match")
    except (configparser.Error, ValueError):
        return None


async def _names_stick_run(job: dict, render: dict):
    """The job behind "She's fine" (see above); progress and result go into job."""
    word, voice = job["word"], job.get("kind") == "voice"
    try:
        heard = await _names_features(render["pcm"], render.get("rate", 24000))
        target = _names_target(word)
        if voice:           # a person said the name alone: all of it is the name
            folded = pron_match.fold_spans(heard["spans"])
            frames = (folded[0][1], folded[-1][2]) if folded and folded[-1][2] - folded[0][1] >= 3 else None
            window = [s for s, _, _ in folded]
        else:
            before, after = pron.sentence_sounds(render["before"]), pron.sentence_sounds(render["after"])
            frames = pron_match.name_frames(heard, before, target, after) if before is not None and after is not None else None
            window = pron.name_in_sentence(before, target, after, [s for s, _, _ in pron_match.fold_spans(heard["spans"])]) \
                if frames else []
        if not frames:
            job.update(status="error", detail="Nothing was heard in your recording - say it again, closer to the mic."
                       if voice else "Couldn't find the name in that take - play it again, then She's fine.")
            return
        approved = pron_match.segment(heard, frames)
        heard_as, approved_text = pron.respell_ipa(window), render["spelling"]
        if not target:      # no expected pronunciation on file: find the name in later takes by the reference's sounds
            target = pron.target_from_recording(window)
        match = _names_voice_match() if voice else pron_match.MATCH
        job.update(heard_as=heard_as, approved=approved_text, stage="testing")
        tested, queue = [], [approved_text]

        def done(sp):
            return sp.lower() in {t["spelling"].lower() for t in tested}

        while True:
            if not queue:
                fresh = []
                if not job.get("own_sounds"):
                    job["own_sounds"] = True                  # once: spellings built from the take's own sounds
                    fresh = list(pron.respellings_for(window))
                queue = [sp for sp in dict.fromkeys(fresh) if not done(sp)][:STICK_MAX]
                if not queue:
                    break
                job["stage"] = "testing"
            sp = queue.pop(0)
            job.update(current=sp, tested=len(tested), queued=len(queue))
            best_d = min((t["distance"] for t in tested if t["distance"] is not None), default=None)
            tested.append(await _names_try_spelling(sp, approved, target, job, match, best_d))
            if match is not None and _names_rank(tested)[0]["right"] >= STICK_SAVE:
                break
        ranked = _names_rank(tested)
        best, n = ranked[0], len(STICK_SENTENCES)
        out = {"status": "done", "results": ranked[:3], "all": ranked, "tested": len(tested), "takes": n,
               "heard_as": heard_as, "approved": approved_text, "current": None}
        if match is None:
            out.update(outcome="pick",
                       detail=f"Closest to how you said it (~{heard_as}), best first - play them, then Use it. Your "
                              f"picks teach it when to save by itself (after about 10).")
        elif best["right"] >= STICK_SAVE:
            if best["spelling"].lower() == word.lower():
                pron.mark_reviewed(word, "fine", heard=" ".join(window))
                out.update(outcome="fine",
                           detail=f"Kept: as spelled she says it the way you heard {best['right']} of {n} times.")
            else:
                path = _names_edit_target()
                pron.set_pronunciation(path, word, best["spelling"])
                pron.mark_reviewed(word, "set", respelling=best["spelling"], heard=" ".join(window))
                out.update(outcome="set", respelling=best["spelling"], file=path,
                           detail=f"Kept: {word} = {best['spelling']} - she says it the way you heard "
                                  f"{best['right']} of {n} times.")
        else:
            out.update(outcome="varies",
                       detail=f"Her voice doesn't repeat {'your voice' if voice else 'the take you heard'} reliably: the "
                              f"closest spelling, {best['spelling']}, sounded like it {best['right']} of {best['takes']} "
                              f"times. Nothing saved - try one below, or {'say it again' if voice else 'play it again and She' + chr(39) + 's fine on a take you like'}.")
        job.update(out)
    except Exception as e:
        job.update(status="error", detail=repr(e))


def _names_job_path(job_id: str, ext: str) -> str:
    return os.path.join(NAMES_QUEUE_DIR, f"{job_id}.{ext}")


def _names_job_save(job: dict):
    os.makedirs(NAMES_QUEUE_DIR, exist_ok=True)
    tmp = _names_job_path(job["id"], "json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(job, fh, ensure_ascii=False)
    os.replace(tmp, _names_job_path(job["id"], "json"))


def _names_job_delete(job_id: str):
    _names_jobs.pop(job_id, None)
    for ext in ("json", "wav"):
        try:
            os.remove(_names_job_path(job_id, ext))
        except OSError:
            pass


def _names_queue_load():
    """Jobs from disk (after a restart): unfinished ones go back in line, old finished ones are let go."""
    if not os.path.isdir(NAMES_QUEUE_DIR):
        return
    for name in sorted(os.listdir(NAMES_QUEUE_DIR)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(NAMES_QUEUE_DIR, name), encoding="utf-8") as fh:
                job = json.load(fh)
        except (OSError, ValueError):
            continue
        if job.get("status") in ("done", "error") and time.time() - job.get("finished_at", 0) > NAMES_KEEP_DONE_S:
            _names_job_delete(job["id"])
            continue
        if job.get("status") == "running":
            job["status"] = "queued"
        _names_jobs.setdefault(job["id"], job)


def _names_waiting() -> list:
    return sorted((j for j in _names_jobs.values() if j["status"] == "queued"), key=lambda j: j["queued_at"])


def _names_enqueue(word: str, render: dict = None, voice_wav: bytes = None) -> dict:
    """She's fine: the take you heard is saved with its sentence and joins the line; Say it: the person's recording
    does (kind "voice"). A newer take or recording of a name still waiting replaces the older one."""
    for j in _names_waiting():
        if j["word"].lower() == word.lower():
            j["status"] = "replaced"              # the worker may be holding it while she talks
            _names_job_delete(j["id"])
    job_id = (time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}-"
              + re.sub(r"[^a-z0-9]+", "-", word.lower()).strip("-")[:30])
    while job_id in _names_jobs:
        job_id += "x"
    os.makedirs(NAMES_QUEUE_DIR, exist_ok=True)
    with open(_names_job_path(job_id, "wav"), "wb") as fh:
        fh.write(voice_wav if voice_wav is not None else _wav_file_bytes(render["pcm"]))
    if voice_wav is not None:
        job = {"id": job_id, "word": word, "kind": "voice", "spelling": word, "before": "", "after": "",
               "status": "queued", "queued_at": time.time()}
    else:
        job = {"id": job_id, "word": word, "kind": "take", "spelling": render["spelling"], "before": render["before"],
               "after": render["after"], "status": "queued", "queued_at": time.time()}
    _names_jobs[job_id] = job
    _names_job_save(job)
    _names_queue_start()
    return job


def _her_busy() -> bool:
    """She is talking with someone (her LLM or voice used in the last NAMES_QUIET_S seconds)."""
    return _llm_gateway is not None and _llm_gateway.seconds_since_activity() < NAMES_QUIET_S


def _names_queue_start():
    global _names_queue_task
    if _names_queue_task is None or _names_queue_task.done():
        _names_queue_task = asyncio.create_task(_names_queue_worker())


async def _names_queue_worker():
    """One job at a time, oldest first, each waiting while she is in a conversation."""
    while True:
        waiting = _names_waiting()
        if not waiting:
            return
        job = waiting[0]
        while _her_busy():
            job["stage"] = "waiting: she's talking with someone"
            await asyncio.sleep(NAMES_BUSY_POLL_S)
            if job["status"] != "queued":          # cancelled or replaced meanwhile
                break
        if job["status"] != "queued":
            continue
        job.update(status="running", stage="listening to your take", started_at=time.time())
        try:
            with open(_names_job_path(job["id"], "wav"), "rb") as fh:
                pcm_b64, rate = _uploaded_wav(fh.read())
            await _names_stick_run(job, {"pcm": np.frombuffer(base64.b64decode(pcm_b64), dtype=np.int16), "rate": rate,
                                         "spelling": job["spelling"], "before": job["before"], "after": job["after"]})
        except Exception as e:
            job.update(status="error", detail=repr(e))
        if job["status"] == "running":
            job["status"] = "error"
        job["finished_at"] = time.time()
        _names_job_save(job)


def _names_latest(word: str):
    mine = [j for j in _names_jobs.values() if j["word"].lower() == word.lower()]
    return max(mine, key=lambda j: j["queued_at"]) if mine else None


def _names_position(job: dict):
    waiting = _names_waiting()
    return next((k + 1 for k, j in enumerate(waiting) if j["id"] == job["id"]), None)


def _names_confirmed() -> list:
    """Names confirmed on this page (a respelling set, or marked fine), in
    the order they were confirmed, each with how her voice says it now -
    the live pronunciation files applied, exactly as in conversation."""
    folder = _kb_folder()
    live = Pronunciations.load(pron.AUTO_PRONUNCIATIONS, KIOSK_PRONUNCIATIONS,
                               *([os.path.join(folder, PRONUNCIATIONS_FILE)] if folder else []))
    done = sorted((v for v in pron.load_reviewed().values() if v.get("decision") in ("set", "fine")),
                  key=lambda v: v.get("when", ""))
    return [{"word": v["word"], "decision": v["decision"], "spoken": live.apply(v["word"])} for v in done]


@app.get("/admin/names/confirmed.wav")
async def names_confirmed_wav(limit: int = 100):
    """Her voice saying each confirmed name once, in a row, with a short
    pause between - to hear the whole set the way she'll say it."""
    gap = np.zeros(int(0.5 * 24000), dtype=np.int16)
    pieces = []
    for item in _names_confirmed()[:limit]:
        wav = await asyncio.get_event_loop().run_in_executor(
            None, lambda s=item["spoken"]: _synthesize(f"{s}.", settings.REFERENCE_VOICE))
        pieces += [_pcm16(wav), gap]
    pcm = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.int16)
    return Response(content=_wav_file_bytes(pcm), media_type="audio/wav")


@app.post("/admin/names/say_confirmed")
async def names_say_confirmed():
    """Have her say each confirmed name once, in order, at the kiosk: the
    orchestrator speaks them through her real voice path, applying the
    respellings it has loaded - proof they were picked up."""
    words = [c["word"] + "." for c in _names_confirmed()]
    if words:
        pron.request_say(words)
    return {"status": "ok", "queued": len(words)}


@app.get("/admin/names/recording/{word}")
async def names_recording(word: str):
    """The person's last recording of a word, to hear what the mic caught."""
    path = _names_recording_path(word)
    if not os.path.exists(path):
        return Response(status_code=404)
    with open(path, "rb") as fh:
        return Response(content=fh.read(), media_type="audio/wav")


@app.post("/admin/names/calibrate")
async def names_calibrate(word: str = Form(...), audio: UploadFile = File(...)):
    try:
        return await _names_calibrate(word, await audio.read())
    except Exception as e:
        return {"status": "error", "detail": repr(e)}


class NameTry(BaseModel):
    word: str
    spelling: str


@app.post("/admin/names/try")
async def names_try(req: NameTry):
    try:
        return await _names_try(req.word, req.spelling)
    except Exception as e:
        return {"status": "error", "detail": repr(e)}


class NameAccept(BaseModel):
    word: str
    respelling: str


@app.post("/admin/names/accept")
async def names_accept(req: NameAccept):
    """Write 'word = respelling' into the pronunciations file (the drive's if
    possible); the orchestrator applies it within seconds."""
    path = _names_edit_target()
    pron.set_pronunciation(path, req.word, req.respelling)
    pron.mark_reviewed(req.word, "set", respelling=req.respelling)
    job = _names_latest(req.word)
    if job and job.get("kind") == "voice" and job.get("status") == "done":
        os.makedirs(NAMES_QUEUE_DIR, exist_ok=True)
        with open(os.path.join(NAMES_QUEUE_DIR, VOICE_PICKS), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": time.time(), "word": req.word, "job": job["id"], "chosen": req.respelling,
                                 "tested": job.get("all", [])}, ensure_ascii=False) + "\n")
    return {"status": "ok", "file": path}


class NameWord(BaseModel):
    word: str
    peg: bool = False    # pin her to the version of the word the person just played


@app.post("/admin/names/fine")
async def names_fine(req: NameWord):
    """She says it right - take it off the list. With peg: keep her to the take the person just played (as
    spelled or a typed respelling) - starts the job above; the page polls GET /admin/names/fine/<word>."""
    if req.peg:
        render = _names_renders.get(req.word.lower())
        if not render or time.monotonic() - render["at"] > NAMES_RENDER_KEEP_S:
            return {"status": "error", "detail": "The take you played isn't kept any more (over 15 minutes, or "
                                                 "the server restarted) - play it again, then She's fine."}
        job = _names_enqueue(req.word, render)
        return {"status": "queued", "id": job["id"], "position": _names_position(job)}
    pron.mark_reviewed(req.word, "fine")
    return {"status": "ok", "outcome": "fine", "detail": "Marked fine."}


@app.get("/admin/names/fine/{word}")
async def names_fine_progress(word: str):
    job = _names_latest(word)
    return {**job, "position": _names_position(job)} if job else {"status": "none"}


@app.get("/admin/names/queue")
async def names_queue():
    """Every job still waiting, working, or finished in the last week - for the page's queue panel."""
    if not _names_jobs:
        _names_queue_load()
    if _names_waiting():
        _names_queue_start()
    jobs = sorted(_names_jobs.values(), key=lambda j: j["queued_at"])
    return {"jobs": [{**j, "position": _names_position(j)} for j in jobs],
            "her_busy": _her_busy() and any(j["status"] == "queued" for j in jobs)}


@app.delete("/admin/names/queue/{job_id}")
async def names_queue_cancel(job_id: str):
    job = _names_jobs.get(job_id)
    if not job or job["status"] != "queued":
        return JSONResponse({"error": "only a job still waiting can be taken out of the line"}, status_code=400)
    job["status"] = "cancelled"
    _names_job_delete(job_id)
    return {"ok": True}


class NamesListening(BaseModel):
    paused: bool
    minutes: float = 10.0


@app.post("/admin/names/listening")
async def names_listening(req: NamesListening):
    """Pause her listening while a person reads names at the kiosk (so she
    doesn't answer each one); it resumes on its own after `minutes`."""
    pron.pause_listening(req.minutes * 60 if req.paused else 0)
    return {"paused": pron.listening_paused()}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
    )
