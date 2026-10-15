"""
LLM Gateway — session-aware proxy to llama-server with single-session enforcement.

Manages one user session at a time. KV cache stays warm across all HTTP requests
within a session. Cache is erased on session end (idle timeout or explicit signal).

No disk persistence of KV cache — ever. Cross-session knowledge belongs in the
hierarchical memory system (memory_store.py), not raw attention state.

Session lifecycle:
  new user → create session → turns reuse warm KV cache → 5min idle → erase → ready
"""

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx
from starlette.requests import Request

# Single-user mode's one live conversation (orchestrator/single_user.py
# SESSION_PREFIX) never expires: there is no next user to protect, and its
# KV cache is the conversation she is still in.
PERSISTENT_SESSION_PREFIX = "single-user-"


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    turn_count: int = 0


class LLMGateway:
    """Single-session LLM proxy with request serialization and idle cache management."""

    def __init__(
        self,
        llama_url: str = "http://localhost:8080",
        idle_timeout_s: float = 300.0,  # 5 minutes
    ):
        self._llama_url = llama_url
        self._idle_timeout_s = idle_timeout_s

        # Request serialization — one LLM call at a time
        self._lock = asyncio.Lock()

        # Active session tracking
        self._session: SessionState | None = None

        # In-flight request cancellation for barge-in
        self._cancel_event = asyncio.Event()

        # Reference to active llama-server stream — for explicit cancel
        self._active_stream: httpx.Response | None = None

        # Background idle monitor
        self._idle_task: asyncio.Task | None = None

        # HTTP client (reused across requests for connection pooling)
        self._client: httpx.AsyncClient | None = None

        # Kiosk-wide activity timestamp — bumped by every LLM call AND by
        # external paths that count as "kiosk hot" (TTS during identification
        # flow). Used by deferred memory compression to know when slot 0 is
        # safe to use without contending with a live conversation. Persists
        # across session changes; 0.0 means "no activity since process start".
        self._last_kiosk_activity: float = 0.0

    async def start(self):
        """Call during app lifespan startup."""
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
        self._idle_task = asyncio.create_task(self._idle_monitor())
        print(f"[LLM Gateway] Started - llama-server at {self._llama_url}, "
              f"idle timeout {self._idle_timeout_s}s")

    async def stop(self):
        """Call during app lifespan shutdown."""
        if self._idle_task:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
        print("[LLM Gateway] Stopped")

    # ─── Public API ───────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        session_id: str,
        request: Request,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = True,
    ):
        """
        Proxy an LLM request to llama-server with session management.

        The Request object is used to detect client disconnect — when the
        client drops the connection (barge-in), the generator exits cleanly
        and releases the lock.
        """
        # Defense-in-depth: catch non-alternating history before llama-server
        validation_error = self._validate_messages(messages)
        if validation_error:
            print(f"[LLM Gateway] History validation failed: {validation_error['detail']}")
            if stream:
                return self._error_stream(validation_error)
            else:
                return validation_error

        await self._ensure_session(session_id)

        if stream:
            return self._stream_generate(messages, max_tokens, temperature, request)
        else:
            return await self._blocking_generate(messages, max_tokens, temperature)

    async def end_session(self, session_id: str | None = None) -> dict:
        """
        Explicitly end a session and erase KV cache.
        If session_id is provided, only ends that specific session.
        """
        if self._session is None:
            return {"status": "no_active_session"}

        if session_id and self._session.session_id != session_id:
            return {"status": "session_mismatch",
                    "active": self._session.session_id,
                    "requested": session_id}

        sid = self._session.session_id
        turns = self._session.turn_count
        await self._erase_cache()
        self._session = None
        print(f"[LLM Gateway] Session '{sid}' ended explicitly ({turns} turns)")
        return {"status": "ended", "session_id": sid, "turns": turns}

    async def cancel(self) -> dict:
        """
        Cancel any in-flight LLM generation. KV cache preserved.
        Safety net — primary barge-in mechanism is client disconnect.
        """
        if self._session is None:
            return {"status": "no_active_session", "cancelled": False}

        was_generating = self._lock.locked()
        self._cancel_event.set()

        if self._active_stream is not None:
            try:
                await self._active_stream.aclose()
            except Exception:
                pass
            self._active_stream = None

        return {
            "status": "cancelled" if was_generating else "idle",
            "cancelled": was_generating,
            "session_id": self._session.session_id,
            "turn_count": self._session.turn_count,
        }

    def session_info(self) -> dict:
        """Return current session state for health/debug endpoints."""
        if self._session is None:
            return {"active": False}
        idle_s = time.monotonic() - self._session.last_activity
        return {
            "active": True,
            "session_id": self._session.session_id,
            "turn_count": self._session.turn_count,
            "idle_seconds": round(idle_s, 1),
            "timeout_in_seconds": round(max(0, self._idle_timeout_s - idle_s), 1),
            "lock_held": self._lock.locked(),
        }

    # ─── Validation ─────────────────────────────────────────────────────────

    def _validate_messages(self, messages: list[dict]) -> dict | None:
        """Validate message history before sending to llama-server.

        Returns error dict if invalid, None if OK. Catches the common
        barge-in bug: consecutive user messages after a dropped assistant turn.
        """
        prev_role = None
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            if role == "system":
                prev_role = role
                continue
            if role == prev_role and role in ("user", "assistant"):
                return {
                    "error": "invalid_history",
                    "detail": (f"Consecutive {role} messages at index {i-1} and {i}. "
                               f"Chat template requires strict user/assistant alternation."),
                    "suggestion": "Add an assistant message between consecutive user messages.",
                }
            prev_role = role
        return None

    async def _error_stream(self, error: dict):
        """Yield a single SSE error event. Used when validation fails in streaming mode."""
        yield f"data: {json.dumps(error)}\n\n"

    # ─── Session management ───────────────────────────────────────────────────

    async def _ensure_session(self, session_id: str):
        """Create or validate session. Erase cache on session change."""
        if self._session is None:
            self._session = SessionState(session_id=session_id)
            print(f"[LLM Gateway] New session '{session_id}'")
        elif self._session.session_id != session_id:
            old_id = self._session.session_id
            await self._erase_cache()
            self._session = SessionState(session_id=session_id)
            print(f"[LLM Gateway] Session switch: '{old_id}' -> '{session_id}' (cache erased)")

    async def _wait_slot_idle(self, timeout_s: float = 2.0) -> bool:
        """Poll llama-server until slot 0 is idle."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                resp = await self._client.get(f"{self._llama_url}/slots")
                if resp.status_code == 200:
                    slots = resp.json()
                    if slots and not slots[0].get("is_processing", True):
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.05)
        return False

    async def _erase_cache(self):
        """Erase llama-server slot 0 KV cache. No disk write."""
        try:
            resp = await self._client.post(
                f"{self._llama_url}/slots/0?action=erase"
            )
            if resp.status_code == 200:
                print("[LLM Gateway] Slot 0 KV cache erased")
            else:
                print(f"[LLM Gateway] Cache erase returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[LLM Gateway] Cache erase failed: {e}")

    def bump_activity(self):
        """Mark the gateway as active. Updates the active session's
        last_activity (if any) and the kiosk-wide activity timestamp.

        Callers: every LLM generate path bumps via this. External callers
        (e.g., TTS during identification flow) bump kiosk activity to keep
        deferred memory compression from firing while the kiosk is in use.
        """
        now = time.monotonic()
        if self._session is not None:
            self._session.last_activity = now
        self._last_kiosk_activity = now

    def seconds_since_activity(self) -> float:
        """Seconds since any kiosk activity (LLM or TTS).

        Used by deferred memory compression: a value >= the idle threshold
        means slot 0 is safe to use without contending with a live
        conversation. Starts at process_uptime_seconds (large) until the
        first bump_activity() call.
        """
        return time.monotonic() - self._last_kiosk_activity

    async def _idle_monitor(self):
        """Background task: erase cache and clear session after idle timeout."""
        while True:
            await asyncio.sleep(30)
            if self._session is None:
                continue
            if self._session.session_id.startswith(PERSISTENT_SESSION_PREFIX):
                continue
            idle_s = time.monotonic() - self._session.last_activity
            if idle_s >= self._idle_timeout_s:
                sid = self._session.session_id
                turns = self._session.turn_count
                await self._erase_cache()
                self._session = None
                print(f"[LLM Gateway] Session '{sid}' expired after {idle_s:.0f}s idle "
                      f"({turns} turns)")

    # ─── LLM proxy (streaming) ────────────────────────────────────────────────

    async def _stream_generate(self, messages, max_tokens, temperature, request: Request):
        """Stream tokens from llama-server. Exits cleanly on client disconnect."""
        # Signal any in-flight request to abort
        self._cancel_event.set()

        # Force-close any active stream (for explicit cancel path)
        if self._active_stream is not None:
            try:
                await self._active_stream.aclose()
            except Exception:
                pass
            self._active_stream = None

        async with self._lock:
            self._cancel_event.clear()

            # Verify slot is idle before generating
            if not await self._wait_slot_idle(timeout_s=2.0):
                print("[LLM Gateway] Slot busy — erasing cache as fallback")
                await self._erase_cache()
                await asyncio.sleep(0.1)

            self._session.turn_count += 1
            self.bump_activity()
            t0 = time.monotonic()
            ttft = None
            tokens_generated = 0

            payload = {
                "model": "gemma",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "cache_prompt": True,
                "id_slot": 0,
            }

            try:
                async with self._client.stream(
                    "POST",
                    f"{self._llama_url}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    self._active_stream = resp
                    if resp.status_code != 200:
                        body = await resp.aread()
                        error_text = body.decode(errors="replace")
                        print(f"[LLM Gateway] llama-server returned {resp.status_code}: {error_text[:200]}")
                        yield f"data: {json.dumps({'error': 'llm_error', 'status': resp.status_code, 'detail': error_text, 'session_id': self._session.session_id, 'turn_count': self._session.turn_count})}\n\n"
                        return

                    line_iter = resp.aiter_lines()
                    # ONE pending __anext__ task, reused across heartbeat
                    # timeouts. Creating a fresh task per iteration double-
                    # enters the async generator ("anext(): asynchronous
                    # generator is already running") whenever a line takes
                    # >200ms, as on every prefill (llama-server can send
                    # nothing until prefill completes).
                    line_task = None
                    while True:
                        # Race: next token vs client disconnect, with heartbeat timeout.
                        # The 200ms timeout yields an SSE comment to trigger uvicorn's
                        # disconnect detection — uvicorn only notices the client is gone
                        # when it tries to write to the socket.
                        if line_task is None:
                            line_task = asyncio.create_task(
                                line_iter.__anext__()
                            )

                        done, pending = await asyncio.wait(
                            {line_task},
                            timeout=0.2,
                        )

                        if not done:
                            # No token in 200ms — yield heartbeat to probe connection
                            yield ": heartbeat\n\n"
                            # Check if uvicorn detected disconnect during the yield
                            if await request.is_disconnected():
                                line_task.cancel()
                                try:
                                    await line_task
                                except (asyncio.CancelledError, StopAsyncIteration):
                                    pass
                                print(f"[LLM Gateway] Client disconnected after "
                                      f"{tokens_generated} tokens — clean exit")
                                return
                            # Also check cancel event
                            if self._cancel_event.is_set():
                                line_task.cancel()
                                try:
                                    await line_task
                                except (asyncio.CancelledError, StopAsyncIteration):
                                    pass
                                print(f"[LLM Gateway] Request cancelled after "
                                      f"{tokens_generated} tokens")
                                yield f"data: {json.dumps({'token': '', 'done': True, 'cancelled': True, 'tokens_generated': tokens_generated})}\n\n"
                                return
                            continue

                        # Cancel event check
                        if self._cancel_event.is_set():
                            print(f"[LLM Gateway] Request cancelled after "
                                  f"{tokens_generated} tokens")
                            yield f"data: {json.dumps({'token': '', 'done': True, 'cancelled': True, 'tokens_generated': tokens_generated})}\n\n"
                            return

                        # Process the line (task consumed — next iteration
                        # starts a fresh __anext__)
                        try:
                            line = line_task.result()
                        except StopAsyncIteration:
                            break
                        finally:
                            line_task = None

                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        finish = choices[0].get("finish_reason")

                        if content:
                            if ttft is None:
                                ttft = (time.monotonic() - t0) * 1000
                            tokens_generated += 1
                            yield f"data: {json.dumps({'token': content, 'done': False})}\n\n"

                        if finish:
                            break

            except (httpx.ReadError, httpx.StreamClosed, httpx.RemoteProtocolError):
                print(f"[LLM Gateway] Stream interrupted after {tokens_generated} tokens")
            except Exception as e:
                print(f"[LLM Gateway] Stream error: {type(e).__name__}: {e}")
            finally:
                self._active_stream = None

            elapsed = time.monotonic() - t0
            tps = tokens_generated / max(elapsed, 0.001)
            self.bump_activity()

            print(f"[LLM Gateway] Turn {self._session.turn_count}: "
                  f"{tokens_generated} tokens, TTFT={ttft or 0:.0f}ms, "
                  f"TPS={tps:.1f}")

            yield f"data: {json.dumps({'token': '', 'done': True, 'tokens_generated': tokens_generated, 'ttft_ms': round(ttft or 0, 1), 'tps': round(tps, 1)})}\n\n"

    async def _blocking_generate(self, messages, max_tokens, temperature):
        """Non-streaming LLM call. Returns complete response dict."""
        async with self._lock:
            self._session.turn_count += 1
            self.bump_activity()
            t0 = time.monotonic()

            payload = {
                "model": "gemma",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "cache_prompt": True,
                "id_slot": 0,
            }

            resp = await self._client.post(
                f"{self._llama_url}/v1/chat/completions",
                json=payload,
            )

            elapsed = time.monotonic() - t0
            self.bump_activity()

            if resp.status_code != 200:
                print(f"[LLM Gateway] llama-server returned {resp.status_code}: {resp.text[:200]}")
                return {"error": "llm_error", "status": resp.status_code, "detail": resp.text,
                        "session_id": self._session.session_id, "turn_count": self._session.turn_count}

            result = resp.json()
            text = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("completion_tokens", len(text.split()))

            return {
                "text": text,
                "tokens_generated": tokens,
                "ttft_ms": round(elapsed * 1000, 1),
                "tps": round(tokens / max(elapsed, 0.001), 1),
            }
