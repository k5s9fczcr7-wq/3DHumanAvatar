"""LLM-forced classification of spoken yes/no confirmations.

The identification flow asks yes/no questions ("Are you Sam?", "Your PIN is
1234 — right?"). Visitors answer in infinitely many ways ("You can be assured
that I am", "you bet", "not me, I'm his brother"), and ASR adds transcription
noise on top. Pattern lists cannot cover that space, so the LLM classifies —
its output is grammar-constrained to exactly one of yes|no|unclear, making a
malformed answer impossible. temperature=0 for determinism.

Latency: one short completion on the warm llama-server (~100-300 ms). All
confirmation moments happen BEFORE the conversation starts, so the prompt-cache
eviction this causes is free in practice (the first chat turn pays a cold
prefill anyway).

Resilience: if llama-server is unreachable or times out, returns "unclear" so
the calling flow reprompts gracefully instead of breaking the id state machine.
"""
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger("confirm_intent")

LLM_URL = "http://127.0.0.1:8080"

# GBNF grammar: the model can emit ONLY one of these three literals.
_GRAMMAR = 'root ::= "yes" | "no" | "unclear"'

_SYSTEM = (
    "You are a strict classifier. A kiosk avatar asked a visitor a yes/no "
    "question. Classify the visitor's spoken reply (the speech-to-text "
    "transcript may contain recognition errors) as exactly one of: yes, no, "
    "unclear.\n"
    "- yes: any affirmation, agreement, or confirmation, however phrased.\n"
    "- no: any denial, refusal, or correction, however phrased.\n"
    "- unclear: unrelated, garbled, evasive, or genuinely ambiguous.\n"
    "Answer with one word only."
)


_NAME_SYSTEM = (
    "You extract a person's name from a spoken reply to the question "
    "\"What's your name?\". The transcript may contain speech-recognition "
    "errors.\n"
    "- If the reply gives a name, output ONLY the name with normal "
    "capitalization (e.g. Dave, Mary Ann).\n"
    "- Greetings (hi, hello, hey), acknowledgements, filler, questions, "
    "refusals, and small talk are NOT names: output exactly none.\n"
    "- BUT a standalone word given as the answer IS the name even when it is "
    "also a common word or nickname (Sunny, Red, Summer, Angel) — people "
    "answer this question with unusual names. Only greetings, yes/no words, "
    "and obvious non-answers are none.\n"
    "- Never output more than three words and never add punctuation."
)

_NAME_WORD_RE = None  # built lazily to avoid import cost


def _valid_name(text: str) -> str:
    """Post-validate an LLM-extracted name. Returns cleaned name or ''."""
    import re
    cleaned = text.strip().strip(".,!?\"'").strip()
    if not cleaned or cleaned.lower() == "none":
        return ""
    words = cleaned.split()
    if len(words) > 3:
        return ""
    reject = {"none", "hi", "hello", "hey", "skip", "yes", "no", "yeah", "okay", "ok", "unknown", "name"}
    for w in words:
        if w.lower() in reject:
            return ""
        if not re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ'\-]{1,24}", w):
            return ""
    return " ".join(w[:1].upper() + w[1:] for w in words)


async def extract_name(
    reply: str,
    client: Optional[httpx.AsyncClient] = None,
    url: str = LLM_URL,
    timeout: float = 3.0,
) -> dict:
    """Extract a person's name from a spoken reply, or '' when none given.

    Returns {"name": str|None, "latency_ms": float, "source": "llm"|"fallback"}.
    name == ""  -> LLM confidently says the reply contains no name (re-ask).
    name is None -> LLM unreachable (caller falls back to regex extraction).
    """
    body = {
        "model": "local",
        "messages": [
            {"role": "system", "content": _NAME_SYSTEM},
            {"role": "user", "content": f'Reply: "{reply}"\nName:'},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "cache_prompt": True,
    }
    t0 = time.perf_counter()
    own_client = client is None
    try:
        if own_client:
            client = httpx.AsyncClient(timeout=timeout)
        r = await client.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        text = (r.json()["choices"][0]["message"]["content"] or "").strip()
        latency_ms = (time.perf_counter() - t0) * 1000
        return {"name": _valid_name(text), "latency_ms": latency_ms, "source": "llm"}
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.warning(f"[CONFIRM] LLM name-extract failed ({e!r}) -> regex fallback")
        return {"name": None, "latency_ms": latency_ms, "source": "fallback"}
    finally:
        if own_client and client is not None:
            await client.aclose()


async def classify_confirmation(
    question: str,
    reply: str,
    client: Optional[httpx.AsyncClient] = None,
    url: str = LLM_URL,
    timeout: float = 3.0,
) -> dict:
    """Classify a spoken reply to a yes/no question.

    Returns {"intent": "yes"|"no"|"unclear", "latency_ms": float,
    "source": "llm"|"fallback"}. Never raises: LLM failure -> "unclear"
    with source="fallback" (caller reprompts).
    """
    body = {
        "model": "local",
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": f'Question: "{question}"\nReply: "{reply}"\nClassification:',
            },
        ],
        "max_tokens": 4,
        "temperature": 0.0,
        "grammar": _GRAMMAR,
        "cache_prompt": True,
    }
    t0 = time.perf_counter()
    own_client = client is None
    try:
        if own_client:
            client = httpx.AsyncClient(timeout=timeout)
        r = await client.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        text = (r.json()["choices"][0]["message"]["content"] or "").strip().lower()
        latency_ms = (time.perf_counter() - t0) * 1000
        intent = text if text in ("yes", "no", "unclear") else "unclear"
        return {"intent": intent, "latency_ms": latency_ms, "source": "llm"}
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.warning(f"[CONFIRM] LLM classify failed ({e!r}) -> unclear")
        return {"intent": "unclear", "latency_ms": latency_ms, "source": "fallback"}
    finally:
        if own_client and client is not None:
            await client.aclose()
