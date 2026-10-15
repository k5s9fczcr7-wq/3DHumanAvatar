"""
Memory Compressor — LLM-based summarization for hierarchical memory tiers.

Uses the LLM (via the LLM Gateway) to compress conversation transcripts
into progressively shorter summaries. Runs at session end or on-demand.

Compression pipeline:
  Raw transcript → Verbatim (last 5 turns, lossless)
                 → Light (filler removed, ~300 tokens)
                 → Medium (topic/intent summaries, ~200 tokens)
                 → Facts extraction (key facts, ~150 tokens)
                 → Cross-session (session summary, ~200 tokens)
"""

import json
import time

import httpx

from memory_store import MemoryStore, TIER_BUDGETS, estimate_tokens


# ─── Prompts for LLM-based compression ────────────────────────────────────────

LIGHT_COMPRESS_PROMPT = """Compress this conversation excerpt. Remove filler, pleasantries, and repetition. Keep the substance — what was discussed, what the user revealed, what commitments were made. Preserve the user's exact words for important statements (names, preferences, emotions).

Output ONLY the compressed text, no preamble.

Target: {budget} tokens maximum.

Conversation:
{text}"""

MEDIUM_COMPRESS_PROMPT = """Summarize this conversation into topic and intent summaries. What subjects were discussed? What did the user want? What was the outcome? Be concise.

Output ONLY the summary, no preamble.

Target: {budget} tokens maximum.

Content:
{text}"""

FACTS_EXTRACT_PROMPT = """Extract key facts from this conversation that should be remembered permanently. Focus on:
- User's name, identity, role
- Stated preferences ("I like...", "I prefer...", "I hate...")
- Emotional moments or significant reactions
- Commitments made by either party ("I'll...", "Next time...")
- Important personal information (job, family, health, hobbies)

Output ONLY a bulleted list of facts, one per line starting with "- ". If no important facts, output "- No notable facts".

Conversation:
{text}"""

SESSION_SUMMARY_PROMPT = """Write a one-paragraph summary of this conversation session for future reference. Include: who the user is (if known), what they talked about, their emotional tone, and any unresolved topics.

Output ONLY the summary paragraph, no preamble.

Target: {budget} tokens maximum.

Session transcript:
{text}"""


FACT_RECONCILE_PROMPT = """You maintain the long-term memory about a kiosk user. Below are the CURRENT stored facts and the transcript of their newest conversation session.

Rewrite the COMPLETE fact list, applying these rules:
- Keep facts that are still true and unaffected by the session.
- ADD new lasting facts from the session (identity, preferences, family, work, hobbies, commitments).
- CONVERSATIONAL PREFERENCES AND BOUNDARIES are first-class facts. If the user asks to stop discussing a topic, calls a topic settled, states how they want to be addressed, or corrects the record, store it — phrase boundaries as: "Topic boundary: <topic> — do not bring it up; the user will raise it if they want." A boundary supersedes older enthusiasm facts about that topic: keep the underlying interest fact only if it still matters, and never phrase anything in a way that invites raising the closed topic.
- UPDATE facts the session contradicts or revises (age, job, pets, plans). Never keep both the old and new version.
- DROP transient states (moods, being busy on a particular day), sentence fragments, truncated or meaningless entries, duplicates, and anything the session shows to be wrong.
- One concise sentence per fact. Every fact must be at least four words.

Output ONLY the revised list, one fact per line starting with "- ". No preamble, no commentary. If there are no facts worth keeping at all, output "- No notable facts".

CURRENT FACTS:
{facts}

NEW SESSION:
{text}"""


FACT_CONSOLIDATION_PROMPT = """Below are all stored facts about a user. Many are redundant, contradictory, or outdated.

Consolidate them into a clean, deduplicated list. Rules:
- Merge duplicates into one fact (keep the most complete/recent version)
- If facts contradict (e.g., "cat" vs "dog"), keep the most specific/recent
- Remove facts that are subsets of other facts
- Preserve ALL unique information — don't drop facts that aren't duplicated
- Keep each fact to one concise sentence
- Output ONLY a bulleted list, one fact per line starting with "- "

Current facts:
{facts}"""


LIVE_FACTS_PROMPT = """Extract key facts from this single conversation exchange that should be remembered permanently. Focus on:
- User's name or identity ("I'm Alex", "my name is...")
- Stated preferences ("I like...", "I prefer...", "I hate...", "I'm allergic to...")
- Personal information (job, family, health, hobbies, location)
- Emotional revelations ("I've been struggling with...", "I'm excited about...")
- Commitments or plans ("I'll be back tomorrow", "Next time I want to...")

Rules:
- Output ONLY facts worth remembering long-term. Skip pleasantries and small talk.
- One fact per line, starting with "- ".
- If nothing worth pinning, output exactly: NONE
- Do NOT invent or infer facts not explicitly stated.
- Keep each fact to one concise sentence.

User said: {user_text}
The assistant replied: {assistant_text}"""


class MemoryCompressor:
    """Compresses conversation transcripts into tiered memory using LLM summarization."""

    def __init__(
        self,
        store: MemoryStore,
        llm_url: str = "http://localhost:8100/llm/generate",
    ):
        self._store = store
        self._llm_url = llm_url
        self._client = httpx.Client(timeout=60.0)

    def close(self):
        self._client.close()

    def _llm_call(self, prompt: str, max_tokens: int = 512) -> str:
        """Call the LLM Gateway for compression. Uses a dedicated session."""
        resp = self._client.post(
            self._llm_url,
            json={
                "messages": [
                    {"role": "system", "content": "You are a precise text compression assistant. Follow instructions exactly. Output only what is requested."},
                    {"role": "user", "content": prompt},
                ],
                "session_id": "memory-compressor",
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "stream": False,
            },
        )
        result = resp.json()
        return result.get("text", "").strip()

    def compress_session(self, user_id: str) -> dict:
        """
        Compress all uncompressed transcripts for a user.
        Returns a summary of what was compressed.
        """
        transcripts = self._store.get_uncompressed_transcripts(user_id)
        if not transcripts:
            return {"status": "no_uncompressed_transcripts"}

        results = {"transcripts_processed": 0, "tiers_updated": []}

        for transcript_id, turns in transcripts:
            # Format transcript as readable text
            transcript_text = self._format_transcript(turns)
            transcript_tokens = estimate_tokens(transcript_text)

            if transcript_tokens < 10:
                self._store.mark_transcript_compressed(transcript_id)
                continue

            # 1. Verbatim tier: store last 5 turns lossless
            verbatim_turns = turns[-10:]  # last 5 exchanges (user+assistant = 10 entries)
            verbatim_text = self._format_transcript(verbatim_turns)
            if estimate_tokens(verbatim_text) <= TIER_BUDGETS["verbatim"]:
                self._store.set_tier(user_id, "verbatim", verbatim_text)
                results["tiers_updated"].append("verbatim")

            # 2. Light compression: full transcript → remove filler
            if transcript_tokens > TIER_BUDGETS["verbatim"]:
                light_text = self._llm_call(
                    LIGHT_COMPRESS_PROMPT.format(
                        budget=TIER_BUDGETS["light"],
                        text=transcript_text,
                    ),
                    max_tokens=TIER_BUDGETS["light"] * 2,
                )
                if light_text:
                    self._store.set_tier(user_id, "light", light_text)
                    results["tiers_updated"].append("light")

            # 3. Medium compression: summarize into topics/intents
            medium_text = self._llm_call(
                MEDIUM_COMPRESS_PROMPT.format(
                    budget=TIER_BUDGETS["medium"],
                    text=transcript_text,
                ),
                max_tokens=TIER_BUDGETS["medium"] * 2,
            )
            if medium_text:
                self._store.set_tier(user_id, "medium", medium_text)
                results["tiers_updated"].append("medium")

            # 4. Fact RECONCILIATION: the LLM rewrites the whole fact list
            # against the new session: add / update / supersede / drop.
            # Plain extract-and-append loses boundaries and corrections and
            # lets junk accumulate.
            rec = self.reconcile_facts(user_id, transcript_text)
            if rec.get("status") == "ok":
                results["tiers_updated"].append("facts")
                results["facts_reconciled"] = {
                    "before": rec["before"], "after": rec["after"],
                }
            else:
                results["facts_reconcile_status"] = rec.get("status")

            # 5. Cross-session summary
            summary_text = self._llm_call(
                SESSION_SUMMARY_PROMPT.format(
                    budget=TIER_BUDGETS["cross_session"],
                    text=transcript_text,
                ),
                max_tokens=TIER_BUDGETS["cross_session"] * 2,
            )
            if summary_text:
                self._store.set_tier(user_id, "cross_session", summary_text)
                results["tiers_updated"].append("cross_session")

            self._store.mark_transcript_compressed(transcript_id)
            results["transcripts_processed"] += 1

        # (Per-session reconciliation above covers fact consolidation;
        # consolidate_facts() is for manual/admin cleanup.)
        return results

    def reconcile_facts(self, user_id: str, transcript_text: str) -> dict:
        """LLM rewrite of the user's complete fact list against a new session.

        Handles adds, updates, supersessions (incl. topic boundaries), and
        junk/stale removal in one pass. Guardrails: a failed or empty LLM
        response leaves the store untouched, and a rewrite that shrinks an
        established fact list by more than half is rejected as suspicious
        (protects memory from one bad LLM call).
        """
        existing = self._store.get_all_facts(user_id)
        facts_block = "\n".join(f"- {f}" for f in existing) if existing else "(none yet)"

        raw = self._llm_call(
            FACT_RECONCILE_PROMPT.format(facts=facts_block, text=transcript_text),
            max_tokens=600,
        )
        if not raw or not raw.strip():
            return {"status": "llm_failed"}

        new_facts = []
        for line in raw.strip().splitlines():
            fact = line.strip().lstrip("-•* ").strip()
            # Junk guard: no fragments under four words (e.g. "User is concerned")
            if fact and len(fact.split()) >= 4 and fact.lower() != "no notable facts":
                new_facts.append(fact)

        if not new_facts and existing:
            return {"status": "parse_failed", "raw": raw[:200]}
        if len(existing) > 6 and len(new_facts) < 0.5 * len(existing):
            print(f"[MemoryCompressor] Reconcile REJECTED for user {user_id}: "
                  f"{len(existing)} -> {len(new_facts)} facts (suspicious shrinkage)")
            return {"status": "suspicious_shrinkage",
                    "before": len(existing), "proposed": len(new_facts)}

        self._store.replace_all_facts(user_id, new_facts)
        print(f"[MemoryCompressor] Reconciled facts for user {user_id}: "
              f"{len(existing)} -> {len(new_facts)}")
        return {"status": "ok", "before": len(existing), "after": len(new_facts),
                "facts": new_facts}

    def consolidate_facts(self, user_id: str) -> dict:
        """
        LLM-based fact consolidation. Reads all stored facts, asks the LLM
        to merge duplicates and resolve contradictions, replaces the facts tier.
        """
        existing = self._store.get_all_facts(user_id)
        if len(existing) <= 1:
            return {"status": "not_needed", "count": len(existing)}

        facts_text = "\n".join(f"- {f}" for f in existing)
        consolidated_raw = self._llm_call(
            FACT_CONSOLIDATION_PROMPT.format(facts=facts_text),
            max_tokens=400,
        )

        if not consolidated_raw or not consolidated_raw.strip():
            return {"status": "llm_failed"}

        # Parse consolidated facts
        new_facts = []
        for line in consolidated_raw.strip().splitlines():
            fact = line.strip().lstrip("- ").strip()
            if fact and len(fact) > 3:
                new_facts.append(fact)

        if not new_facts:
            return {"status": "parse_failed", "raw": consolidated_raw}

        # Replace all facts with consolidated set
        self._store.replace_all_facts(user_id, new_facts)

        print(f"[MemoryCompressor] Consolidated {len(existing)} facts → {len(new_facts)} for user {user_id}")
        return {
            "status": "ok",
            "before": len(existing),
            "after": len(new_facts),
            "facts": new_facts,
        }

    def extract_facts_from_turn(self, user_id: str, user_text: str, assistant_text: str) -> dict:
        """
        Extract pinned facts from a single conversation turn.
        Deduplicates against existing facts before inserting.
        Returns summary of what was extracted.
        """
        # Ask LLM to extract facts
        raw = self._llm_call(
            LIVE_FACTS_PROMPT.format(
                user_text=user_text,
                assistant_text=assistant_text,
            ),
            max_tokens=200,
        )

        if not raw or raw.strip().upper() == "NONE":
            return {"status": "no_facts", "extracted": 0}

        # Parse facts from LLM output
        candidates = []
        for line in raw.strip().splitlines():
            fact = line.strip().lstrip("- ").strip()
            if fact and len(fact) > 5:
                candidates.append(fact)

        if not candidates:
            return {"status": "no_facts", "extracted": 0}

        # Deduplicate against existing facts
        existing = self._store.get_all_facts(user_id)
        existing_lower = [f.lower() for f in existing]

        new_facts = []
        for fact in candidates:
            # Skip if substantially similar to an existing fact
            if self._is_duplicate(fact, existing_lower):
                continue
            new_facts.append(fact)

        # Insert new facts as pinned
        for fact in new_facts:
            self._store.add_fact(user_id, fact, pinned=True)

        return {
            "status": "ok",
            "extracted": len(new_facts),
            "skipped_duplicates": len(candidates) - len(new_facts),
            "facts": new_facts,
        }

    @staticmethod
    def _is_duplicate(new_fact: str, existing_lower: list[str]) -> bool:
        """Check if a fact is substantially similar to any existing fact."""
        new_lower = new_fact.lower()

        for existing in existing_lower:
            # Exact match
            if new_lower == existing:
                return True
            # Substring containment (either direction)
            if new_lower in existing or existing in new_lower:
                return True
            # Word overlap: if >70% of words match, it's a duplicate
            new_words = set(new_lower.split())
            existing_words = set(existing.split())
            if not new_words or not existing_words:
                continue
            overlap = len(new_words & existing_words)
            smaller = min(len(new_words), len(existing_words))
            if smaller > 0 and overlap / smaller > 0.7:
                return True

        return False

    def _format_transcript(self, turns: list[dict]) -> str:
        """Format turns into readable text for the LLM."""
        lines = []
        for turn in turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
        return "\n".join(lines)
