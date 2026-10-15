"""
User Identification State Machine

Conversational identification flow:
  1. Ask name (optional — user can refuse)
  2. If name given → check DB for match
     - Match found → confirm identity → verify PIN → persistent session
     - No match → offer to save → if yes, create PIN → persistent
  3. If name refused → anonymous mode (conversation works, nothing persists)

The flow is conversational, not a gate — the user can start chatting
at any point by declining to identify.
"""

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger("user_id")


class IdState(Enum):
    """Identification state machine states."""
    IDLE = "idle"                        # no face, waiting
    GREETING = "greeting"                # new face, asking name
    GREETING_RETRY = "greeting_retry"    # name extraction failed, asking again
    CONFIRMING_NAME = "confirming_name"  # "Did I get that right, Joanna?"
    CONFIRMING_MATCH = "confirming_match"  # DB match found, "are you X?"
    VERIFYING_PIN = "verifying_pin"      # confirmed match, asking PIN
    ASKING_SAVE = "asking_save"          # no match, "want me to remember you?"
    ASKING_NEW_PIN = "asking_new_pin"    # user wants to save, asking for PIN
    CONFIRMING_PIN = "confirming_pin"    # readback PIN confirmation
    IDENTIFIED = "identified"            # fully identified, persistent session
    ANONYMOUS = "anonymous"              # user declined, no persistence


# Regex for extracting a 4-digit PIN from speech
PIN_RE = re.compile(r'\b(\d{4})\b')

# NOTE: there are deliberately NO yes/no or name word-lists here. Spoken
# affirmation and denial are open-ended ("that's me", "who else would I be")
# and a list can only ever cover the phrasings someone thought of. The LLM
# classifies every confirmation and extracts every name; when it is
# unreachable the orchestrator says so out loud rather than guessing (see
# _speak_degraded_mode). Only PIN digits and the explicit refusal opt-out
# below stay pattern-matched — those are closed sets, not open language.

# Phrases that mean "I don't want to identify"
REFUSE_RE = re.compile(
    r"(?:i'?d rather not|no thanks|skip|don'?t (?:want to|save|store|remember)|"
    r"prefer not|anonymous|just chat|private|pass)\b",
    re.IGNORECASE,
)


class IdentificationManager:
    """Manages the conversational user identification flow.

    Call `on_face_event()` when face recognition fires an event.
    Call `process_utterance()` with each user transcript during identification.
    Check `state` and `avatar_line` for what the avatar should say.
    Check `is_identifying` to know if utterances should bypass the LLM.
    """

    def __init__(self, avatar_name: str = "Kiah"):
        self.avatar_name = avatar_name
        self._state = IdState.IDLE
        self._pending_name: Optional[str] = None
        self._pending_pin: Optional[str] = None
        self._pending_user_id: Optional[int] = None
        self._pending_user_name: Optional[str] = None
        self._pending_similarity: float = 0.0
        self._avatar_line: Optional[str] = None
        self._pin_attempts: int = 0
        self._name_retried: bool = False
        self.MAX_PIN_ATTEMPTS = 2
        # Per-turn LLM verdicts (set by process_utterance; None = LLM
        # unreachable, which is never guessed around).
        self._turn_intent: Optional[str] = None       # "yes" | "no" | "unclear" | None
        self._turn_name_guess: Optional[str] = None   # extracted name | "" (no name) | None
        self._unclear_count: int = 0                  # consecutive unclear confirms

    @property
    def state(self) -> IdState:
        return self._state

    @property
    def is_identifying(self) -> bool:
        """True during the identification flow (utterances bypass the LLM)."""
        return self._state not in (IdState.IDLE, IdState.IDENTIFIED, IdState.ANONYMOUS)

    @property
    def is_anonymous(self) -> bool:
        return self._state == IdState.ANONYMOUS

    @property
    def avatar_line(self) -> Optional[str]:
        """Get and clear the next line for the avatar to speak."""
        line = self._avatar_line
        self._avatar_line = None
        return line

    def reset(self):
        """Reset to idle state (e.g., when face is lost)."""
        self._state = IdState.IDLE
        self._pending_name = None
        self._pending_pin = None
        self._pending_user_id = None
        self._pending_user_name = None
        self._pending_similarity = 0.0
        self._avatar_line = None
        self._pin_attempts = 0
        self._name_retried = False
        self._turn_intent = None
        self._turn_name_guess = None
        self._unclear_count = 0

    # ── LLM-intent helpers ───────────────────────────────────────────────
    # The orchestrator classifies each confirm-state utterance with the LLM
    # (grammar-forced yes/no/unclear) and extracts names with the LLM, then
    # passes the verdict in. An absent verdict means the LLM could not be
    # reached; that is NOT a cue to guess from a word list, so both read
    # False and the caller takes its unclear branch. In practice the
    # orchestrator intercepts that case earlier and speaks up instead.

    def _says_yes(self) -> bool:
        return self._turn_intent == "yes"

    def _says_no(self) -> bool:
        return self._turn_intent == "no"

    @property
    def expected_input(self) -> str:
        """What kind of reply the current state is waiting for — tells the
        orchestrator which LLM resolvers to run before process_utterance().
        "yesno" | "name" | "yesno_name" | "pin" | "none"
        """
        s = self._state
        if s in (IdState.CONFIRMING_MATCH, IdState.ASKING_SAVE, IdState.CONFIRMING_PIN):
            return "yesno"
        if s == IdState.CONFIRMING_NAME:
            return "yesno_name"      # yes/no OR a corrected name
        if s in (IdState.GREETING, IdState.GREETING_RETRY):
            return "name"
        if s in (IdState.VERIFYING_PIN, IdState.ASKING_NEW_PIN):
            return "pin"
        return "none"

    @property
    def pending_question(self) -> str:
        """The yes/no question currently on the table (classifier context)."""
        s = self._state
        if s == IdState.CONFIRMING_MATCH:
            return f"Are you {self._pending_user_name}?"
        if s == IdState.CONFIRMING_NAME:
            return f"Is your name {self._pending_name}?"
        if s == IdState.ASKING_SAVE:
            return "Do you want me to remember you for next time?"
        if s == IdState.CONFIRMING_PIN:
            return "Did I hear your PIN correctly?"
        return "Is that right?"

    # ── Face events ──────────────────────────────────────────────────────

    def on_face_event(self, event: str, user_name: Optional[str] = None,
                      user_id: Optional[int] = None, similarity: float = 0.0):
        """Called by the orchestrator when face recognition fires an event."""
        if event == "new_face":
            self._state = IdState.GREETING
            self._avatar_line = (
                "<<emotion:happy intensity:0.7>><<gesture:wave>> "
                f"Hi there! I'm {self.avatar_name}. What's your name?"
            )
            logger.info("[ID] New face → greeting")

        elif event == "user_identified":
            self._pending_user_id = user_id
            self._pending_user_name = user_name
            self._pending_similarity = similarity

            if similarity >= 0.65:
                # Confident match — ask if it's them
                self._state = IdState.CONFIRMING_MATCH
                self._avatar_line = (
                    f"<<emotion:happy intensity:0.8>><<gesture:wave>> "
                    f"Hey! You look like {user_name} — is that you?"
                )
                logger.info(f"[ID] Confident match ({user_name}, sim={similarity:.3f}) → confirming")
            else:
                # Soft match — still ask name, mention resemblance
                self._state = IdState.GREETING
                self._pending_user_id = user_id
                self._pending_user_name = user_name
                self._avatar_line = (
                    f"<<emotion:happy intensity:0.6>><<gesture:wave>> "
                    f"Hi there! I'm {self.avatar_name}. You look a bit like someone I know called {user_name}. "
                    f"What's your name?"
                )
                logger.info(f"[ID] Soft match ({user_name}, sim={similarity:.3f}) → greeting with hint")

        elif event == "user_left":
            self.reset()
            logger.info("[ID] Face lost → reset")

    # ── Utterance processing ─────────────────────────────────────────────

    def process_utterance(self, transcript: str,
                          intent: Optional[str] = None,
                          name_guess: Optional[str] = None) -> Optional[str]:
        """Process a user utterance during identification.

        intent: LLM-classified reading of the reply ("yes"|"no"|"unclear"),
            or None when the classifier was unavailable (regex fallback).
        name_guess: LLM-extracted name; "" = LLM says no name present;
            None = extractor unavailable (regex fallback).

        Returns an action string for the orchestrator:
            "check_db:NAME"         — look up name in face DB
            "register:NAME:PIN"     — create new user with name and PIN
            "check_pin:USER_ID:PIN" — verify PIN for existing user
            "anonymous"             — enter anonymous mode
            "identified:USER_ID"    — identification complete (PIN verified)
            None                    — still in flow, check avatar_line
        """
        transcript = transcript.strip()
        self._turn_intent = intent
        self._turn_name_guess = name_guess
        if intent in ("yes", "no"):
            self._unclear_count = 0

        if self._state == IdState.GREETING:
            return self._handle_greeting(transcript)

        elif self._state == IdState.GREETING_RETRY:
            return self._handle_greeting(transcript)

        elif self._state == IdState.CONFIRMING_NAME:
            return self._handle_name_confirmation(transcript)

        elif self._state == IdState.CONFIRMING_MATCH:
            return self._handle_match_confirmation(transcript)

        elif self._state == IdState.VERIFYING_PIN:
            return self._handle_pin_verification(transcript)

        elif self._state == IdState.ASKING_SAVE:
            return self._handle_save_decision(transcript)

        elif self._state == IdState.ASKING_NEW_PIN:
            return self._handle_new_pin(transcript)

        elif self._state == IdState.CONFIRMING_PIN:
            return self._handle_pin_confirmation(transcript)

        return None

    # ── State handlers ───────────────────────────────────────────────────

    def _handle_greeting(self, transcript: str) -> Optional[str]:
        """Handle response to 'What's your name?'"""
        # Check for refusal first
        if REFUSE_RE.search(transcript):
            self._state = IdState.ANONYMOUS
            self._avatar_line = (
                "<<emotion:happy intensity:0.6>><<gesture:nod>> "
                "No problem at all! Let's just chat. What can I help you with?"
            )
            logger.info("[ID] User refused to identify → anonymous")
            return "anonymous"

        # (Deflections like "what do you mean?" need no special pattern: the
        # LLM extractor returns "" for them, which is the no-name path below.)

        # The LLM extractor is the only namer: "" means it saw no name in this
        # reply (e.g. "Hi.", "what do you mean?") and None means it could not
        # be reached. Neither is ever enrolled.
        name = self._turn_name_guess or None
        if not name:
            if not self._name_retried:
                self._name_retried = True
                self._state = IdState.GREETING_RETRY
                self._avatar_line = (
                    "<<emotion:happy intensity:0.5>> "
                    "Sorry, I didn't catch that. Could you say your name again? Or say 'skip' to just chat."
                )
                logger.info(f"[ID] Couldn't extract name from: {transcript}")
                return None
            else:
                self._state = IdState.ANONYMOUS
                self._avatar_line = (
                    "<<emotion:happy intensity:0.5>><<gesture:nod>> "
                    "No worries! Let's just chat. What can I help you with?"
                )
                logger.info("[ID] Name retry failed → anonymous")
                return "anonymous"

        # Got a name — confirm it before proceeding
        self._pending_name = name
        self._state = IdState.CONFIRMING_NAME
        self._avatar_line = (
            f"<<emotion:happy intensity:0.7>> "
            f"Nice to meet you, {name}! Did I get that right?"
        )
        logger.info(f"[ID] Got name: {name} → confirming")
        return None

    def _handle_name_confirmation(self, transcript: str) -> Optional[str]:
        """Handle 'Did I get your name right?' yes/no/correction."""
        if REFUSE_RE.search(transcript):
            self._state = IdState.ANONYMOUS
            self._avatar_line = (
                "<<emotion:happy intensity:0.6>><<gesture:nod>> "
                "No problem! Let's just chat. What can I help you with?"
            )
            logger.info("[ID] User refused at name confirmation → anonymous")
            return "anonymous"

        if self._says_yes():
            # Name confirmed — check DB
            logger.info(f"[ID] Name confirmed: {self._pending_name} → checking DB")
            return f"check_db:{self._pending_name}"

        # Try extracting a corrected name (handles "No, it's Sarah" / "Call me Sarah")
        new_name = self._turn_name_guess
        if new_name and new_name.lower() != self._pending_name.lower():
            self._pending_name = new_name
            self._avatar_line = (
                f"<<emotion:happy intensity:0.6>> "
                f"Oh, {new_name}! Did I get it right this time?"
            )
            logger.info(f"[ID] Name correction: {new_name} → re-confirming")
            return None

        if self._says_no():
            # Plain "no" without a correction — ask again
            self._state = IdState.GREETING
            self._pending_name = None
            self._avatar_line = (
                "<<emotion:happy intensity:0.5>> "
                "Sorry about that! What should I call you?"
            )
            logger.info("[ID] Name rejected → asking again")
            return None

        # Unclear — one warm reprompt, then just re-ask the name fresh
        self._unclear_count += 1
        if self._unclear_count >= 2:
            self._unclear_count = 0
            self._state = IdState.GREETING
            self._pending_name = None
            self._avatar_line = (
                "<<emotion:happy intensity:0.5>> "
                "Let's start over — what should I call you?"
            )
            logger.info("[ID] Name confirm unclear twice → re-asking name")
            return None
        self._avatar_line = (
            f"<<emotion:happy intensity:0.5>> "
            f"Sorry — is {self._pending_name} right?"
        )
        return None

    def on_db_result(self, matched: bool, user_id: Optional[int] = None,
                     user_name: Optional[str] = None, similarity: float = 0.0):
        """Called by orchestrator after checking the name against the face DB.

        Args:
            matched: True if a user with this name exists
            user_id: matched user's ID
            user_name: matched user's name (from DB)
            similarity: face similarity to the matched user
        """
        if matched and user_id is not None:
            self._pending_user_id = user_id
            self._pending_user_name = user_name
            self._state = IdState.VERIFYING_PIN
            self._pin_attempts = 0
            self._avatar_line = (
                f"<<emotion:happy intensity:0.8>><<gesture:nod>> "
                f"Welcome back, {user_name}! Could you tell me your PIN?"
            )
            logger.info(f"[ID] DB match for {user_name} (id={user_id}) → asking PIN")
        else:
            # No match — offer to save
            self._state = IdState.ASKING_SAVE
            self._avatar_line = (
                f"<<emotion:happy intensity:0.7>><<gesture:nod>> "
                f"Nice to meet you, {self._pending_name}! "
                f"Would you like me to remember you for next time?"
            )
            logger.info(f"[ID] No DB match for {self._pending_name} → offering to save")

    def _handle_match_confirmation(self, transcript: str) -> Optional[str]:
        """Handle 'Are you X?' confirmation for confident face match."""
        if REFUSE_RE.search(transcript):
            self._state = IdState.ANONYMOUS
            self._avatar_line = (
                "<<emotion:happy intensity:0.6>><<gesture:nod>> "
                "No problem! Let's just chat. What can I help you with?"
            )
            logger.info("[ID] User refused at confirmation → anonymous")
            return "anonymous"
        if self._says_yes():
            self._state = IdState.VERIFYING_PIN
            self._pin_attempts = 0
            self._avatar_line = (
                f"<<emotion:happy intensity:0.7>> "
                f"Great! What's your PIN, {self._pending_user_name}?"
            )
            logger.info(f"[ID] Confirmed as {self._pending_user_name} → asking PIN")
        elif self._says_no():
            # Not them — ask for real name
            self._state = IdState.GREETING
            self._pending_user_id = None
            self._pending_user_name = None
            self._avatar_line = (
                "<<emotion:happy intensity:0.6>><<gesture:nod>> "
                "My mistake! What's your name?"
            )
            logger.info("[ID] Match rejected → asking name")
        else:
            # Unclear — one warm reprompt, then fall to the name flow
            self._unclear_count += 1
            if self._unclear_count >= 2:
                self._unclear_count = 0
                self._state = IdState.GREETING
                self._pending_user_id = None
                self._pending_user_name = None
                self._avatar_line = (
                    "<<emotion:happy intensity:0.6>><<gesture:nod>> "
                    "No worries — what's your name, then?"
                )
                logger.info("[ID] Match confirm unclear twice → asking name")
            else:
                self._avatar_line = (
                    f"<<emotion:happy intensity:0.5>> "
                    f"Sorry — was that a yes? Are you {self._pending_user_name}?"
                )
        return None

    def _handle_pin_verification(self, transcript: str) -> Optional[str]:
        """Handle PIN input for returning user."""
        # Check for refusal/forgot
        if REFUSE_RE.search(transcript):
            self._state = IdState.ANONYMOUS
            self._avatar_line = (
                "<<emotion:happy intensity:0.6>><<gesture:nod>> "
                "That's fine! Let's just chat. What can I help you with?"
            )
            logger.info("[ID] User refused PIN → anonymous")
            return "anonymous"

        if re.search(r'\b(forgot|don\'?t remember|can\'?t remember|no idea|lost it)\b',
                      transcript, re.IGNORECASE):
            # Forgot PIN — offer to create new profile
            self._state = IdState.ASKING_SAVE
            self._avatar_line = (
                "<<emotion:happy intensity:0.5>><<gesture:nod>> "
                f"No worries, {self._pending_user_name}! Would you like me to set up a new PIN for you?"
            )
            logger.info("[ID] PIN forgotten → offering new save")
            return None

        pin = self._extract_pin(transcript)
        if pin:
            return f"check_pin:{self._pending_user_id}:{pin}"
        else:
            self._avatar_line = (
                "<<emotion:confused intensity:0.4>> "
                "I need a four digit PIN. Could you say it for me? Or say 'skip' to chat without signing in."
            )
            logger.info(f"[ID] Couldn't extract PIN from: {transcript}")
        return None

    def on_pin_result(self, verified: bool):
        """Called by orchestrator after checking PIN against FaceLibrary."""
        if verified:
            self._state = IdState.IDENTIFIED
            name = self._pending_user_name or self._pending_name or "friend"
            self._avatar_line = (
                f"<<emotion:happy intensity:0.9>><<gesture:nod>> "
                f"Welcome back, {name}! What can I help you with today?"
            )
            logger.info(f"[ID] PIN verified for {name}")
        else:
            self._pin_attempts += 1
            if self._pin_attempts >= self.MAX_PIN_ATTEMPTS:
                self._state = IdState.ASKING_SAVE
                self._avatar_line = (
                    "<<emotion:happy intensity:0.5>><<gesture:nod>> "
                    "That didn't match. Would you like me to set you up with a new PIN?"
                )
                logger.info("[ID] PIN failed max attempts → offering new save")
            else:
                self._avatar_line = (
                    "<<emotion:confused intensity:0.5>> "
                    "Hmm, that doesn't match. Want to try again, or say 'skip' to just chat?"
                )
                logger.info(f"[ID] PIN wrong, attempt {self._pin_attempts}")

    def _handle_save_decision(self, transcript: str) -> Optional[str]:
        """Handle 'Want me to remember you?' yes/no."""
        if self._says_yes():
            self._state = IdState.ASKING_NEW_PIN
            self._avatar_line = (
                "<<emotion:happy intensity:0.7>> "
                "Great! Pick a four digit PIN so I can recognize you next time."
            )
            logger.info("[ID] User wants to save → asking for new PIN")
        elif self._says_no() or REFUSE_RE.search(transcript):
            self._state = IdState.ANONYMOUS
            name = self._pending_name or ""
            if name:
                self._avatar_line = (
                    f"<<emotion:happy intensity:0.6>><<gesture:nod>> "
                    f"Sure thing, {name}! Let's chat. What can I help you with?"
                )
            else:
                self._avatar_line = (
                    "<<emotion:happy intensity:0.6>><<gesture:nod>> "
                    "Sure thing! Let's chat. What can I help you with?"
                )
            logger.info("[ID] User declined to save → anonymous")
            return "anonymous"
        else:
            # Unclear — one warm reprompt, then default to not saving
            self._unclear_count += 1
            if self._unclear_count >= 2:
                self._unclear_count = 0
                self._state = IdState.ANONYMOUS
                self._avatar_line = (
                    "<<emotion:happy intensity:0.6>><<gesture:nod>> "
                    "No pressure — let's just chat for now. What can I help you with?"
                )
                logger.info("[ID] Save decision unclear twice → anonymous")
                return "anonymous"
            self._avatar_line = (
                "<<emotion:happy intensity:0.5>> "
                "Sorry — should I remember you for next time?"
            )
        return None

    def _handle_new_pin(self, transcript: str) -> Optional[str]:
        """Extract PIN from new user, then confirm."""
        if REFUSE_RE.search(transcript):
            self._state = IdState.ANONYMOUS
            self._avatar_line = (
                "<<emotion:happy intensity:0.6>><<gesture:nod>> "
                "No problem! Let's just chat. What can I help you with?"
            )
            logger.info("[ID] User refused PIN creation → anonymous")
            return "anonymous"

        pin = self._extract_pin(transcript)
        if pin:
            self._pending_pin = pin
            self._state = IdState.CONFIRMING_PIN
            digits = " ".join(pin)  # "1 2 3 4" for clear readback
            self._avatar_line = (
                f"<<emotion:happy intensity:0.6>> "
                f"I heard {digits}. Is that right?"
            )
            logger.info(f"[ID] Got PIN {pin} → confirming")
        else:
            self._avatar_line = (
                "<<emotion:confused intensity:0.4>> "
                "I need a four digit number, like one two three four. Or say 'skip' to chat without saving."
            )
            logger.info(f"[ID] Couldn't extract PIN from: {transcript}")
        return None

    def _handle_pin_confirmation(self, transcript: str) -> Optional[str]:
        """Handle yes/no confirmation of PIN readback."""
        if self._says_yes():
            self._state = IdState.IDENTIFIED
            self._avatar_line = (
                f"<<emotion:happy intensity:0.9>><<gesture:nod>> "
                f"All set, {self._pending_name}! I'll remember you next time. "
                f"So, what can I help you with today?"
            )
            logger.info(f"[ID] PIN confirmed, registering {self._pending_name}")
            return f"register:{self._pending_name}:{self._pending_pin}"
        elif self._says_no():
            self._state = IdState.ASKING_NEW_PIN
            self._avatar_line = (
                "<<emotion:happy intensity:0.5>> "
                "No problem! Let's try again. What four digit PIN would you like?"
            )
            logger.info("[ID] PIN rejected, asking again")
        else:
            # Unclear — one warm reprompt, then re-collect the PIN fresh
            self._unclear_count += 1
            if self._unclear_count >= 2:
                self._unclear_count = 0
                self._state = IdState.ASKING_NEW_PIN
                self._avatar_line = (
                    "<<emotion:happy intensity:0.5>> "
                    "Let's try the PIN once more — what four digits would you like?"
                )
                logger.info("[ID] PIN confirm unclear twice → re-asking PIN")
            else:
                self._avatar_line = (
                    "<<emotion:happy intensity:0.5>> "
                    "Sorry — did I get your PIN right?"
                )
        return None

    # ── Extraction helpers ───────────────────────────────────────────────

    @staticmethod
    def _extract_pin(transcript: str) -> Optional[str]:
        """Extract a 4-digit PIN from transcript, tolerating ASR formatting.

        Handles: "1234", "1 2 3 4", "1-2-3-4", "1.2.3.4", "1,2,3,4",
        "one two three four", "1 two 3 four", leading/trailing junk, etc.
        """
        word_map = {
            "zero": "0", "oh": "0", "o": "0",
            "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        }
        # Walk tokens in order: single digits OR digit-words
        tokens = re.findall(r"\d|[a-z]+", transcript.lower())
        digits = []
        for tok in tokens:
            if tok.isdigit():
                digits.append(tok)
            elif tok in word_map:
                digits.append(word_map[tok])

        if len(digits) >= 4:
            return "".join(digits[:4])
        return None
