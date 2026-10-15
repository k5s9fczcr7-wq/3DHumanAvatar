"""
Knowledge base, whole-KB-in-context (the simplest mode).

At startup the orchestrator finds a knowledge-base folder (kiosk_config.ini
[kb] path, or else the first folder with text files on a removable drive),
reads its text files, and puts them in her system prompt after the persona
and before any per-session memory. The KB is then a stable prefix that llama
keeps cached across turns, so after one pre-read at startup it costs little
per reply. Needs the 262,144-token context (start-llama.bat); a KB larger
than KB_TOKEN_BUDGET is refused rather than crowding out the conversation.

Names, for any KB: speech recognition turns names it doesn't know into
ordinary words, and the voice guesses at their pronunciation. So the KB's own
proper names (found by kb_names, no KB-specific files needed) become a
vocabulary hint for speech recognition, and pronunciations.txt respellings
change only what the voice reads, never what she writes or remembers. That
file can sit beside the KB on the drive (its names travel with it) and next
to kiosk_config.ini (the kiosk's own); the drive's entries win.

Not supported: retrieval for larger KBs, office formats, a drive watcher with
spoken announcements, KB identity/redeploy handling, a coverage report.
"""

import collections
import configparser
import ctypes
import logging
import os
import re
import string
from typing import Optional

from .gaze_calculator import CONFIG_PATH

logger = logging.getLogger("knowledge")

PRONUNCIATIONS_FILE = "pronunciations.txt"   # for her voice, so never read as KB text
KIOSK_PRONUNCIATIONS = os.path.join(os.path.dirname(CONFIG_PATH), PRONUNCIATIONS_FILE)
# Whisper's hint slot holds 223 tokens (faster-whisper truncates beyond that);
# stay under it so a name is never cut in half.
ASR_HINT_TOKENS = 200

KB_EXTENSIONS = (".md", ".txt", ".csv")
# Of 262,144: leaves >= 52K for persona, memory and conversation. KB text
# with numbers, tables and markdown runs ~3 characters per token, not 4.
# Measured: a 602 KB test KB (117 files) is 193,822 tokens.
KB_TOKEN_BUDGET = 210_000
MAX_FILE_BYTES = 2_000_000    # a single oversized file is skipped, not loaded
# Read first, so the model meets the KB's own map and conventions up front.
LEAD_FILES = ("readme.md", "index.md")

_DRIVE_REMOVABLE = 2


def load_kb_config(path: str = CONFIG_PATH) -> tuple:
    """([kb] enabled, [kb] path) from kiosk_config.ini. Off when missing."""
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8-sig")
    enabled = cfg.getboolean("kb", "enabled", fallback=False)
    folder = cfg.get("kb", "path", fallback="").strip()
    return enabled, folder


def _has_kb_files(folder: str) -> bool:
    for _dirpath, _dirs, files in os.walk(folder):
        if any(f.lower().endswith(KB_EXTENSIONS) for f in files):
            return True
    return False


def removable_drive_roots() -> list:
    """Roots of removable drives (USB sticks), e.g. ["D:\\"]. Windows only."""
    try:
        get_type = ctypes.windll.kernel32.GetDriveTypeW
        mask = ctypes.windll.kernel32.GetLogicalDrives()
    except AttributeError:
        return []
    roots = []
    for i, letter in enumerate(string.ascii_uppercase):
        if mask & (1 << i):
            root = f"{letter}:\\"
            if get_type(root) == _DRIVE_REMOVABLE:
                roots.append(root)
    return roots


def find_kb_folder(configured: str = "", drive_roots: Optional[list] = None) -> Optional[str]:
    """The configured folder if it exists; otherwise the first top-level
    folder on a removable drive that holds KB text files."""
    if configured:
        return configured if os.path.isdir(configured) else None
    for root in (removable_drive_roots() if drive_roots is None else drive_roots):
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        for name in entries:
            folder = os.path.join(root, name)
            if (os.path.isdir(folder) and not name.startswith(("$", "."))
                    and name.lower() != "system volume information" and _has_kb_files(folder)):
                return folder
    return None


def read_kb(folder: str) -> tuple:
    """(text, files_loaded, files_skipped). Lead files first, then the rest in
    path order; each file under a '### FILE:' header with its relative path."""
    paths = []
    for dirpath, dirs, files in os.walk(folder):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(KB_EXTENSIONS):
                paths.append(os.path.join(dirpath, f))
    kiosk_file = os.path.join(folder, PRONUNCIATIONS_FILE).lower()
    paths = [p for p in paths if p.lower() != kiosk_file]

    def order(p):
        rel = os.path.relpath(p, folder).replace("\\", "/")
        lead = rel.lower() in LEAD_FILES
        return (0 if lead else 1, LEAD_FILES.index(rel.lower()) if lead else 0, rel.lower())

    parts, loaded, skipped = [], 0, 0
    for p in sorted(paths, key=order):
        rel = os.path.relpath(p, folder).replace("\\", "/")
        try:
            if os.path.getsize(p) > MAX_FILE_BYTES:
                logger.warning(f"[KB] Skipping {rel}: larger than {MAX_FILE_BYTES:,} bytes")
                skipped += 1
                continue
            with open(p, encoding="utf-8-sig", errors="replace") as fh:
                body = fh.read().strip()
        except OSError as e:
            logger.warning(f"[KB] Skipping {rel}: {e}")
            skipped += 1
            continue
        parts.append(f"### FILE: {rel}\n{body}")
        loaded += 1
    return "\n\n".join(parts), loaded, skipped


def kb_block(name: str, text: str, sounds: Optional[dict] = None) -> str:
    """Her system-prompt section: how to use the KB, the KB itself, how the
    names that need it are pronounced (sounds: name -> respelling, from the
    verified pronunciations - so she can match a name speech recognition
    spelled by sound), then a short reminder - with ~200K tokens between the
    rules and the question, the rules need restating where the conversation
    begins. The guide comes after the KB text so a change to it leaves the
    cached KB prefix intact."""
    guide = ""
    if sounds:
        guide = ("How some names in the knowledge base sound - use this to recognize them when speech "
                 "recognition writes them the way they sound; always write the real name:\n"
                 + "\n".join(f"- {n} sounds like \"{say.lower()}\"" for n, say in sounds.items()) + "\n\n")
    return (
        "## Knowledge base\n"
        f"This kiosk has a knowledge base loaded from its information drive (\"{name}\"), below. "
        "It is the authority for everything it covers: it is newer than your training and "
        "specific to this kiosk.\n"
        "- Answer from it whenever it covers the question.\n"
        "- If it disagrees with what you remember from your training, the knowledge base is right. "
        "If its own files say which source wins when facts conflict, follow that.\n"
        "- If it doesn't cover the question, you may answer from general knowledge, but say so "
        "first, for example \"That's not in my materials, but...\".\n"
        "- Never make up facts, figures or dates.\n"
        "- Everything you know is already in front of you. Never say you'll look something up, "
        "check, or get back to them; answer in this reply. If a question needs counting or combining "
        "many facts, give your best answer and say how sure you are.\n"
        "- Questions about specific companies, people, products and dates are exactly what you are "
        "here for. Answer them directly and don't remark on the questions themselves.\n"
        "- What they say reaches you through speech recognition, which often gets names wrong: a name "
        "can arrive misspelled or broken into ordinary words. If something sounds like a name in the "
        "knowledge base, assume they meant that name and answer about it, using the right name, without "
        "mentioning the mix-up. Only if it could be more than one name, ask which they meant.\n"
        "- Your answers are spoken aloud: never read out tables, lists, file names or reference "
        "numbers; round numbers for speech; keep to one to three sentences unless asked for more.\n\n"
        f"[KNOWLEDGE BASE: {name}]\n{text}\n[/KNOWLEDGE BASE]\n\n"
        f"{guide}"
        "Reminder: answer from the knowledge base above first; it overrides your own knowledge. "
        "If it doesn't cover the question, say so before answering from general knowledge."
    )


# -- Names: pronunciation respellings and the speech-recognition hint --------

class Pronunciations:
    """How she says names the voice gets wrong: 'Name = respelling' lines in
    pronunciations.txt ('#' starts a note). Applied only to text on its way
    to TTS, so what she writes, shows and remembers keeps the real spelling."""

    def __init__(self, table: Optional[dict] = None):
        self.table = dict(table or {})    # name -> respelling, in file order
        self._by_key = {_key(name): say for name, say in self.table.items()}
        self._re = None
        if self.table:
            longest_first = sorted(self.table, key=len, reverse=True)
            alternatives = "|".join(r"\s+".join(map(re.escape, name.split())) for name in longest_first)
            self._re = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)

    @classmethod
    def load(cls, *paths) -> "Pronunciations":
        """Read the files in order; a later file's entry for a name replaces
        an earlier one (the drive's over the kiosk's). Missing files are fine."""
        table = {}
        for path in paths:
            try:
                with open(path, encoding="utf-8-sig") as fh:
                    lines = fh.read().splitlines()
            except OSError:
                continue
            for line in lines:
                name, sep, say = line.partition("=")
                name, say = " ".join(name.split()), say.split("#", 1)[0].strip()
                if not sep or not name or not say or name.startswith("#"):
                    continue
                for old in [n for n in table if _key(n) == _key(name)]:
                    del table[old]
                table[name] = say
        return cls(table)

    @property
    def names(self) -> list:
        return list(self.table)

    def apply(self, text: str) -> str:
        if not self._re:
            return text
        return self._re.sub(lambda m: self._by_key[_key(m.group())], text)


def _key(name: str) -> str:
    return " ".join(name.lower().split())


# Words and single punctuation marks; a possessive stays on its word.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*|\S")


def _tokens(body: str):
    """(token, start, end) for each word and mark. A hyphenated word stays
    whole only when every part is Capitalized, the way brands are written
    (Rolls-Royce, Coca-Cola, Hewlett-Packard); otherwise its parts come out
    one by one with the hyphen between, so "AI-assisted", "spun-off" or
    "Oslo-based" never pass for a name."""
    for m in _TOKEN_RE.finditer(body):
        tok, pos = m.group(), m.start()
        if "-" in tok and not all(p[:1].isupper() for p in tok.split("-")):
            for k, part in enumerate(tok.split("-")):
                if k:
                    yield "-", pos, pos + 1
                    pos += 1
                yield part, pos, pos + len(part)
                pos += len(part)
        else:
            yield tok, pos, m.end()
# Small words that can sit inside a name ("Bank of America", "van Leeuwenhoek").
_CONNECTORS = {"of", "&", "de", "da", "di", "del", "du", "la", "le", "van", "von", "der", "den"}
# Capitalized, but never the name itself: dates, company suffixes, generations.
_NOT_NAMES = set("""january february march april may june july august september october
    november december jan feb mar apr jun jul aug sep sept oct nov dec monday tuesday
    wednesday thursday friday saturday sunday inc corp corporation co ltd llc plc ag gmbh
    sa nv jr sr ii iii iv""".split())


def is_date_word(word: str) -> bool:
    """Months, weekdays and ranges of them ("Jan-May", "Sep-Dec")."""
    parts = [p for p in re.split(r"[-–'’]", word.lower()) if p]
    return bool(parts) and all(p in _NOT_NAMES for p in parts)


def _name_word(tok: str) -> bool:
    """Capitalized and could be part of a name. Initials, short acronyms
    (spoken letter by letter, which recognition handles) and dates can't."""
    return (tok[0].isupper() and len(tok) > 1 and not is_date_word(tok)
            and not (tok.isupper() and len(tok) <= 3))


# Where names aren't looked for: web addresses and quoted titles - headlines
# are Title Case ("Example Corporation Appoints Chief Operating Officer"), so
# their verbs would pass for names. Their words still count as evidence of
# what's an ordinary word.
_NOT_PROSE = re.compile(r"https?://\S+|www\.\S+|\"[^\"\n]{1,300}\"|“[^”\n]{1,300}”")


def _runs(body: str, lower: collections.Counter = None, cap: collections.Counter = None):
    """Yield runs of name words side by side on one line, counting every
    word's lowercase and Capitalized uses along the way (when given the
    counters)."""
    run, prev_end = [], 0
    for tok, start, end in _tokens(body):
        if "\n" in body[prev_end:start] and run:
            yield run
            run = []
        prev_end = end
        if tok[-2:] in ("'s", "’s"):
            tok = tok[:-2]
        if not tok or not tok[0].isalpha():
            if run:
                yield run
            run = []
            continue
        if cap is not None:
            for word in [tok] + (tok.split("-") if "-" in tok else []):   # a brand's parts count too
                if word:
                    (cap if word[0].isupper() else lower)[word] += 1
        if _name_word(tok):
            run.append(tok)
        elif tok.lower() in _CONNECTORS and run:
            run.append(tok)
        elif run:
            yield run
            run = []
    if run:
        yield run


def kb_names(text: str, count_tokens, limit: int = 300) -> list:
    """The KB's proper names that speech recognition is likely to mishear,
    most widespread first. Works on any KB, no KB-specific files: a name is
    a run of Capitalized words, trimmed of words the KB also uses in
    lowercase (so "The", "Under" or a Title Case heading's "Revenue" drop
    away), ranked by how many files mention it. A name is kept only if it
    adds a word rare enough that the recognizer's vocabulary splits it into
    pieces (count_tokens(word) > 1); common names it already spells right.
    text is read_kb's output; count_tokens counts Whisper tokens for a word."""
    lower, cap = collections.Counter(), collections.Counter()
    bodies = [part.partition("\n")[2] for part in re.split(r"(?m)^### FILE: ", text) if part.strip()]
    for body in bodies:                  # every word counts as evidence of what's ordinary...
        for _ in _runs(body, lower, cap):
            pass
    runs_by_file = [list(_runs(_NOT_PROSE.sub(" ", body))) for body in bodies]   # ...names come from prose

    def ordinary(word):
        if "-" in word:      # "AI-Assisted" in a heading: ordinary if all its real parts are
            parts = [p for p in word.split("-") if len(p) > 1 and not (p.isupper() and len(p) <= 3)]
            return bool(parts) and all(ordinary(p) for p in parts)
        return word.lower() in _CONNECTORS or lower[word.lower()] > 0.2 * cap[word]

    files_with, mentions = collections.Counter(), collections.Counter()
    for runs in runs_by_file:
        seen = set()
        for run in runs:
            while run and ordinary(run[0]):
                run = run[1:]
            while run and ordinary(run[-1]):
                run = run[:-1]
            if not run or len(run) > 4 or (len(run) == 1 and len(run[0]) < 3):
                continue
            name = " ".join(run)
            mentions[name] += 1
            seen.add(name)
        files_with.update(seen)

    picked, covered = [], set()
    for name in sorted(files_with, key=lambda n: (-files_with[n], -mentions[n], n)):
        rare = {w.lower() for w in name.split() if w.lower() not in _CONNECTORS and count_tokens(w) > 1}
        if rare - covered:
            picked.append(name)
            covered |= rare
            if len(picked) >= limit:
                break
    return picked


# A word where a pronunciations.txt entry would match it (Pronunciations.apply): letters, maybe joined by hyphens or
# apostrophes, with no letter, digit or underscore touching either end - so "Win32" holds no "Win".
_MENTION_RE = re.compile(r"(?<!\w)[^\W\d_]+(?:[-'’][^\W\d_]+)*(?!\w)")


def word_mentions(text: str) -> collections.Counter:
    """How many times the KB uses each word (lowercase keys), in what she could
    say from it - file headers and web addresses left out. A hyphenated word
    counts for each run of its parts ("Rolls-Royce" is also a "Rolls"), and a
    possessive for its word ("Acme's"). text is read_kb's output."""
    text = re.sub(r"(?m)^### FILE: .*$", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    counts = collections.Counter()
    for m in _MENTION_RE.finditer(text):
        parts = re.split(r"([-'’])", m.group().lower())
        words, marks = parts[0::2], parts[1::2]
        for i in range(len(words)):
            run = words[i]
            counts[run] += 1
            for j in range(i + 1, len(words)):
                run += marks[j - 1] + words[j]
                counts[run] += 1
    return counts


def asr_hint(names, count_tokens, budget: int = ASR_HINT_TOKENS) -> str:
    """Comma-separated names for Whisper's hint, in the order given, as many
    as fit the budget; duplicates dropped."""
    picked, used, seen = [], 0, set()
    for name in names:
        if _key(name) in seen:
            continue
        cost = count_tokens(name) + 1    # + its comma
        if used + cost > budget:
            continue
        picked.append(name)
        seen.add(_key(name))
        used += cost
    return ", ".join(picked)
