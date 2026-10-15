"""
Automatic pronunciation check for a knowledge base's names - any KB.

Her voice (Chatterbox TTS) guesses how to say names it doesn't know, and
guesses differently from take to take (an unusual name can come out wrong in
one take of three). For each rare word in the KB's names (knowledge.kb_names)
this:
  1. gets the pronunciation (in IPA, the International Phonetic Alphabet)
     from two independent sources: CMUdict, a curated pronouncing
     dictionary, and the LLM, asked with the KB in front of it but without
     seeing the dictionary's answer. The LLM also writes respellings;
  2. has her voice say the word, as spelled, in five sentences, and a
     phoneme recognizer (pronunciation_worker.py) write down the SOUNDS;
  3. judges each take by what a listener would notice (a wrong stressed
     vowel, a wrong or missing consonant, a dropped or extra syllable) and,
     if most takes are wrong, tries the respellings the same way.
A fix is applied only when both sources agree on the pronunciation. Each
alone is wrong for some names (measured on a test KB: the LLM for three
names, the dictionary for one), so anything else goes to a review list for
a person, who settles it on the /admin Names page: accept a proposal, type
a respelling, or say the name (their recording becomes the target). Speech
recognition can't be the judge: it maps anything close onto the name it
expects.

Each word is checked once, ever (a cache). Fixes and the review list go to
pronunciations_auto.txt, which knowledge.Pronunciations reads before the
hand-written pronunciations.txt files - so a person's entry always wins. The
orchestrator runs the check in the background while nobody is at the kiosk.
"""

import asyncio
import configparser
import datetime
import functools
import json
import os
import re
import sys
import time
import unicodedata

from .gaze_calculator import CONFIG_PATH
from .knowledge import _CONNECTORS, _key, is_date_word

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(SERVER_DIR, "data", "pronunciation_cache.json")
AUTO_PRONUNCIATIONS = os.path.join(SERVER_DIR, "data", "pronunciations_auto.txt")
WORKER_LOG = os.path.join(SERVER_DIR, "logs", "pronunciation_worker.log")
# CMUdict (github.com/cmusphinx/cmudict, BSD licence; LICENSE beside it).
# Without it every target is the LLM's, so nothing is fixed, only listed.
DICTIONARY_PATH = os.path.join(SERVER_DIR, "models", "cmudict", "cmudict.dict")

CHECK_VERSION = 5          # bump when scoring changes, to check every word again
# Five sentences per spelling: her voice varies take to take, and names come
# mid-sentence, first in a sentence, and in lists (where she can spell an
# all-capitals name out letter by letter). A word is judged take by take -
# one odd take (hers, or a recognizer slip) mustn't decide it.
TAKES = ("I can tell you about {} today.", "{} came up again this morning.", "Let's talk about {} next.",
         "The list includes Morgan, {}, and Wells.", "That includes {} and more.")
TAKE_OK = 0.2              # a take is right: no mistake a listener would notice (_serious),
                           # and not much else off per sound
MAJORITY = 4               # a spelling is right when this many of its takes are (of 5)
LLM_BATCH = 12             # words per LLM request (~5 s of generation)
QUIET_S = 30.0             # nobody at the kiosk, nothing in flight, this long before each step


def check_enabled(path: str = CONFIG_PATH) -> bool:
    """kiosk_config.ini [kb] pronunciation_check, on unless set false."""
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8-sig")
    return cfg.getboolean("kb", "pronunciation_check", fallback=True)


# -- Sounds -----------------------------------------------------------------
# Folded onto one inventory: the recognizer writes espeak-style IPA, the LLM
# dictionary-style IPA, and American English doesn't care about length marks.

_PAIRS = {"tʃ", "dʒ", "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "əʊ", "ei", "ai", "au", "ou", "oi"}
_FOLD = {"ɝ": "ɚ", "ɜ": "ɚ", "ᵻ": "ɪ", "ɐ": "ə", "ɒ": "ɑ", "a": "ɑ", "r": "ɹ", "ɾ": "t", "ɡ": "g",
         "ɫ": "l", "ʍ": "w", "x": "k", "ʔ": "", "e": "eɪ", "o": "oʊ", "əʊ": "oʊ", "ei": "eɪ",
         "ai": "aɪ", "au": "aʊ", "ou": "oʊ", "oi": "ɔɪ"}
_VOWELS = {"i", "ɪ", "eɪ", "ɛ", "æ", "ɑ", "ɔ", "oʊ", "ʊ", "u", "ə", "ɚ", "ʌ", "aɪ", "aʊ", "ɔɪ",
           "y", "ø", "œ", "ɵ", "ɯ", "ɨ", "ʏ"}
_SKIP = set("ːˑ./[]()-‿ʰʲʷˠˤ' \t")
# Close enough that the recognizer confuses them or a listener barely notices.
_NEAR_VOWELS = {frozenset(p) for p in [("ɑ", "ɔ"), ("ɑ", "ʌ"), ("ɑ", "ə"), ("ʌ", "ə"), ("ɪ", "i"),
                                       ("ɪ", "ə"), ("ɛ", "æ"), ("ɛ", "eɪ"), ("ʊ", "u"), ("oʊ", "ɔ"),
                                       ("ɚ", "ə")]}
_REDUCED = {"ə", "ɪ", "ɚ"}    # what an unstressed vowel naturally relaxes to
_VOICING = {frozenset(p) for p in [("p", "b"), ("t", "d"), ("k", "g"), ("f", "v"), ("θ", "ð"),
                                   ("s", "z"), ("ʃ", "ʒ"), ("tʃ", "dʒ")]}
_NEAR_CONSONANTS = {frozenset(p) for p in [("θ", "f"), ("ð", "v"), ("ð", "d"), ("θ", "t"),
                                           ("s", "ʃ"), ("n", "ŋ"), ("m", "n")]}
_GLIDES = {frozenset(p) for p in [("ɚ", "ɹ"), ("i", "j"), ("u", "w")]}


def _merge_r(seq: list) -> list:
    """ə + ɹ not before a vowel is one sound, ɚ (as in 'her')."""
    out = []
    for k, (sym, stressed) in enumerate(seq):
        nxt = seq[k + 1][0] if k + 1 < len(seq) else None
        if sym == "ɹ" and out and out[-1][0] in ("ə", "ɚ") and nxt not in _VOWELS:
            out[-1] = ("ɚ", out[-1][1])
            continue
        out.append((sym, stressed))
    return out


PRIMARY, SECONDARY = 2, 1      # stress levels; 0 = unstressed


def parse_ipa(ipa: str) -> list:
    """[(sound, stress)] from an IPA string such as /ˌkæləməˈzuː/: the
    first vowel after ˈ gets PRIMARY stress, after ˌ SECONDARY, others 0."""
    s = unicodedata.normalize("NFD", ipa).replace("\u0361", "").replace("\u035c", "")
    out, pending, i = [], 0, 0
    while i < len(s):
        c = s[i]
        if c in "ˈˌ":
            pending, i = (PRIMARY if c == "ˈ" else SECONDARY), i + 1
            continue
        if c in _SKIP or unicodedata.combining(c):
            i += 1
            continue
        sym = s[i:i + 2] if s[i:i + 2] in _PAIRS else c
        i += len(sym)
        sym = _FOLD.get(sym, sym)
        if not sym or not sym[0].isalpha():
            continue
        stress = 0
        if sym in _VOWELS:
            stress, pending = pending, 0
        out.append((sym, stress))
    return _merge_r(out)


# Recognizer symbols that are two sounds here ('ɔːɹ' in "for" is ɔ + ɹ).
_SPLIT = {"aɪɚ": ("aɪ", "ɚ"), "aɪə": ("aɪ", "ə"), "aʊɚ": ("aʊ", "ɚ"), "iə": ("i", "ə"), "eə": ("ɛ", "ə"),
          "ʊə": ("ʊ", "ə"), "əl": ("ə", "l"), "oɹ": ("ɔ", "ɹ")}


def _pieces(p: str) -> list:
    """One recognizer symbol as sounds: 'ɔːɹ' -> ɔ ɹ, syllabic 'l̩' -> ə l."""
    p = unicodedata.normalize("NFD", p)
    syllabic = "\u0329" in p
    # tone numbers from the model's other languages ('ɪ5') dropped: English has none
    p = "".join(c for c in p if c not in _SKIP and not unicodedata.combining(c) and not c.isdigit())
    if not p:
        return []
    if syllabic:
        return ["ə", p]
    if p in _SPLIT:
        return list(_SPLIT[p])
    if len(p) > 1 and p.endswith("ɹ"):
        return _pieces(p[:-1]) + ["ɹ"]
    return [p]


def fold_heard(phonemes: list) -> list:
    """The recognizer's phonemes on the same inventory as parse_ipa."""
    seq = []
    for p in phonemes:
        for piece in _pieces(p):
            piece = _FOLD.get(piece, piece)
            if piece and piece[0].isalpha():
                seq.append((piece, False))
    return [sym for sym, _ in _merge_r(seq)]


def _sub(a: str, b: str, stress: int) -> float:
    """What saying b in place of target sound a costs."""
    if a == b:
        return 0.0
    pair = frozenset((a, b))
    if a in _VOWELS and b in _VOWELS:
        if b in _REDUCED and stress != PRIMARY:
            return 0.3 if stress else 0.2      # relaxing to 'uh' off the main stress
        if pair in _NEAR_VOWELS:
            return 0.5 if stress else 0.3
        return 1.0 if stress else 0.6
    if a not in _VOWELS and b not in _VOWELS:
        if pair in _VOICING:
            return 0.25
        return 0.5 if pair in _NEAR_CONSONANTS else 1.0
    return 0.5 if pair in _GLIDES else 1.0


_DIPHTHONGS = {"aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ"}
_SCHWAS = {"ə", "ɚ"}
# An extra sound is usually the recognizer's slip or a quick 'uh'; an extra
# full vowel is an extra syllable ("Mis-chee-vee-us" for "Mis-chuh-vus").
_EXTRA_OK = {"ə", "ɚ", "ɪ", "ʌ"}


def _serious_extra(h: str) -> bool:
    return h in _VOWELS and h not in _EXTRA_OK


def _serious(a: str, stress: int, b: str = None) -> bool:
    """Would a listener call it a mispronunciation, rather than an accent or
    a relaxed syllable? b is what was said instead; None if a was dropped."""
    if b is None:                   # a dropped consonant or syllable; a quick 'uh' may go
        return a not in _SCHWAS
    if a in _VOWELS and b in _VOWELS:
        if frozenset((a, b)) in _NEAR_VOWELS:
            return False
        if stress == PRIMARY:
            return True             # "Ka-luh-muh-ZOH" for "Ka-luh-muh-ZOO"
        if stress == SECONDARY:
            return b not in _REDUCED    # "KAY-luh-" for "KA-luh-"; relaxing to 'uh' is fine
        return b in _DIPHTHONGS and a not in _DIPHTHONGS    # "-ite" for "-id", "-hay" for "-her"
    return _sub(a, b, stress) >= 1.0    # a different consonant, or a vowel in its place


def align(target: list, heard: list) -> tuple:
    """(score, start, end, serious): how far what was heard is from the
    target pronunciation, per target sound (0 = exact); where in the heard
    sounds it matched; and how many differences a listener would call
    mistakes (_serious). The target is matched wherever it fits best, so the
    sentence around the word costs nothing."""
    if not target:
        return 1.0, 0, 0, 1
    # Each cell: (cost, where this match started in heard, serious so far).
    prev = [(0.0, j, 0) for j in range(len(heard) + 1)]      # start anywhere
    for sym, stress in target:
        # A dropped unstressed vowel costs a bit more than a wrong one, so the
        # match can't skip a wrong ending ("-hay" for "-her") by ending early.
        missing = 0.7 if sym in _VOWELS and not stress else 1.0
        dropped = int(_serious(sym, stress))
        cur = [(prev[0][0] + missing, prev[0][1], prev[0][2] + dropped)]
        for j, h in enumerate(heard, 1):
            sub = _sub(sym, h, stress)
            cur.append(min((prev[j][0] + missing, prev[j][1], prev[j][2] + dropped),     # target sound not said
                           (cur[j - 1][0] + 0.5, cur[j - 1][1], cur[j - 1][2] + _serious_extra(h)),   # an extra sound said
                           (prev[j - 1][0] + sub, prev[j - 1][1],
                            prev[j - 1][2] + (sub > 0 and _serious(sym, stress, h)))))
        prev = cur
    end = min(range(len(prev)), key=lambda j: prev[j])        # ...and end anywhere
    cost, start, serious = prev[end]
    return cost / len(target), start, end, serious


def take_right(score_: float, serious: int) -> bool:
    return serious == 0 and score_ <= TAKE_OK


def score(target: list, heard: list) -> float:
    return align(target, heard)[0]


_SPELL = {"i": "ee", "ɪ": "i", "eɪ": "ay", "ɛ": "e", "æ": "a", "ɑ": "ah", "ɔ": "aw", "oʊ": "oh", "ʊ": "uu",
          "u": "oo", "ə": "uh", "ʌ": "uh", "ɚ": "er", "aɪ": "eye", "aʊ": "ow", "ɔɪ": "oy", "tʃ": "ch", "dʒ": "j",
          "ʃ": "sh", "ʒ": "zh", "θ": "th", "ð": "th", "ŋ": "ng", "j": "y", "ɹ": "r"}


def respell_ipa(sounds: list, spell: dict = None) -> str:
    """A readable sounds-like spelling, syllables split before the consonant
    that starts each vowel: k æ l ə m ə z u -> Ka-luh-muh-zoo. Shows
    what she says, and is one more respelling to try."""
    syms = [s[0] if isinstance(s, tuple) else s for s in sounds]
    syllables, cur, has_vowel = [], "", False
    for k, s in enumerate(syms):
        vowel = s in _VOWELS
        starts_syllable = (vowel or (k + 1 < len(syms) and syms[k + 1] in _VOWELS)) and has_vowel
        if starts_syllable:
            syllables.append(cur)
            cur, has_vowel = "", False
        cur += (spell or _SPELL).get(s, s)
        has_vowel = has_vowel or vowel
    if cur:
        syllables.append(cur)
    return "-".join(syllables).capitalize()


# -- The dictionary ----------------------------------------------------------

_ARPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɚ", "EY": "eɪ",
         "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u", "B": "b", "CH": "tʃ", "D": "d",
         "DH": "ð", "F": "f", "G": "g", "HH": "h", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
         "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z",
         "ZH": "ʒ"}


def arpabet_to_ipa(phones: list) -> str:
    """CMUdict's ARPAbet ('W UH1 S T ER0') as IPA with stress marks on
    the stressed vowels ('wˈʊstɚ'), the form parse_ipa reads."""
    out = []
    for p in phones:
        base, stress = p.rstrip("012"), p[len(p.rstrip("012")):]
        sym = "ə" if base == "AH" and stress == "0" else _ARPA[base]
        out.append({"1": "ˈ", "2": "ˌ"}.get(stress, "") + sym)
    return "".join(out)


def lookup_dictionary(words, path: str = None) -> dict:
    """{word (lowercase): [IPA, ...]} for those of the words CMUdict has -
    all its variants, the usual one first. Reads only what it needs."""
    wanted, out = {w.lower() for w in words}, {}
    try:
        with open(path or DICTIONARY_PATH, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split("#", 1)[0].split()
                if len(parts) < 2:
                    continue
                key = parts[0].split("(", 1)[0]
                if key in wanted:
                    try:
                        out.setdefault(key, []).append(arpabet_to_ipa(parts[1:]))
                    except KeyError:
                        continue
    except OSError:
        return {}
    return out


# -- Asking the LLM ----------------------------------------------------------

_RESPELLING = re.compile(r"^[A-Za-z][A-Za-z' -]{0,40}$")


def _looks_like_ipa(cell: str) -> bool:
    """One word's IPA: no spaces, capitals or digits, a stress mark or an IPA
    letter, and a vowel - so "no idea" or "N/A" in the IPA column is skipped."""
    return (bool(cell) and not any(c.isspace() or c.isdigit() or ("A" <= c <= "Z") for c in cell)
            and ("ˈ" in cell or "ˌ" in cell or any(ord(c) > 127 for c in cell))
            and any(sym in _VOWELS for sym, _ in parse_ipa(cell)))


def llm_request(words: list) -> str:
    """The request, appended to her live conversation (never stored in it) so
    llama's cached KB is reused. words: [(word, a name it appears in)]. The
    dictionary's pronunciation is deliberately not shown: the LLM's answer is
    a second, independent opinion."""
    lines = "\n".join(w if ex.lower() == w.lower() else f"{w} (as in {ex})" for w, ex in words)
    return (
        "[KIOSK MAINTENANCE - not a visitor. Don't answer as yourself and add no tags; output only the table.]\n"
        "For each word below, from names in your knowledge base: how it is normally pronounced in "
        "American English, in IPA with stress marks, then three different respellings that a "
        "text-to-speech voice would read aloud that way - plain lowercase letters, each spelled "
        "differently. Make the first one change as few letters of the word as possible (like "
        "wooster for Worcester); join the syllables of the others with hyphens.\n"
        "One line per word, exactly: word | IPA | respelling | respelling | respelling\n"
        "Example: Worcester | ˈwʊstɚ | wooster | wuss-ter | wuh-stir\n\n" + lines
    )


def agree(ipas: list, other: str) -> bool:
    """Do two sources give the same pronunciation, by the listener rules?
    ipas: one source's variants (any may match); other: the second source."""
    sounds = [s for s, _ in parse_ipa(other)]
    for ipa in ipas:
        s, _, _, serious = align(parse_ipa(ipa), sounds)
        if take_right(s, serious):
            return True
    return False


def _tidy(respelling: str) -> str:
    """'KA-luh-muh-ZOO' -> 'Ka-luh-muh-zoo': her voice may read a capital
    syllable letter by letter."""
    return respelling.lower().capitalize() if any(c.isupper() for c in respelling[1:]) else respelling.capitalize()


def parse_llm_table(text: str, words: list) -> dict:
    """{word: (ipa, [respellings])} for the words asked about. Lines that
    don't parse are skipped (those words are asked again another time)."""
    wanted = {w.lower(): w for w in words}
    out = {}
    for line in text.splitlines():
        line = re.sub(r"<<[^>]*>>|\[[a-z]+\]|\([^)|]*\)", "", line)   # her tags; "(as in ...)" echoes
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
        cells = [c.strip().strip("\"'`*").strip() for c in line.split("|")]
        if len(cells) < 2 or cells[0].lower() not in wanted:
            continue
        word = wanted[cells[0].lower()]
        ipa = cells[1].strip("/[] ")
        if not _looks_like_ipa(ipa):
            continue
        says = []
        for c in cells[2:5]:
            if _RESPELLING.match(c) and c.lower() != word.lower() and c.lower() not in (s.lower() for s in says):
                says.append(_tidy(c))
        out[word] = (ipa, says)
    return out


# -- Deciding ----------------------------------------------------------------

def words_to_check(names: list, count_tokens, skip) -> list:
    """[(word, a name it appears in)]: each rare word of the KB's names once,
    in the names' order (most widespread first) - the words her voice has to
    guess at. Words in skip (checked before, or in a pronunciations.txt) are
    left out."""
    out, seen = [], {s.lower() for s in skip}
    for name in names:
        for word in name.split():
            key = word.lower()
            if key in _CONNECTORS or key in seen or len(word) < 3 or count_tokens(word) <= 1:
                continue
            seen.add(key)
            out.append((word, name))
    return out


async def rate_spelling(spelling: str, targets: list, say_and_hear) -> tuple:
    """Her voice says one spelling in each of TAKES; judged against the
    target(s) - any one may match. Returns ({"right": takes right, "score":
    the median take's score}, the median take's sounds - a typical rendering
    - and every take's sounds as strings), or (None, None, []) unless every
    take was heard (a take lost to a voice-server restart would skew the
    count; the word is simply tried again later)."""
    takes = []
    for take in TAKES:
        heard = await say_and_hear(take.format(spelling))
        if heard:
            sounds = fold_heard(heard)
            s, start, end, serious = min(align(t, sounds) for t in targets)
            takes.append((s, take_right(s, serious), sounds[start:end]))
    if len(takes) < len(TAKES):
        return None, None, []
    renderings = [" ".join(w) for _, _, w in takes]
    takes.sort(key=lambda t: t[0])
    middle = takes[len(takes) // 2]
    return {"right": sum(ok for _, ok, _ in takes), "score": round(middle[0], 3)}, middle[2], renderings


async def check_word(word: str, ipa: str, respellings: list, say_and_hear,
                     trusted: bool = True, alternatives=()) -> dict:
    """Score the word as spelled against the target; if it sounds wrong, try
    the respellings (plus one made from the target). trusted: the target is
    a dictionary's, so a clear fix becomes status 'fixed' (applied);
    otherwise it's the LLM's guess and a disagreement is 'review' (listed,
    with the best respelling as a proposal). alternatives: other acceptable
    pronunciations (dictionary variants). say_and_hear(text) returns the
    phonemes heard, or None."""
    targets = [parse_ipa(ipa)] + [parse_ipa(a) for a in alternatives]

    async def rate(spelling):
        return await rate_spelling(spelling, targets, say_and_hear)

    result = {"word": word, "ipa": ipa, "trusted": trusted, "takes": len(TAKES), "say": None, "tried": {}}
    plain, heard, renderings = await rate(word)
    result.update(plain=plain, heard=" ".join(heard or []), heard_all=renderings)
    if plain is None:
        result["status"] = "error"
        return result
    if plain["right"] >= MAJORITY:
        result["status"] = "ok"
        return result
    best = None
    for spelling in dict.fromkeys(respellings + [respell_ipa(targets[0])]):
        r = result["tried"][spelling] = (await rate(spelling))[0]
        if r and r["right"] >= MAJORITY and r["right"] > plain["right"] and (
                best is None or (r["right"], -r["score"]) > (result["tried"][best]["right"], -result["tried"][best]["score"])):
            best = spelling
            if r["right"] == len(TAKES) and r["score"] == 0:
                break    # can't do better
    if not trusted:
        result.update(status="review", say=best)
    elif best:
        result.update(status="fixed", say=best)
    else:
        result["status"] = "unresolved"
    return result


def _takes(r: dict, n: int) -> str:
    return f"{r['right']}/{n} takes right" if r else "not heard"


def _says(r: dict) -> str:
    """How she says it: each different way she said it across the takes."""
    ways = list(dict.fromkeys(respell_ipa(h.split()) for h in r.get("heard_all") or [] if h))
    if not ways and r.get("heard"):
        ways = [respell_ipa(r["heard"].split())]
    if not ways:
        return "she wasn't heard"
    return "she says ~" + " / ~".join(ways[:3]) + (" (varies)" if len(ways) > 1 else "")


def _source(r: dict) -> str:
    if r.get("other_ipa"):
        return f"the dictionary says /{r['ipa']}/ but the LLM says /{r['other_ipa']}/"
    source = r.get("source", "the dictionary" if r.get("trusted") else "the LLM")
    return f"{source} {'say' if ' and ' in source else 'says'} /{r['ipa']}/"


def describe(result: dict) -> str:
    """One word's outcome, for the log."""
    s, n = result["status"], result.get("takes", len(TAKES))
    plain = _takes(result.get("plain"), n)
    if s == "ok":
        return f"right as spelled ({plain}; {_source(result)})"
    if s == "fixed":
        return f"fixed as '{result['say']}' ({plain} -> {_takes(result['tried'][result['say']], n)}); {_says(result)}"
    if s == "review":
        proposal = (f"'{result['say']}' would fix it ({_takes(result['tried'][result['say']], n)})"
                    if result["say"] else "no respelling worked")
        return f"for review: {_says(result)} ({plain}); {_source(result)}; {proposal}"
    return f"still wrong ({plain}); {_says(result)}; {_source(result)}; no respelling worked"


def summarize(words: dict) -> str:
    n = {s: sum(1 for r in words.values() if r.get("status") == s) for s in ("ok", "fixed", "review", "unresolved")}
    return (f"{n['ok']} right as spelled, {n['fixed']} fixed, {n['review']} for review, "
            f"{n['unresolved']} still wrong (see {AUTO_PRONUNCIATIONS})")


# -- Remembering -------------------------------------------------------------

def load_cache(path: str = None) -> dict:
    """{word (lowercase): result}; empty if missing or from an older check."""
    try:
        with open(path or CACHE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data.get("words", {}) if data.get("version") == CHECK_VERSION else {}


def save_cache(words: dict, path: str = None):
    _write_atomic(path or CACHE_PATH, json.dumps({"version": CHECK_VERSION, "words": words}, ensure_ascii=False, indent=1))


def write_auto_file(words: dict, path: str = None):
    """The agreed fixes as 'Word = respelling' lines, then - as notes - the
    words for review (most important first), the ones still wrong, and the
    rest."""
    rows = sorted(words.values(), key=lambda r: r["word"].lower())
    by = {s: [r for r in rows if r.get("status") == s] for s in ("fixed", "review", "unresolved")}
    by["review"].sort(key=lambda r: r.get("rank", 10 ** 9))
    ok = [r["word"] for r in rows if r.get("status") == "ok"]
    n = len(TAKES)

    def right(r):
        return f"{r['right']}/{r.get('takes', n)}" if r else "-"

    lines = [
        "# GENERATED by the kiosk's pronunciation check (orchestrator/pronunciation.py) - don't edit.",
        "# To correct a word, add a line to pronunciations.txt (next to kiosk_config.ini, or beside",
        "# the KB on its drive); those always win over this file.",
        f"# Each spelling was said in {n} sentences; a take is right when nothing in it is off in a way",
        "# a listener would notice (a wrong stressed vowel, a wrong or missing consonant, a dropped or",
        "# extra syllable). Delete pronunciation_cache.json and this file to check every word again,",
        "# e.g. after changing her voice.",
        "",
        f"# Fixed - the pronouncing dictionary and the LLM agree on how it's said ({len(by['fixed'])}):",
    ]
    for r in by["fixed"]:
        lines.append(f"{r['word']} = {r['say']}    # /{r['ipa']}/  {_says(r)}; right as spelled "
                     f"{right(r['plain'])} -> {right(r['tried'][r['say']])}")
    lines += ["", f"# For review, most important first ({len(by['review'])}) - she doesn't say these the way the",
              "# LLM or the dictionary thinks they're said, and the two don't both vouch for it. To accept a",
              "# proposal, copy its line without the '# ' into pronunciations.txt:"]
    for r in by["review"]:
        note = f"# {_source(r)}; {_says(r)}; right as spelled {right(r['plain'])}"
        if r["say"]:
            lines.append(f"# {r['word']} = {r['say']}    {note} -> {right(r['tried'][r['say']])}")
        else:
            lines.append(f"#   {r['word']}: no respelling worked    {note}")
    split = sorted((r for r in rows if r.get("status") == "ok" and r.get("other_ipa")),
                   key=lambda r: r.get("rank", 10 ** 9))
    lines += ["", f"# Worth a listen - she follows the dictionary, but the LLM says otherwise ({len(split)}):"]
    for r in split:
        lines.append(f"#   {r['word']}: {_source(r)}; {_says(r)}")
    lines += ["", f"# Still wrong - the dictionary and the LLM agree, but no respelling helped ({len(by['unresolved'])}):"]
    for r in by["unresolved"]:
        tried = ", ".join(f"{k} {right(v)}" for k, v in r["tried"].items() if v)
        lines.append(f"#   {r['word']}  /{r['ipa']}/  {_says(r)}; right as spelled {right(r['plain'])}"
                     + (f"; tried {tried}" if tried else ""))
    lines += ["", f"# Said right as spelled ({len(ok)}):"]
    line = "#  "
    for w in ok:
        if len(line) + len(w) > 98:
            lines.append(line.rstrip(","))
            line = "#  "
        line += f" {w},"
    if ok:
        lines.append(line.rstrip(","))
    _write_atomic(path or AUTO_PRONUNCIATIONS, "\n".join(lines) + "\n")


def _write_atomic(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def today() -> str:
    return datetime.date.today().isoformat()


# -- The phoneme recognizer subprocess --------------------------------------

class PhonemeWorker:
    """pronunciation_worker.py, started on first use and closed when the
    check is done, so its model never lives in the orchestrator."""

    def __init__(self, device: str, log_path: str = WORKER_LOG, command: list = None):
        self.log_path = log_path
        self.command = command or [sys.executable, "-u", "-m", "orchestrator.pronunciation_worker",
                                   "--device", device]
        self._proc = None
        self._log = None
        self._next_id = 0

    async def _reply(self, timeout: float) -> dict:
        """The next JSON line from the worker; anything else it prints is skipped."""
        while True:
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
            if not line:
                raise RuntimeError(f"phoneme recognizer exited (see {self.log_path})")
            try:
                return json.loads(line)
            except ValueError:
                continue

    async def start(self):
        self._log = open(self.log_path, "w", encoding="utf-8")
        self._proc = await asyncio.create_subprocess_exec(
            *self.command, cwd=SERVER_DIR, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self._log,
            limit=32 * 1024 * 1024)        # a features reply is one ~0.5 MB line per take (pron_match)
        try:
            ready = await self._reply(timeout=300.0)
            if not ready.get("ready"):
                raise RuntimeError(f"phoneme recognizer didn't start: {ready}")
        except BaseException:
            await self.close()      # so the next use starts afresh
            raise
        return ready.get("device")

    async def hear(self, audio_base64: str, sample_rate: int, features: bool = False):
        """The phonemes heard in a clip. With features: a dict that also has each
        phoneme's frames ("spans"), the model's per-frame features ("hidden",
        frames x dim), loudness ("energy") and pitch ("pitch") - see pron_match.py."""
        if self._proc is None:
            await self.start()
        self._next_id += 1
        req = {"id": self._next_id, "audio": audio_base64, "rate": sample_rate}
        if features:
            req["features"] = True
        self._proc.stdin.write((json.dumps(req) + "\n").encode())
        await self._proc.stdin.drain()
        reply = await self._reply(timeout=60.0)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        if not features:
            return reply["phonemes"]
        import base64
        import numpy as np
        hidden = np.frombuffer(base64.b64decode(reply["hidden"]), dtype=np.float16)
        return {"phonemes": reply["phonemes"], "spans": reply["spans"],
                "hidden": hidden.reshape(reply["frames"], reply["dim"]).astype(np.float32),
                "energy": np.frombuffer(base64.b64decode(reply["energy"]), dtype=np.float16).astype(np.float32),
                "pitch": (np.frombuffer(base64.b64decode(reply["pitch"]), dtype=np.float16).astype(np.float32)
                          if reply.get("pitch") else np.full(reply["frames"], np.nan, dtype=np.float32))}

    async def close(self):
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()    # it exits at end of input
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except Exception:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass                    # already gone
        finally:
            self._proc = None
            self._log.close()


# -- Settling names by hand (the /admin Names page) --------------------------
# A person is the one reliable authority for a brand name: they accept a
# proposal, type a respelling, or say the name - their recording becomes the
# target her voice is fitted to. Decisions and recordings live in
# REVIEWED_PATH; accepted respellings go into a pronunciations.txt.

REVIEWED_PATH = os.path.join(SERVER_DIR, "data", "pronunciations_reviewed.json")
# The words the name finder currently takes from the KB (lowercase). Results
# for words it no longer finds (a finder fix, a new drive) stay cached but
# aren't shown for review.
WORDS_PATH = os.path.join(SERVER_DIR, "data", "pronunciation_words.json")


def save_current_words(words, path: str = None):
    _write_atomic(path or WORDS_PATH, json.dumps(sorted({w.lower() for w in words}), ensure_ascii=False))


def load_current_words(path: str = None):
    """The current words as a set, or None if the check hasn't recorded any."""
    try:
        with open(path or WORDS_PATH, encoding="utf-8") as fh:
            return set(json.load(fh))
    except (OSError, ValueError):
        return None
LISTENING_PAUSED = os.path.join(SERVER_DIR, "data", "listening_paused")
# Lines for her to say at the kiosk, one per line, through her real voice path
# (respellings applied, lip sync) - so a person hears what visitors will hear.
SAY_REQUEST = os.path.join(SERVER_DIR, "data", "say_request.txt")


def request_say(lines: list, path: str = None):
    """Queue lines for her to say at the kiosk (the orchestrator picks them
    up within seconds, when she isn't in the middle of a conversation)."""
    _write_atomic(path or SAY_REQUEST, "\n".join(l.strip() for l in lines if l.strip()) + "\n")


def take_say_request(path: str = None) -> list:
    """The queued lines, removing the request; [] if none."""
    target = path or SAY_REQUEST
    try:
        with open(target, encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        os.remove(target)
    except OSError:
        return []
    return lines
_SPELL_ALT = {"eɪ": "ai", "i": "ea", "oʊ": "o", "ɑ": "o", "ə": "a", "ɚ": "ur", "aɪ": "y", "u": "ew",
              "ɛ": "eh", "ʊ": "oo", "aʊ": "ou", "ɔ": "aw"}


def target_from_recording(sounds: list) -> list:
    """A person's recording of a name, as a target: their full vowels count
    as stressed (a different full vowel in her rendering is a mistake), the
    relaxed ones ('uh', 'ih', 'er') stay flexible, as in anyone's speech."""
    return [(s, SECONDARY if s in _VOWELS and s not in _REDUCED else 0) for s in sounds]


def respellings_for(sounds: list) -> list:
    """Respellings to try for a pronunciation, made from its sounds alone (no
    LLM): the readable one (Ka-luh-muh-zoo), the same run together, and
    one using the other common spelling of each vowel."""
    readable = respell_ipa(sounds)
    other = respell_ipa(sounds, {**_SPELL, **_SPELL_ALT})
    return [s for s in dict.fromkeys([readable, readable.replace("-", "").capitalize(), other]) if s]


@functools.lru_cache(maxsize=16)
def sentence_sounds(text: str) -> tuple:
    """The sounds of the ordinary words around a name ('I can tell you about'),
    by the dictionary's usual pronunciation, all unstressed - said in passing,
    their vowels relax. None if the dictionary lacks one of the words."""
    words = re.findall(r"[a-z']+", text.lower())
    found = lookup_dictionary(words)
    if any(w not in found for w in words):
        return None
    return tuple((s, 0) for w in words for s, _ in parse_ipa(found[w][0]))


def _gap_owner(left, right, sound: str):
    """Who an extra heard sound between two sentence parts belongs to (0 =
    the words before the name, 1 = the name, 2 = the words after)."""
    if left == right:
        return left                             # inside a word, or inside the name
    if 1 in (left, right) or (left, right) == (0, 2):
        return None if sound == "ə" else 1      # at the name's edge; a stray 'uh' is nobody's
    return left if left is not None else right


def name_in_sentence(before, target, after, heard: list, span: bool = False):
    """The heard sounds of a name said inside a known sentence. The sentence
    is matched as a whole - the words around the name by their sounds
    (sentence_sounds), the name by its target - so the name is everything
    between its neighbours, including sounds the target lacks: an added
    syllable or a spelled-out letter, which matching the target alone would
    cut off. A sound at the boundary that could be either's ('about Tucson'
    heard with one t) goes to the name. With span=True: the name's (first,
    past-the-last) index in heard instead, or None."""
    seq = ([(s, 0, 0) for s, _ in before] + [(s, st, 1) for s, st in target]
           + [(s, 0, 2) for s, _ in after])
    edges = {len(before) - 1, len(before) + len(target)}       # the neighbours' boundary sounds
    n, m = len(seq), len(heard)
    if not m:
        return None if span else []

    def gap(i, sound):
        """Owner of an extra sound heard after seq[:i], and what it costs:
        cheap in the name (it can be any sounds), dear anywhere else."""
        owner = _gap_owner(seq[i - 1][2] if i else None, seq[i][2] if i < n else None, sound)
        return owner, 0.2 if owner == 1 else 0.5

    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        if i:
            sym, stress, part = seq[i - 1]
            drop = 0.7 if sym in _VOWELS and not stress else 1.0
            if part != 1 and i - 1 in edges:
                drop = 0.5          # the name claims a boundary sound it matches (nearly)
        for j in range(m + 1):
            options = []
            if i and j:
                options.append((cost[i - 1][j - 1] + _sub(sym, heard[j - 1], stress), "sub"))
            if i:
                options.append((cost[i - 1][j] + drop, "drop"))
            if j:
                options.append((cost[i][j - 1] + gap(i, heard[j - 1])[1], "extra"))
            if options:
                cost[i][j], back[i][j] = min(options, key=lambda o: o[0])
    i, j, owner = n, m, [None] * m
    while i or j:
        step = back[i][j]
        if step == "sub":
            owner[j - 1] = seq[i - 1][2]
            i, j = i - 1, j - 1
        elif step == "drop":
            i -= 1
        else:
            owner[j - 1] = gap(i, heard[j - 1])[0]
            j -= 1
    mine = [k for k, o in enumerate(owner) if o == 1]
    if span:
        return (mine[0], mine[-1] + 1) if mine else None
    return heard[mine[0]:mine[-1] + 1] if mine else []


def set_pronunciation(path: str, name: str, respelling: str):
    """Put 'Name = respelling' into a pronunciations.txt: an existing entry
    for the name (any case) is replaced where it stands, otherwise the line
    is added at the end; notes and other entries are untouched. A missing
    file is created with a short header."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = ["# How the avatar should SAY names her voice gets wrong: Name = respelling.",
                 "# Only what her voice reads changes; what she writes and remembers keeps the real name.", ""]
    entry = f"{' '.join(name.split())} = {respelling.strip()}"
    for k, line in enumerate(lines):
        left, sep, _ = line.partition("=")
        if sep and not left.strip().startswith("#") and _key(left) == _key(name):
            lines[k] = entry
            break
    else:
        lines.append(entry)
    _write_atomic(path, "\n".join(lines) + "\n")


def load_reviewed(path: str = None) -> dict:
    """{word (lowercase): {"word", "decision": "fine" | "set" | "recorded",
    "respelling"?, "recorded"? (the person's sounds), "when"}}"""
    try:
        with open(path or REVIEWED_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def mark_reviewed(word: str, decision: str, path: str = None, **fields) -> dict:
    """Record a person's decision on a word (fields kept from before unless
    replaced, so a recording survives a later 'set')."""
    reviewed = load_reviewed(path)
    entry = reviewed.get(word.lower(), {})
    entry.update(fields, word=word, decision=decision, when=datetime.datetime.now().isoformat(timespec="seconds"))
    reviewed[word.lower()] = entry
    _write_atomic(path or REVIEWED_PATH, json.dumps(reviewed, ensure_ascii=False, indent=1))
    return entry


def review_items(cache: dict, settled: set, reviewed: dict, current: set = None) -> list:
    """What still needs a person, most important first: words for review,
    words still wrong, and words where the dictionary and the LLM disagree
    though she follows one - minus words already settled (an entry in a
    pronunciations.txt, or a decision on the page) and, given the current
    words, any the name finder no longer takes from the KB."""
    items = []
    for key, r in cache.items():
        if key in settled or reviewed.get(key, {}).get("decision") in ("fine", "set"):
            continue
        if current is not None and key not in current:
            continue
        if is_date_word(r["word"]):     # "Jan-May": a date range, not a name, even if it is cached
            continue
        status = r.get("status")
        if status == "ok" and r.get("other_ipa"):
            status = "listen"
        if status not in ("review", "unresolved", "listen"):
            continue
        n = r.get("takes", len(TAKES))
        ways = list(dict.fromkeys(respell_ipa(h.split()) for h in r.get("heard_all") or [] if h))
        proposal = r.get("say")
        items.append({
            "word": r["word"], "rank": r.get("rank"), "status": status, "takes": n,
            "says": ways or ([respell_ipa(r["heard"].split())] if r.get("heard") else []),
            "right_as_spelled": (r.get("plain") or {}).get("right"),
            "ipa": r.get("ipa"), "other_ipa": r.get("other_ipa"), "source": r.get("source"),
            "proposal": proposal,
            "proposal_right": ((r.get("tried") or {}).get(proposal) or {}).get("right") if proposal else None,
            "recorded": reviewed.get(key, {}).get("recorded"),
        })
    items.sort(key=lambda i: (i["rank"] is None, i["rank"] or 0))
    return items


def by_mentions(items: list, mentions, least: int) -> tuple:
    """(the review items the KB uses at least `least` times, most used first -
    each with its count as "mentions" - and how many rarer ones were left out).
    mentions: knowledge.word_mentions of the KB; None (no KB) keeps the list as
    it was."""
    if mentions is None:
        return items, 0
    for it in items:
        it["mentions"] = mentions.get(it["word"].lower(), 0)
    kept = sorted((i for i in items if i["mentions"] >= least), key=lambda i: -i["mentions"])
    return kept, len(items) - len(kept)


def listening_paused(path: str = None) -> bool:
    """True while the /admin Names page has paused her listening - a person
    reading names aloud at the kiosk. The pause runs out on its own."""
    try:
        with open(path or LISTENING_PAUSED, encoding="utf-8") as fh:
            until = float(fh.read().strip())
    except (OSError, ValueError):
        return False
    return time.time() < until


def pause_listening(seconds: float, path: str = None):
    """Pause her listening for this long; 0 resumes it now."""
    target = path or LISTENING_PAUSED
    if seconds <= 0:
        try:
            os.remove(target)
        except OSError:
            pass
        return
    _write_atomic(target, str(time.time() + seconds))
