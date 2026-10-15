"""Compare two takes of a name in her voice by how they SOUND, not by a transcription (used by the names page).

Once she says a name right, she has to say it that way every time. Her voice (Chatterbox) re-performs every take,
so the page searches for a spelling she repeats. The phoneme recognizer's transcription is not a reliable judge of
a new take: it can mishear a name's ending, and it ignores pacing. This judges a take against the approved take,
directly:

  * where the name is: the recognizer's phoneme events with their 20 ms frames (pronunciation_worker features),
    folded like fold_heard, and the sentence alignment of pronunciation.name_in_sentence(span=True);
  * sound: dynamic time warping (DTW: the best way to line up two stretches of speech that run at different
    speeds) over the recognizer's last-layer features, cosine distance per frame pair;
  * pacing: the length ratio of the two names, plus how much uneven stretching the line-up needed;
  * stress: how well the loudness contours agree along the line-up. Pitch is measured too (from each take's own middle,
    so a lower voice compares with hers) and reported, but not scored: her pitch follows the sentence - a question
    rises, a list item doesn't - so across the 8 sentences it adds noise, not stress (measured: same separation,
    fewer true matches).

Pure numpy; the recognizer and her voice are passed in, so tests run without either.
"""

from __future__ import annotations

import math

import numpy as np

from .pronunciation import _FOLD, _VOWELS, _pieces, fold_heard, name_in_sentence

# Calibrated on 66 takes of three unusual product names (11 spellings x 6 sentences): takes the recognizer heard the
# same way vs differently separate with AUC 0.92 on sound alone, 0.93 with a little timing and stress. Timing is kept
# at 0.2 although that label can't hear pacing - pacing is what a listener hears.
# At MATCH 0.16, 69% of same-sounding pairs match and 8% of different ones (strict on purpose: a wrong "match" is
# worse than a missed one).
W_TIMING, W_STRESS = 0.2, 0.1
MATCH = 0.16          # a take whose distance to the approved take is at most this counts as "the same"
MARGIN = 8            # frames (160 ms) kept around a judged take's name; the name is found inside (open_ends)


def fold_spans(spans: list) -> list:
    """The recognizer's phoneme events [(phoneme, first frame, past-the-last frame)] folded exactly like
    fold_heard, keeping frames: a phoneme split into pieces shares its frames, an 'ə + ɹ' merged into 'ɚ'
    covers both."""
    pieces = []
    for p, f0, f1 in spans:
        parts = [_FOLD.get(x, x) for x in _pieces(p)]
        parts = [x for x in parts if x and x[0].isalpha()]
        if not parts:
            continue
        step = max(1, (f1 - f0)) / len(parts)
        for k, x in enumerate(parts):
            a = f0 + int(round(k * step))
            pieces.append([x, a, max(a + 1, f0 + int(round((k + 1) * step)))])
    out = []
    for k, (sym, a, b) in enumerate(pieces):
        nxt = pieces[k + 1][0] if k + 1 < len(pieces) else None
        if sym == "ɹ" and out and out[-1][0] in ("ə", "ɚ") and nxt not in _VOWELS:
            out[-1] = ["ɚ", out[-1][1], b]
            continue
        out.append([sym, a, b])
    assert [s for s, _, _ in out] == fold_heard([p for p, _, _ in spans]), "fold_spans drifted from fold_heard"
    return [tuple(x) for x in out]


def name_frames(heard: dict, before, target, after) -> tuple | None:
    """(first, past-the-last) frame of the name in a take of a known sentence, or None: from the onset of its
    first sound to the end of its last, as the recognizer marks them. heard is the worker's features reply."""
    folded = fold_spans(heard["spans"])
    window = name_in_sentence(before, target, after, [s for s, _, _ in folded], span=True)
    if not window:
        return None
    a, b = window
    start, end = folded[a][1], folded[b - 1][2]
    return (start, end) if end - start >= 3 else None


def segment(heard: dict, frames: tuple, margin: int = 0) -> dict:
    """The name's frames; with a margin, that many more on each side - for a take being judged, whose name is
    then found inside it (distance(open_ends=True)), so a neighbour's sound at the edge can't skew it."""
    a, b = max(0, frames[0] - margin), min(heard["hidden"].shape[0], frames[1] + margin)
    out = {"hidden": heard["hidden"][a:b], "energy": heard["energy"][a:b]}
    if heard.get("pitch") is not None:
        out["pitch"] = heard["pitch"][a:b]
    return out


def _dtw(cost: np.ndarray, open_ends: bool = False) -> list:
    """The cheapest monotonic line-up of the two sequences (steps: both, first only, second only). With
    open_ends the whole first sequence is lined up with any stretch of the second."""
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    if open_ends:
        acc[0, :] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    path, i = [], n
    j = int(np.argmin(acc[n, 1:])) + 1 if open_ends else m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    return path[::-1]


def _z(x: np.ndarray) -> np.ndarray:
    x = np.log(np.maximum(x, 1e-6))
    return (x - x.mean()) / (x.std() + 1e-6)


def _pitch_agreement(pa, pb, path: list):
    """How alike the two pitch contours move along the line-up (correlation, -1..1), each measured from its own
    middle so a lower voice and hers compare; None when there is too little voiced speech or both are flat."""
    if pa is None or pb is None:
        return None
    pairs = [(pa[i], pb[j]) for i, j in path if not (np.isnan(pa[i]) or np.isnan(pb[j]))]
    if len(pairs) < 5:
        return None
    x, y = np.array([u for u, _ in pairs]), np.array([v for _, v in pairs])
    x, y = x - np.median(x), y - np.median(y)
    if x.std() < 0.3 or y.std() < 0.3:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def distance(a: dict, b: dict, open_ends: bool = False) -> dict:
    """How far take b's name is from take a's (the approved one): 0 = the same, larger = sounds different (see
    MATCH). With open_ends, b is the name plus a margin (segment) and the name is found inside it."""
    ha = a["hidden"] / (np.linalg.norm(a["hidden"], axis=1, keepdims=True) + 1e-6)
    hb = b["hidden"] / (np.linalg.norm(b["hidden"], axis=1, keepdims=True) + 1e-6)
    cost = 1.0 - ha @ hb.T
    path = _dtw(cost, open_ends)
    sound = float(np.mean([cost[i, j] for i, j in path]))
    na, nb = len(ha), path[-1][1] - path[0][1] + 1
    ratio = abs(math.log(nb / na))
    bends = sum(1 for (i0, j0), (i1, j1) in zip(path, path[1:]) if (i1 - i0) + (j1 - j0) == 1)
    uneven = max(0.0, bends / max(1, len(path) - 1) - abs(na - nb) / max(na, nb))
    timing = ratio + uneven
    ea, eb = _z(a["energy"]), _z(b["energy"])
    xs, ys = np.array([ea[i] for i, _ in path]), np.array([eb[j] for _, j in path])
    corr = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 1e-6 and ys.std() > 1e-6 else 0.0
    stress = (1.0 - corr) / 2.0
    pitch = _pitch_agreement(a.get("pitch"), b.get("pitch"), path)       # reported only (see the module notes)
    total = sound + W_TIMING * timing + W_STRESS * stress
    return {"sound": round(sound, 4), "timing": round(timing, 4), "stress": round(stress, 4), "total": round(total, 4),
            "pitch": round(pitch, 3) if pitch is not None else None}


def matches(d: dict) -> bool:
    return d["total"] <= MATCH
