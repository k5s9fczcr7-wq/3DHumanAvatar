"""
Phoneme recognizer for the pronunciation check (pronunciation.py).

Runs as a short-lived subprocess of the orchestrator, so its model - and
torch/transformers - never load into the 24/7 orchestrator process; it exits
when the check is done. It writes down the speech SOUNDS in a clip (espeak-
style IPA phonemes), not words, which is what lets the check hear that a name
was said wrong even when speech recognition would still spell it right.

Model: facebook/wav2vec2-lv-60-espeak-cv-ft, in models/ (see MODEL_DIR).

Protocol, one JSON object per line:
  out first:  {"ready": true, "device": "cuda:1"}
  in:         {"id": 3, "audio": "<base64 16-bit PCM>", "rate": 24000}
  out:        {"id": 3, "phonemes": ["s", "ɛ", "f", ...]}  or  {"id": 3, "error": "..."}
With "features": true in the request the reply also carries, per 20 ms frame, what the
names page needs to compare two takes of a name by sound and timing (pron_match.py):
  "spans": [["s", 12, 14], ...]   each phoneme with its first and past-the-last frame
  "hidden": "<base64 float16>", "dim": 1024, "frames": T   the model's last-layer features (T x dim)
  "energy": "<base64 float16>"    loudness per frame (T)
  "pitch": "<base64 float16>"     voice pitch per frame in semitones above 100 Hz, NaN where unvoiced (T)
Exits at end of input.

Usage: python -m orchestrator.pronunciation_worker [--device cuda:1]
"""

import argparse
import base64
import json
import math
import os
import sys

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "wav2vec2-lv-60-espeak-cv-ft")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    device = parser.parse_args().device

    import torch
    from scipy.signal import resample_poly
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    features = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
    # Loaded in half precision (half the RAM of a full-precision copy).
    # (low_cpu_mem_usage would need the accelerate package, not installed.)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR, torch_dtype=dtype).to(device).eval()
    with open(os.path.join(MODEL_DIR, "vocab.json"), encoding="utf-8") as fh:
        vocab = json.load(fh)
    symbol = {i: s for s, i in vocab.items()}
    special = {i for s, i in vocab.items() if s.startswith("<")}
    print(json.dumps({"ready": True, "device": device}), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        try:
            pcm = np.frombuffer(base64.b64decode(req["audio"]), dtype=np.int16).astype(np.float32) / 32768.0
            rate = int(req.get("rate", 24000))
            if rate != 16000:
                g = math.gcd(16000, rate)
                pcm = resample_poly(pcm, 16000 // g, rate // g).astype(np.float32)
            x = features(pcm, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            want = bool(req.get("features"))
            with torch.no_grad():
                res = model(x.half() if device != "cpu" else x, output_hidden_states=want)
            ids = res.logits[0].argmax(-1).tolist()
            phonemes, spans, prev = [], [], None
            for f, i in enumerate(ids):   # CTC: collapse repeats, drop blanks
                if i != prev and i not in special:
                    phonemes.append(symbol[i])
                    spans.append([symbol[i], f, f + 1])
                elif i == prev and i not in special:
                    spans[-1][2] = f + 1
                prev = i
            out = {"id": req.get("id"), "phonemes": phonemes}
            if want:
                hidden = res.hidden_states[-1][0].float().cpu().numpy().astype(np.float16)
                hop, win = 320, 400                      # the model's 20 ms frames at 16 kHz
                energy = np.array([np.sqrt(np.mean(pcm[k * hop:k * hop + win] ** 2) + 1e-10)
                                   for k in range(len(ids))], dtype=np.float16)
                import librosa      # pYIN pitch, ~0.1 s per 3 s of speech; its frame j+1 is centred on model frame j
                f0, voiced, _ = librosa.pyin(pcm, fmin=65, fmax=420, sr=16000, frame_length=1024, hop_length=hop)
                semis = np.where(voiced, 12 * np.log2(np.maximum(np.nan_to_num(f0, nan=100.0), 1.0) / 100.0), np.nan)
                pitch = np.full(len(ids), np.nan, dtype=np.float32)
                n = min(len(ids), max(0, len(semis) - 1))
                pitch[:n] = semis[1:n + 1]
                out.update(spans=spans, hidden=base64.b64encode(hidden.tobytes()).decode(), dim=int(hidden.shape[1]),
                           frames=int(hidden.shape[0]), energy=base64.b64encode(energy.tobytes()).decode(),
                           pitch=base64.b64encode(pitch.astype(np.float16).tobytes()).decode())
        except Exception as e:
            out = {"id": req.get("id"), "error": repr(e)}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
