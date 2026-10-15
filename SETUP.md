# Setup

This guide installs the backend (Python), the language model server (llama.cpp) and the Unreal Engine project on
one Windows 11 PC. None of the models, engines or plugins are included in this repository: you download or buy
each one yourself under its own license ([THIRD_PARTY.md](THIRD_PARTY.md)).

Folder layout after setup (only `server/` code and `unreal/` code come from this repository):

```
3DHumanAvatar/
  server/                  backend and orchestrator (Python)
    .venv/                 your Python environment
    models/                downloaded models (step 3)
    llama-server/          llama.cpp build (step 2)
    ReferenceVoice.wav     the voice she speaks with (step 4)
    kiosk_config.ini       your settings (step 6)
    scripts/               start and stop scripts
  unreal/                  Unreal Engine 5.7 project (C++ modules, config, editor scripts)
    Plugins/               third-party plugins you install (step 5)
    Content/               your level, MetaHuman and blueprints (step 5)
```

## 1. Python environment

1. Install Python 3.11 (64-bit).
2. In `server\`, create and activate a virtual environment, then install the packages:
   ```
   cd server
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install --no-deps -r requirements.txt
   ```
   `requirements.txt` is the exact, tested package set, including the CUDA 12.8 build of PyTorch and a prebuilt
   `dlib`. Install it with `--no-deps` as shown: the text-to-speech package declares older PyTorch and NumPy versions
   than the tested set, and `--no-deps` keeps the tested versions.

## 2. Language model server

1. Download a Windows CUDA build of llama.cpp (`llama-server.exe`) from the llama.cpp releases page and unpack it
   to `server\llama-server\`.
2. Download the language model GGUF file (see THIRD_PARTY.md) to `server\models\`.
3. If your file or folder names differ, edit the two lines at the top of `server\scripts\start-llama.bat`
   (`LLAMA_SERVER` and `LLM_MODEL`). The script pins the model to the first GPU (`CUDA_VISIBLE_DEVICES=0`) and uses
   the model's full context (`-c 262144`); lower `-c` if your GPU has less memory.

## 3. Other models

Download these to `server\models\` (sources and licenses in THIRD_PARTY.md):

| File or folder in `server\models\` | Used for |
|---|---|
| `faster-whisper-large-v3-turbo\` | Speech recognition |
| `face_landmarker.task` | Face tracking (eye contact, glances) |
| `enet_b0_8_best_afew.onnx` | Facial expression, for the mood estimate |
| `wav2vec2-lv-60-espeak-cv-ft\` | Pronunciation check (optional) |
| `cmudict\cmudict.dict` | Pronunciation check (optional) |

These are downloaded automatically from their official sources on first run (internet needed once): the
text-to-speech model, the speech-emotion model and the voice activity detector. The text-to-speech download asks for
a free Hugging Face account: sign in once with `huggingface-cli login` (or set `HF_TOKEN`) before the first run. The
face recognition models come with the `face_recognition_models` package.

## 4. Her voice

Text-to-speech clones the voice in `server\ReferenceVoice.wav`: 10 to 20 seconds of clean speech from one speaker,
mono WAV. **Use only a voice you have permission to use.** To keep the file elsewhere, set the environment variable
`REFERENCE_VOICE` to its path.

## 5. Unreal Engine project

1. Install Unreal Engine 5.7 and the MetaHuman plugin, and Visual Studio 2022 with the "Game development with C++"
   workload.
2. Install the two third-party plugins into `unreal\Plugins\` (see THIRD_PARTY.md): Runtime MetaHuman Lip Sync
   and Runtime Audio Importer, both paid on Fab.
3. Right-click `unreal\KioskAvatar.uproject`, choose "Generate Visual Studio project files", then build the
   `KioskAvatarEditor` target (Development Editor, Win64).
4. Build your scene: a level, your own MetaHuman, the blueprints that connect it to the C++ module, and gesture
   clips. [unreal/README.md](unreal/README.md) describes what the code expects and which editor scripts automate
   the animation-graph changes.

## 6. Settings

Edit `server\kiosk_config.ini`. Each setting is described in the file. At minimum:

- `[camera]` and `[monitor]`: where your webcam sits relative to the screen, and the screen's size and height.
  These drive her eye contact.
- `[avatar] name`: her name (default Kiah).
- `[identity] mode`: `multi_user` (recognition, sign-in and memory per person) or `single_user` (one private
  memory, no face data processed). Read [Responsible use](README.md#responsible-use) first.
- `[kb]`: set `enabled = true` and `path` to a folder of `.md`, `.txt` or `.csv` files to give her a knowledge
  base.

## 7. Run

1. Start the backend stack: `server\scripts\start.bat`. It starts llama-server, then the backend, then the
   orchestrator, and waits until each is healthy. Logs go to `server\logs\`.
2. Open `unreal\KioskAvatar.uproject` and press Play (or package it with `server\scripts\package-kiosk.bat`).
3. Open the admin pages in a browser on the same PC:
   - `http://127.0.0.1:8100/admin`: components, modes and devices
   - `http://127.0.0.1:8100/dashboard`: response-time dashboard
   - `http://127.0.0.1:8100/admin/names`: how she pronounces names in your knowledge base
4. Stop everything with `server\scripts\stop.bat`. It shuts the orchestrator down gracefully first, which releases
   the webcam cleanly.

## Troubleshooting

- **She hears herself.** Use a speakerphone with hardware echo cancellation, and make it both the input and the
  output device.
- **Out of GPU memory.** Lower `-c` in `start-llama.bat`, or use a smaller quantization of the model.
- **Eye contact looks off.** Re-measure `[camera]` and `[monitor]`, then adjust `trim_yaw_deg` / `trim_pitch_deg` in
  `[tracking]`. Saved changes apply within seconds.
