# Unreal Engine project

The Unreal side renders the avatar and turns the backend's WebSocket messages into speech, lip sync, expressions,
gaze and gestures. This folder contains the project's C++ modules, configuration and editor scripts. It contains no
content: the level, your MetaHuman, its blueprints and animations are yours to create, and the third-party plugins
are installed separately (see [THIRD_PARTY.md](../THIRD_PARTY.md)).

## Requirements

- Unreal Engine 5.7 with the MetaHuman plugin.
- **Runtime MetaHuman Lip Sync** (paid). The `KioskAvatar` module includes its headers, so the project does not
  compile without it.
- **Runtime Audio Importer** (paid on Fab), to load each spoken WAV at run time.
- Visual Studio 2022 with the "Game development with C++" workload.

## Modules

| Module | Type | Contents |
|---|---|---|
| `KioskAvatar` | Runtime | `AWebSocketManager`, `UEyeGazeComponent`, `UAvatarAnimInstance`, `UGestureLibrary`, `UEmotionMoodLibrary`, `UFabricPinComponent` |
| `KioskAvatarEditor` | Editor only | `UKioskAnimGraphTools`: adds animation-graph nodes from Python, used by the scripts in `Scripts/` |

## What to build in the editor

1. **Level**: the project starts in `/Game/MainLevel` (`Config/DefaultEngine.ini`, `EditorStartupMap` and
   `GameDefaultMap`). Create that level or change the setting. Add a camera that frames her head and upper body.
2. **Avatar**: create your MetaHuman and place its blueprint in the level.
   - Set `UAvatarAnimInstance` as the parent class of the body's Animation Blueprint. It keeps her head turned
     toward the visitor and plays gestures.
   - In the body Animation Blueprint, add a Slot node for `DefaultSlot` (the `GestureSlot` property), applied
     additively over the idle. Gesture clips play into it.
   - Add `UEyeGazeComponent` to the avatar blueprint (`Scripts/add_eye_gaze.py` does this). It aims her eyes at the
     camera, turned toward the visitor. Its defaults expect the MetaHuman face component `Face` and the bones
     `FACIAL_L_Eye`, `FACIAL_R_Eye` and `head`.
   - In the face Animation Blueprint, set up Runtime MetaHuman Lip Sync as its documentation describes.
     `Scripts/add_eye_curve_node.py` adds per-eye gaze control, and `Scripts/set_rest_smile.py` adds a faint resting
     smile.
3. **WebSocket actor**: place `AWebSocketManager` (or a blueprint based on it) in the level. It connects to the
   orchestrator at `ServerURL` (default `ws://127.0.0.1:8765`) and reconnects on its own. Bind its events in the
   blueprint:

   | Event | Do this |
   |---|---|
   | `OnSpeakReceived(WavFilePath)` | Load the WAV with Runtime Audio Importer, feed it to the lip-sync generator, play it |
   | `OnStopReceived` | Stop her audio; `UEmotionMoodLibrary::FeedSilenceToGenerator` settles the mouth |
   | `OnEmotionChanged(Emotion, Intensity)` | `UEmotionMoodLibrary::SetMoodFromEmotion` sets the lip-sync mood / expression |
   | `OnStateChanged(State)` | Optional: show listening / thinking / speaking |
   | `OnIndicatorChanged(Value)` | Optional: status indicator color |
   | `OnNameReceived(UserName)` | Optional: the recognized visitor's name |
   | `OnGazeReceived(Yaw, Pitch)` | Handled by `UEyeGazeComponent`; bind only for custom gaze |
   | `OnGestureReceived(Intent, Clip)` | Informational: `UAvatarAnimInstance` already plays the gesture |

4. **Gestures** (optional): create a `UGestureLibrary` data asset at `/Game/Gestures/DA_GestureLibrary` and add your
   own short upper-body clips. Each clip has an `Intent` (`agree`, `disagree`, `emphasize`, `explain`, `list`,
   `contrast`, `unsure`, `empathize`, `welcome`, `surprise`, `listen`, `talk`, `idle`), optional `AlsoIntents`, a
   `Role` (`speaking`, `listening`, `idle`), a `Tone`, a `Body` region and a `Strength` (0-1). The backend picks
   an intent for each sentence, and the avatar plays the best-fitting enabled clip.
5. **Status indicator** (optional): `Scripts/make_status_pin_obj.py` writes the indicator mesh (import it),
   `Scripts/build_status_pin.py` creates its materials, `Scripts/apply_status_pin.py` applies them, and
   `UFabricPinComponent` (`Scripts/seat_pin_on_top.py`) keeps it seated on her top garment.

The editor scripts run in the Unreal editor (Tools > Execute Python Script, or `py` in the console). Set the asset
paths at the top of each script to your own assets first.

## Messages

The orchestrator sends JSON messages over the WebSocket:

| `type` | Fields | Meaning |
|---|---|---|
| `speak` | `path` | Play this WAV file with lip sync |
| `stop` | | Stop speaking now (the visitor interrupted) |
| `emotion` | `value`, `intensity` | Facial expression for the reply |
| `gesture` | `intent`, `emotion`, `intensity`, `role`, `interrupt` | Play a gesture clip |
| `gaze` | `present`, `yaw`, `pitch`, `distance_in`, `focus_in` | Where the visitor is, from her eyes |
| `state` | `value` | Conversation state |
| `indicator` | `value` | Status indicator |
| `name` | `value` | The recognized visitor's name |
| `av_sync` | `voice_delay_ms` | Delay her audio so it lines up with her lips |

Unreal sends `{"type":"ready"}` when it connects. `UEyeGazeComponent` also sends `eyes_on_screen` (where it draws
her eyes), which the backend uses for gaze.

## Packaging

`server\scripts\package-kiosk.bat` cooks and packages the project into `packaged\` next to the repository.
