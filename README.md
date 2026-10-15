# 3DHumanAvatar

A conversational 3D human avatar for an information kiosk. A visitor walks up and talks; the avatar listens,
answers in a natural voice with matching lip sync, facial expression and gestures, keeps eye contact, and can
answer questions from a knowledge base you provide. Everything runs locally on one Windows PC: no cloud services
are needed at run time.

The avatar's default name is **Kiah**. The name, her voice, her knowledge and her appearance are all yours to
choose.

## Features

- **Spoken conversation**: voice activity detection, speech recognition, a local large language model and
  voice-cloned text-to-speech, streamed sentence by sentence for a fast first reply.
- **Interruptions**: the visitor can talk over her; she stops and listens.
- **Lip sync, expressions and gestures**: an Unreal Engine MetaHuman whose mouth follows her voice, whose face
  shows the emotion the model chose for each reply, and whose upper body plays small gestures timed to the speech.
- **Eye contact**: a webcam tracks the visitor; her eyes follow them and glance away naturally now and then.
- **Knowledge base**: point her at a folder of text files and she answers from it (the whole folder is read into
  the model's context).
- **Pronunciation tools**: an admin page to hear how she says unusual names and fix the ones she gets wrong.
- **Optional recognition and memory**: in multi-user mode she can recognise returning visitors (face recognition
  plus a name and PIN) and remember earlier conversations; in single-user mode she keeps one private memory and
  processes no face data. See [Responsible use](#responsible-use).
- **Visitor mood**: an estimate of the visitor's mood from the camera and microphone adjusts her tone.
- **Admin pages**: component status and controls, a latency dashboard, and the pronunciation tools, served
  locally by the backend.

## How it works

```
 webcam ──► face tracking ──► gaze, glances, (recognition), mood
                                    │
 microphone ──► VAD ──► speech recognition ──► orchestrator ──► LLM (llama.cpp)
                                    │                │
                                    │                └──► text-to-speech ──► audio
                                    ▼                                          │
                     Unreal Engine (MetaHuman) ◄──── WebSocket ◄────────────────┘
                     lip sync · expressions · gestures · eyes
```

| Process | Role | Default address |
|---|---|---|
| `llama-server` (llama.cpp) | Runs the language model | `127.0.0.1:8080` |
| Backend (`server/server.py`, FastAPI) | Model gateway, text-to-speech, audio emotion, memory, admin pages | `127.0.0.1:8100` |
| Orchestrator (`server/orchestrator/`) | Microphone, camera, conversation flow, sends speech and motion to Unreal | WebSocket `127.0.0.1:8765` |
| Unreal Engine project (`unreal/`) | Renders the avatar; C++ module `KioskAvatar` receives the WebSocket messages | - |

The Unreal C++ module provides the pieces the project needs on the engine side: a WebSocket client actor that
plays her speech and applies emotions, gaze and gestures (`AWebSocketManager`), per-eye gaze control
(`UEyeGazeComponent`), a gesture clip library (`UGestureLibrary`), emotion-to-expression mapping
(`UEmotionMoodLibrary`), an animation instance that keeps her head oriented toward the visitor
(`UAvatarAnimInstance`), and a status indicator that rides on her clothing (`UFabricPinComponent`). The editor
module `KioskAvatarEditor` and the scripts in `unreal/Scripts/` apply the required animation-graph changes to
your own MetaHuman.

## Requirements

- Windows 11, Python 3.11, Unreal Engine 5.7 with the MetaHuman plugin.
- NVIDIA GPUs with enough memory for the language model, the voice model and Unreal at the same time. The
  project was developed on two GPUs: a 32 GB card for the language and voice models and a 16 GB card for Unreal
  and speech recognition.
- A webcam (developed with a Logitech C920) and a speakerphone with hardware echo cancellation (developed with
  a Jabra Speak 710), so she does not hear herself.
- Third-party models, engines and plugins that you obtain yourself, including two paid Unreal plugins (lip sync
  and run-time audio import). See [THIRD_PARTY.md](THIRD_PARTY.md).

## Getting started

See [SETUP.md](SETUP.md) for installation, model downloads, the Unreal project setup and the first run.
Settings live in `server/kiosk_config.ini`; every setting is described in the file.

## Responsible use

This project can process personal data. You are responsible for using it lawfully and fairly where you deploy it.

- **Tell people what it does.** Make it clear that they are talking to an AI system, and that a camera and
  microphone are in use.
- **Face recognition and memory** (multi-user mode) store face templates, names, PIN hashes and conversation
  summaries on the PC. Many places regulate biometric data and require informed consent, a retention policy and
  deletion on request. Use single-user mode, or turn recognition off, where you can't meet those requirements.
- **Mood estimation** infers emotional state from face and voice. Some jurisdictions restrict emotion recognition
  in certain settings, such as workplaces and schools. Check the rules that apply to you.
- **Voice cloning**: only clone a voice with the speaker's permission. Audio produced by the text-to-speech model
  carries the model provider's watermark (see THIRD_PARTY.md).
- All processing and storage stay on the local PC. Keep it secured, and don't expose the backend's admin pages to
  a network.

## Third-party components

This repository contains only the project's own code. Models, engines, plugins and 3D assets are not included and
are obtained separately under their own licenses; [THIRD_PARTY.md](THIRD_PARTY.md) lists each one with its source
and license. Some of them have terms that matter for a deployment: two of the face models carry non-commercial
terms, one speech model's license is unclear, and two Unreal plugins are paid. Read that file before you deploy.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The project is provided as is, without warranty or support.
