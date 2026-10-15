"""
Jabra keeper - holds a listening stream on the Jabra while the avatar's system runs.

The Jabra Speak 710 drops its own mute (red lights -> white) whenever the last
capture stream on it closes, and doesn't tell Windows (measured: muted,
orchestrator stopped, live one second later). The orchestrator holds that
stream, so without this process every orchestrator restart would unmute a room
mic that was muted on purpose. This process keeps a second stream open for as
long as the system runs, so a restart never closes the last one and the mute
stays exactly as it was set.

It opens the stream with the orchestrator's own settings (WASAPI,
streamCategory=Speech, auto-convert) so the Jabra works the same for her, and
throws away everything it hears: nothing is recorded, sent or analysed, and the
mute is never touched.

Started by scripts/start-orchestrator.bat (so also by start.bat) - a second copy
exits at once. The stop scripts leave it running, so even a full stop keeps
the mute. To stop it on purpose, end the pythonw.exe running
scripts\jabra_keeper.py - the Jabra then drops its mute. Log: logs/jabra-keeper.log
"""

import ctypes
import logging
import os
import sys
import time

import sounddevice as sd

NAME_PATTERN = "Jabra Speak"        # as the orchestrator's MIC_DEVICE_NAME_PATTERN
LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "jabra-keeper.log")


def find_jabra():
    """The Jabra's WASAPI capture device (index, name), or (None, None)."""
    wasapi = next((i for i, a in enumerate(sd.query_hostapis()) if "WASAPI" in a["name"]), None)
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == wasapi and d["max_input_channels"] > 0 and NAME_PATTERN in d["name"]:
            return i, d["name"]
    return None, None


def speech_settings():
    """The orchestrator's stream settings: streamCategory=Speech (Jabra firmware
    barge-in workaround) and auto-convert to 16 kHz."""
    from sounddevice import _ffi, _lib
    info = _ffi.new("PaWasapiStreamInfo*", dict(
        size=_ffi.sizeof("PaWasapiStreamInfo"), hostApiType=_lib.paWASAPI, version=1,
        flags=_lib.paWinWasapiAutoConvert, streamCategory=_lib.eAudioCategorySpeech))

    class Settings:
        pass
    settings = Settings()
    settings._streaminfo = info
    return settings


_MUTEX = None


def only_copy() -> bool:
    """A named mutex (held for the process's life) makes a second keeper exit at once."""
    global _MUTEX
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    _MUTEX = kernel32.CreateMutexW(None, False, "Local\\KioskAvatarJabraKeeper")
    return ctypes.get_last_error() != 183                    # ERROR_ALREADY_EXISTS


def main() -> int:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    logging.basicConfig(filename=LOG, level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    if not only_copy():
        return 0
    logging.info("Jabra keeper started")
    stream, missing_logged = None, False
    while True:
        try:
            if stream is None:
                sd._terminate()                      # fresh device list (the Jabra may have been replugged)
                sd._initialize()
                index, name = find_jabra()
                if index is None:
                    if not missing_logged:
                        logging.info("no Jabra found - checking again every 10 s")
                        missing_logged = True
                    time.sleep(10)
                    continue
                stream = sd.InputStream(device=index, samplerate=16000, channels=1, blocksize=512, dtype="float32",
                                        callback=lambda *args: None, extra_settings=speech_settings())
                stream.start()
                missing_logged = False
                logging.info(f"holding a listening stream on [{index}] {name} (audio discarded, mute untouched)")
            if not stream.active:
                raise RuntimeError("stream stopped")
            time.sleep(5)
        except Exception as e:
            logging.warning(f"stream lost ({e}) - reopening in 10 s")
            try:
                stream.close()
            except Exception:
                pass
            stream = None
            time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
