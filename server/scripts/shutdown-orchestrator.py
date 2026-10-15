"""Send a graceful shutdown WS message to the orchestrator on :8765.
Used by stop-stack.* to avoid Stop-Process -Force, which wedges the C920
in MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED state. Exits 0 on success,
1 if WS unreachable (likely orchestrator not running, OK to fall through
to a regular kill)."""
import asyncio
import json
import sys

import websockets


async def main() -> int:
    try:
        async with websockets.connect("ws://127.0.0.1:8765", open_timeout=2.0) as ws:
            await ws.send(json.dumps({"type": "shutdown"}))
            print("shutdown sent", flush=True)
            return 0
    except Exception as e:
        print(f"shutdown WS connect failed: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
