import cv2
import torch
import threading
import asyncio
import time

from shared import tts
from active_mode.client import ws_send


tts.start()  # start TTS worker once

# Track active run and a "pending" trigger set by key press
active_in_flight = {"flag": False}


_mute_until = 0.0          # already there

def mute_tts(seconds: float = 2.0):
    """Mute passive TTS for `seconds` from now."""
    global _mute_until
    _mute_until = max(_mute_until, time.time() + seconds)

def tts_is_muted() -> bool:
    return time.time() < _mute_until   

def run_active_once(frame, uri="ws://127.0.0.1:8000/ws"):
    """
    Sends one snapshot to the WS server using your existing ws_send(),
    and speaks the partial + final (handled inside ws_send).
    Runs in a background thread; sets active_in_flight flag while running.
    """
    if active_in_flight["flag"]:
        return  # already running

    # Stop any passive speech for clarity (works if detector uses active_mode.tts)
    tts.cancel()
    active_in_flight["flag"] = True

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    if not ok:
        active_in_flight["flag"] = False
        return
    image_bytes = buf.tobytes()

    def _runner():
        try:
            asyncio.run(ws_send(uri, image_bytes))
        finally:
            active_in_flight["flag"] = False

    threading.Thread(target=_runner, daemon=True).start()