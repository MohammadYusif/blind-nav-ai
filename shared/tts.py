import os, threading, queue, time, subprocess, signal
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_flash_v2_5")  
API_KEY  = os.getenv("ELEVENLABS_API_KEY")

_client = ElevenLabs(api_key=API_KEY)

# --------─ internal state ----------------------------------------------------
_q: "queue.Queue[tuple[str, float] | None]" = queue.Queue(maxsize=1)  # (text, enqueue_ts)
_stop            = threading.Event()
_worker_started  = False
_current_stream  = None
_lock            = threading.Lock()
_ref_t0          = None             
_player_proc     = None            
# -----------------------------------------------------------------------------

def set_ref_time(t0: float | None = None):
    """Mark a reference time (e.g., when SPACE is pressed) for end-to-end timing."""
    global _ref_t0
    _ref_t0 = time.perf_counter() if t0 is None else t0

# ── low-level audio helper ----------------------------------------------------
def _play_stream(stream):
    """
    Feed ElevenLabs chunks to an mpv subprocess so we can terminate
    playback instantly from cancel().
    """
    global _player_proc
    _player_proc = subprocess.Popen(
        ["mpv", "--no-terminal", "--no-cache", "--audio-display=no", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for chunk in stream:
            if _player_proc.poll() is not None:    # mpv already closed
                break
            _player_proc.stdin.write(chunk)
        # Close stdin so mpv finishes gracefully
        _player_proc.stdin.close()
        _player_proc.wait()
    finally:
        _player_proc = None
# -----------------------------------------------------------------------------

def _worker():
    global _current_stream, _ref_t0
    while not _stop.is_set():
        try:
            item = _q.get(timeout=0.1)
        except queue.Empty:
            continue

        # Sentinel for cancel()
        if item is None:
            with _lock:
                if _current_stream is not None:
                    try:
                        _current_stream.close()
                    except Exception:
                        pass
                    _current_stream = None
            with _q.mutex:
                _q.queue.clear()
            print("[TTS] ✋ cancel — stopped current audio and cleared queue")
            continue

        text, enq_t = item
        text = text.strip()
        if not text:
            continue

        try:
            # Clamp speed from env
            try:
                spd = float(os.getenv("ELEVEN_SPEED", "1.18"))
            except Exception:
                spd = 1.18
            spd = max(0.7, min(1.2, spd))  # respect API limits

            s = _client.text_to_speech.stream(
                text=text,
                voice_id=VOICE_ID,
                model_id=MODEL_ID,
                language_code="ar",
                output_format="mp3_22050_32",
                voice_settings={
                    "stability": 0.7,
                    "similarity_boost": 0.85,
                    "style": 0.0,
                    "use_speaker_boost": True,
                    "speed": spd,
                },
            )
            with _lock:
                _current_stream = s

            start_t = time.perf_counter()
            msg = (f"[TTS] ▶ start len={len(text)} chars | enqueue→start={start_t - enq_t:.3f}s")
            if _ref_t0 is not None:
                msg += f" | SPACE→start={start_t - _ref_t0:.3f}s"
            print(msg)

            _play_stream(s)   # ← instant-kill capable playback

            end_t = time.perf_counter()
            msg = (f"[TTS] ■ end | speak={end_t - start_t:.3f}s | enqueue→end={end_t - enq_t:.3f}s")
            if _ref_t0 is not None:
                msg += f" | SPACE→end={end_t - _ref_t0:.3f}s"
            print(msg)

        except Exception as e:
            err_t = time.perf_counter()
            msg = f"[TTS] ✖ error: {e} | since enqueue={err_t - enq_t:.3f}s"
            if _ref_t0 is not None:
                msg += f" | since SPACE={err_t - _ref_t0:.3f}s"
            print(msg)
        finally:
            with _lock:
                _current_stream = None


# ── public helpers -----------------------------------------------------------
def start():
    global _worker_started
    if not _worker_started:
        threading.Thread(target=_worker, daemon=True).start()
        _worker_started = True
        print("[TTS] worker started")

def enqueue(text: str):
    """Queue a chunk for speaking; drops oldest if queue is full."""
    try:
        _q.put_nowait((text, time.perf_counter()))
    except queue.Full:
        try:
            _q.get_nowait()  # drop oldest
        except Exception:
            pass
        _q.put_nowait((text, time.perf_counter()))

def cancel():
    """
    Stop whatever is playing *right now* and discard anything queued.
    Safe to call from any thread.
    """
    global _current_stream, _player_proc
    with _lock:
        # 1) close the live ElevenLabs HTTP stream
        if _current_stream is not None:
            try:
                _current_stream.close()
            except Exception:
                pass
            _current_stream = None

        # 2) nuke mpv if it's still running
        if _player_proc and _player_proc.poll() is None:
            try:
                _player_proc.terminate()  # SIGTERM → immediate stop
            except Exception:
                pass
            _player_proc = None

        # 3) clear any queued text
        with _q.mutex:
            _q.queue.clear()

    # 4) wake the worker if it's blocked on .get()
    try:
        _q.put_nowait(None)  # sentinel; worker will flush and ignore
    except queue.Full:
        pass

def shutdown():
    _stop.set()
    cancel()