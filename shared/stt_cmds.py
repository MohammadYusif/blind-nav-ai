# shared/stt_cmds.py
# EN-only command STT with auto-calibrated near-field gate + fuzzy-matching.
import os, queue, threading, time
import numpy as np
import sounddevice as sd
from difflib import SequenceMatcher
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_SEC   = 1.0      # long enough to catch whole word
BOOST_DB    = 6.0      # trigger = ambient + BOOST_DB (auto-calibrated)

# Optional: fix mic device index via env
MIC_INDEX = os.getenv("MIC_INDEX")
if MIC_INDEX is not None:
    try: MIC_INDEX = int(MIC_INDEX)
    except: MIC_INDEX = None

# English-only tiny for speed+English accuracy
_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# --- Command keywords ---------------------------------------------------------
ACTIVE_WORDS  = (
    "active", "activate", "activ", "activity", "let's go", "lets go",
    "begin", "start", "go", "do this", "octave", "octive", "octav", "oktav"
)
PASSIVE_WORDS = ("passive", "resume", "alerts", "continue")
STOP_WORDS    = ("stop", "mute", "cancel", "that's it", "thats it", "enough", "stop it", "cancel it")

FUZZY_THRESH = 0.78   # ~78% similarity is usually good for short words

# Whisper biasing: push it toward these words
BIAS_PROMPT = (
    "You may hear short voice commands. The valid commands are single words: "
    "'active', 'passive', 'stop'."
)

# --- Utils --------------------------------------------------------------------
def _rms_dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    if x.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(x)))
    if not np.isfinite(rms) or rms <= 1e-9:
        return -120.0
    return 20.0 * np.log10(rms + 1e-12)

def _pre_emphasis(x: np.ndarray, p: float = 0.97) -> np.ndarray:
    if x.size <= 1:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - p * x[:-1]
    return np.nan_to_num(y, copy=False)

def _record(seconds: float = CHUNK_SEC, sr: int = SAMPLE_RATE) -> np.ndarray:
    print("[STT] recording...", flush=True)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=MIC_INDEX)
    sd.wait()
    return audio[:, 0]

def _tokens(t: str):
    # simple tokenization: split on whitespace and punctuation
    out = []
    w = ""
    for ch in t.lower():
        if ch.isalpha():
            w += ch
        else:
            if w:
                out.append(w)
                w = ""
    if w:
        out.append(w)
    return out

def _best_fuzzy(word: str, candidates: tuple[str, ...]) -> float:
    return max(SequenceMatcher(None, word, c).ratio() for c in candidates)

def _classify(text: str) -> str | None:
    """
    Fuzzy-match each token to ACTIVE/PASSIVE/STOP lists and pick the highest score.
    """
    if not text:
        return None
    toks = _tokens(text)
    if not toks:
        return None

    best_cmd, best_score = None, 0.0
    for tok in toks:
        s_act = _best_fuzzy(tok, ACTIVE_WORDS)
        s_pas = _best_fuzzy(tok, PASSIVE_WORDS)
        s_stp = _best_fuzzy(tok, STOP_WORDS)
        sc, cmd = max((s_act, "ACTIVE"), (s_pas, "PASSIVE"), (s_stp, "STOP"))
        if sc > best_score:
            best_score, best_cmd = sc, cmd

    if best_score >= FUZZY_THRESH:
        print(f"[STT] fuzzy match: {best_cmd} ({best_score:.2f})", flush=True)
        return best_cmd
    return None

def _calibrate_ambient(samples: int = 3) -> float:
    print("[STT] calibrating ambient…", flush=True)
    vals = []
    for _ in range(samples):
        raw = _record(0.6, SAMPLE_RATE)
        pe  = _pre_emphasis(raw)
        vals.append(_rms_dbfs(pe))
    amb = np.median(vals)
    trig = max(-50.0, min(-28.0, amb + BOOST_DB))  # clamp to sane range
    print(f"[STT] ambient ~ {amb:.1f} dBFS → trigger {trig:.1f} dBFS", flush=True)
    return trig

# --- Main loop ----------------------------------------------------------------
def listen_cmd_loop(cmd_queue: "queue.Queue[str]") -> None:
    # Mic info
    try:
        dev = sd.query_devices(MIC_INDEX if MIC_INDEX is not None else sd.default.device[0])
        print(f"[STT] using input device: {dev.get('name', dev)}", flush=True)
    except Exception:
        print("[STT] note: could not resolve mic name; using default.", flush=True)

    trigger_db = _calibrate_ambient()

    while True:
        try:
            raw = _record(CHUNK_SEC, SAMPLE_RATE)
            pe  = _pre_emphasis(raw)
            db  = _rms_dbfs(pe)
            print(f"[STT] level: {db:.1f} dBFS (trigger {trigger_db:.1f})", flush=True)
            if db < trigger_db:
                print("[STT] too quiet → skip", flush=True)
                continue

            segments, _ = _model.transcribe(
                pe,
                language="en",
                task="transcribe",
                vad_filter=True,
                beam_size=1,
                temperature=0.0,
                without_timestamps=True,
                initial_prompt=BIAS_PROMPT,   # ← bias toward command words
            )
            text = "".join(seg.text for seg in segments).strip()
            print(f"[STT] heard: {text if text else '(silence)'}", flush=True)

            cmd = _classify(text)
            if cmd:
                print(f"[STT] CMD -> {cmd}", flush=True)
                try:
                    cmd_queue.put_nowait(cmd)
                except queue.Full:
                    pass
            else:
                print("[STT] (no command keyword)", flush=True)

        except Exception as e:
            print(f"[STT] error: {e}", flush=True)
            time.sleep(0.1)

def start_cmd_listener(cmd_queue: "queue.Queue[str]") -> None:
    th = threading.Thread(target=listen_cmd_loop, args=(cmd_queue,), daemon=True)
    th.start()