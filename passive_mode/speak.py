import os, tempfile, threading, queue, subprocess, sys
from elevenlabs import ElevenLabs
from dotenv import load_dotenv  

load_dotenv()  # Load environment variables from .env file

API_KEY  = os.getenv("ELEVENLABS_API_KEY") 
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_multilingual_v2"

_client = ElevenLabs(api_key=API_KEY)
_q: "queue.Queue[str|None]" = queue.Queue(maxsize=1)

def _play_file(path: str):
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("linux"):
            subprocess.run(["mpg123", "-q", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # fallback: simple playsound on Windows
            import playsound
            playsound.playsound(path)
    except Exception as e:
        print("[TTS play error]", e)

def _worker():
    while True:
        text = _q.get()
        if text is None:
            break
        try:
            stream = _client.text_to_speech.convert(
                voice_id=VOICE_ID, model_id=MODEL_ID,
                output_format="mp3_22050_32", text=text
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                for chunk in stream:
                    f.write(chunk)
                path = f.name
            _play_file(path)
            try: os.remove(path)
            except: pass
        except Exception as e:
            print("[TTS error]", e)

_thread = threading.Thread(target=_worker, daemon=True)
_thread.start()

def speak_submit(text: str):
    if not text or not API_KEY:
        return
    # keep only latest request to avoid backlog
    try:
        while not _q.empty():
            try: _q.get_nowait()
            except: break
        _q.put_nowait(text)
        # debug (optional): print("TTS queued:", text)
    except queue.Full:
        pass

def shutdown():
    try: _q.put_nowait(None)
    except: pass

