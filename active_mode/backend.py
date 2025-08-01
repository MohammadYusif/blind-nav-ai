# main.py
import os
import io
import json
import base64
import asyncio
from time import perf_counter

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from openai import AsyncOpenAI
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

# Only flush on full sentence end (no commas)
PUNCT = (".", "!", "؟")   # was (".", "!", "?", "،", "؟", "؛", ",")

# Be less eager to flush mid-sentence
MAX_CHARS = 100           # was 60
FLUSH_MS  = 0.60          # was 0.33


async def _prepare_image_b64(image_bytes: bytes) -> str:
    # Move PIL work off the event loop
    def _work():
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((320, 240))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=40)
        return base64.b64encode(buf.getvalue()).decode()

    return await asyncio.to_thread(_work)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for a single image from the client
            image_bytes = await websocket.receive_bytes()
            img_b64 = await _prepare_image_b64(image_bytes)

            # Stream the model output incrementally
            # --- stream the model output (two-sentence policy) ---
            # --- one short sentence policy ---
            stream = await client.chat.completions.create(
                model="gpt-4o",
                stream=True,
                temperature=0.2,
                max_tokens=26,  # short on purpose
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "أنت مساعد قصير للمكفوفين. اكتب جملة واحدة فقط (7–9 كلمات)، "
                            "من دون فواصل طويلة. اذكر أهم معلومة مع اتجاه مكاني واضح "
                            "(يمين/يسار/أمام/الخلف). لا تذكر نصوصًا مكتوبة أو أسماء علامات."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "صف المشهد بإيجاز شديد وبطريقة عملية."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        ],
                    },
                ],
            )

            buffer = ""
            SENT_END = ("؟", "!", ".")

            def _clip_words(s: str, n: int = 10) -> str:
                w = s.split()
                if len(w) <= n:
                    return s.strip()
                return (" ".join(w[:n]).rstrip("،,؛.")) + "."

            sent_out = False

            async for chunk in stream: 
                if not chunk.choices:
                    continue
                piece = getattr(chunk.choices[0].delta, "content", None)
                if not piece:
                    continue

                buffer += piece

                # If the first sentence ended, send it and stop streaming
                idx = max(buffer.rfind("؟"), buffer.rfind("!"), buffer.rfind("."))
                if idx != -1:
                    first = buffer[:idx + 1]
                    out = _clip_words(first, 9)  # 7–9 words for best TTS time
                    await websocket.send_text(json.dumps({"type": "partial", "text": out}))
                    sent_out = True
                    break

                # Safety: if model keeps going without punctuation, cut at ~10 words
                if len(buffer.split()) >= 10:
                    out = _clip_words(buffer, 10)
                    await websocket.send_text(json.dumps({"type": "partial", "text": out}))
                    sent_out = True
                    break

            # Finish
            # (We intentionally do NOT send any tail; one sentence only)
            await websocket.send_text(json.dumps({"type": "done"}))
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        finally:
            await websocket.close()