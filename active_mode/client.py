# client.py
import cv2
import asyncio
import websockets
import threading
import json
import time

from shared import tts

def clip_words(text: str, max_words: int = 8) -> str:
    w = text.split()
    if len(w) <= max_words:
        return text
    short = " ".join(w[:max_words]).rstrip("،,؛.")
    return short + "."


async def ws_send(uri, image_bytes):
    t0 = time.perf_counter()
    tts.set_ref_time(t0)

    # NEW: gate audio to exactly two chunks
    spoke_first = False
    last_partial = None

    try:
        async with websockets.connect(uri, max_size=2 * 1024 * 1024) as websocket:
            await websocket.send(image_bytes)
            print("✅ Image sent to server")

            first_partial_time = None

            while True:
                msg = await websocket.recv()
                data = json.loads(msg)

                if data.get("type") == "partial":
                    text = (data.get("text") or "").strip()
                    if not text:
                        continue

                    if first_partial_time is None:
                        first_partial_time = time.perf_counter()
                        print(f"⏱ first-partial {first_partial_time - t0:.3f}s")
                    print("🟢 partial:", text)

                    # 👉 Speak ONLY the first partial now; keep overwriting the tail
                    if not spoke_first:
                        tts.enqueue(clip_words(text, 8))
                        spoke_first = True
                    else:
                        last_partial = text

                elif data.get("type") == "done":
                    # 👉 Speak only the final tail once
                    if last_partial:
                        tts.enqueue(last_partial)
                    t_done = time.perf_counter()
                    print(f"✅ done @ {t_done - t0:.3f}s")
                    break

                elif data.get("type") == "error":
                    print("❌ server error:", data.get("message"))
                    break

    except Exception as e:
        print("❌ WebSocket Error:", e)

def send_image_thread(uri, image_bytes, flag_holder):
    try:
        asyncio.run(ws_send(uri, image_bytes))
    finally:
        flag_holder["in_flight"] = False


# def main():
#     uri = "ws://127.0.0.1:8000/ws"
#     cap = cv2.VideoCapture(0)
#     tts.start()

#     print("📷 Webcam started — Press SPACE to capture or Q to quit.")
#     state = {"in_flight": False}

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             print("❌ Failed to capture frame.")
#             break

#         cv2.imshow("Webcam", frame)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord(' '):
#             # Fresh capture: stop any ongoing speech to keep it live
#             tts.cancel()

#             if state["in_flight"]:
#                 # Optional: ignore repeated presses while one is running
#                 continue

#             # JPEG compress on the client to reduce upload size
#             _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
#             image_bytes = buf.tobytes()

#             state["in_flight"] = True
#             threading.Thread(
#                 target=send_image_thread,
#                 args=(uri, image_bytes, state),
#                 daemon=True
#             ).start()

#         elif key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("👋 Webcam closed.")


# if __name__ == "__main__":
#     main()