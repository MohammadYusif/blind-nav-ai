# import cv2, torch, time
# from passive_mode.detector import load_model, detect_and_rank
# from shared import tts                    
# import shared.mode as sm                        
# from shared.mode import mute_tts           
    

# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# print("Device:", DEVICE)

# model = load_model()

# cap = cv2.VideoCapture(0)

# while True:
#     ok, frame = cap.read()
#     if not ok:
#         print("❌ Failed to capture frame.")
#         break

#     key = cv2.waitKey(1) & 0xFF

    
#     # ----- ACTIVE MODE: 'a' to describe -----
#     if key == ord('a') and not sm.active_in_flight["flag"]:
#         tts.cancel()            # stop any current speech
#         mute_tts(2.0)           # mute passive TTS for 2 s
#         sm.run_active_once(frame)         # launches Active in background
#         continue                # skip passive processing this frame


#     # ----- PASSIVE MODE: detect and speak ----
#     if not sm.active_in_flight["flag"]:
#         frame, _ = detect_and_rank(frame, model,
#                                    device=DEVICE, imgsz=512, conf=0.50)


        
#     if key == ord('q'):
#         print("Exiting…")
#         break

#     cv2.imshow("Blind Assist", frame)

# cap.release()
# cv2.destroyAllWindows()


# main.py
import cv2, torch, time, queue

from passive_mode.detector import load_model, detect_and_rank
from shared import tts                              # unified streaming TTS
import shared.mode as sm                            # active_in_flight, run_active_once, mute_tts
from shared.mode import mute_tts
from shared.stt_cmds import start_cmd_listener      # STT commands: ACTIVE | PASSIVE | STOP

def now():
    return time.strftime("%H:%M:%S")

# Device & model
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[{now()}] Device: {DEVICE}")

model = load_model()
tts.start()   # start TTS worker once
print(f"[{now()}] TTS worker started")

# Start STT command listener
cmd_q: "queue.Queue[str]" = queue.Queue(maxsize=2)
start_cmd_listener(cmd_q)     # ← no language arg
print(f"[{now()}] STT command listener running (en)")

# Camera loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(f"[{now()}] ERROR: Cannot open camera.")
    raise SystemExit(1)

print(f"[{now()}] Loop started — say 'active' / press 'a', 'q' to quit.")

prev_active = False

while True:
    ok, frame = cap.read()
    if not ok:
        print(f"[{now()}] ERROR: Failed to capture frame.")
        break

    # Poll STT command (non-blocking)
    try:
        cmd = cmd_q.get_nowait()
        print(f"[{now()}] Voice CMD: {cmd}")   # <- prints ACTIVE | PASSIVE | STOP
    except queue.Empty:
        cmd = None

    # Keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print(f"[{now()}] Exiting…")
        break
    if key == ord('a'):
        print(f"[{now()}] Key: 'a' → request ACTIVE")

    # Commands / Keys → Active / Stop
    if (cmd == "ACTIVE" or key == ord('a')) and not sm.active_in_flight["flag"]:
        print(f"[{now()}] tts.cancel()")
        tts.cancel()
        print(f"[{now()}] mute_tts(2.0)")
        mute_tts(2.0)
        print(f"[{now()}] Launching Active (snapshot → WS → speak)")
        sm.run_active_once(frame)
        # Skip passive on this frame
        continue

    if cmd == "STOP":
        print(f"[{now()}] STOP → tts.cancel()")
        tts.cancel()

    # Mode status change logs
    now_active = sm.active_in_flight["flag"]
    if now_active != prev_active:
        if now_active:
            print(f"[{now()}] ACTIVE started")
        else:
            print(f"[{now()}] ACTIVE finished → back to PASSIVE")
        prev_active = now_active

    # Overlay
    mode_label = "ACTIVE…" if now_active else "PASSIVE"
    cv2.putText(frame, f"MODE: {mode_label}  (say 'active' / press 'a', 'q' quit)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Passive detection (only when not in Active)
    if not now_active:
        frame, _ = detect_and_rank(frame, model, device=DEVICE, imgsz=512, conf=0.50)

    cv2.imshow("Blind Assist", frame)

cap.release()
cv2.destroyAllWindows()
print(f"[{now()}] Camera released. Bye.")