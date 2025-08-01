import cv2, torch, time
from passive_mode.detector import load_model, detect_and_rank
from shared import tts                    
import shared.mode as sm                        
from shared.mode import mute_tts           
    

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", DEVICE)

model = load_model()

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        print("❌ Failed to capture frame.")
        break

    key = cv2.waitKey(1) & 0xFF

    
    # ----- ACTIVE MODE: 'a' to describe -----
    if key == ord('a') and not sm.active_in_flight["flag"]:
        tts.cancel()            # stop any current speech
        mute_tts(2.0)           # mute passive TTS for 2 s
        sm.run_active_once(frame)         # launches Active in background
        continue                # skip passive processing this frame


    # ----- PASSIVE MODE: detect and speak ----
    if not sm.active_in_flight["flag"]:
        frame, _ = detect_and_rank(frame, model,
                                   device=DEVICE, imgsz=512, conf=0.50)


        
    if key == ord('q'):
        print("Exiting…")
        break

    cv2.imshow("Blind Assist", frame)

cap.release()
cv2.destroyAllWindows()