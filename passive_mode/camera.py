import cv2
import torch
from detector import load_model, detect_and_rank

# DEVICE MPS FOR MY MAC
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", DEVICE)

model = load_model() 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to capture frame.")
        break

    frame, summary = detect_and_rank(frame, model, device=DEVICE, imgsz=512, conf=0.50)
    if summary:
        print("Top-3 risky objects:", summary)

    cv2.imshow("Blind Assist - Passive Mode", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()