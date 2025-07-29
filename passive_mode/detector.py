# detector.py
import cv2
from ultralytics import YOLO
import time


try:
    from speak import speak_submit
    _TTS_OK = True
except Exception:
    _TTS_OK = False

TTS_COOLDOWN = 1.5     # speak at most ~every 1.5s
TTS_REPEAT_SAME = 4.0  # allow same phrase again after 4s
_last_tts_time = 0.0
_last_tts_text = ""
_last_spoken = "" # to avoid repeat


# Arabic maps 
_AR_LABEL = {
    "person":"شخص","chair":"كرسي","table":"طاولة","dining table":"طاولة",
    "car":"سيارة","bicycle":"دراجة","motorcycle":"دراجة نارية","sofa":"كنبة",
    "bed":"سرير","potted plant":"نبتة","suitcase":"حقيبة سفر","backpack":"حقيبة ظهر",
    "handbag":"حقيبة","bottle":"قارورة","refrigerator":"ثلاجة","bench":"مقعد", "tv":"تلفاز",
}
_AR_SIDE  = {"center":"قدامك","left":"يسارك","right":"يمينك"}
_AR_LEVEL = {"near":"قريب","medium":"متوسط","far":"بعيد"}

OBSTACLE = {
    'person','chair','table','bench','bicycle','motorcycle','car','sofa','bed',
    'potted plant','suitcase','backpack','handbag','bottle','dining table','table', "refrigerator","tv"
}

SIDE_COLOR = {
    "left":   (255, 0, 0),    # blue (BGR)
    "center": (0, 255, 255),  # yellow
    "right":  (0, 0, 255),    # red
}

FLOOR_Y = 0.60
CENTER_WIN   = (0.35, 0.65)   # wider center corridor
LEFT_MAX     = 0.45
RIGHT_MIN    = 0.55
CENTER_FRAC  = 0.40
MID_TOL      = 0.10           # midpoint within ±10% => center
PATH_WEIGHT  = 0.35           # if someone walks in the center we gonna increase the score (e,g: 0.35+)

def load_model():
    return YOLO("yolov8n.pt")

def _frac_overlap(a1, a2, b1, b2): # fraction of overlap between two ranges [a1, a2] and [b1, b2]
    ov = max(0.0, min(a2, b2) - max(a1, b1))
    return ov / max(1e-6, (a2 - a1))

def _side_by_zones(x1, x2, w):
    fx1, fx2 = x1 / w, x2 / w
    xc = (fx1 + fx2) / 2.0
    cmin, cmax = CENTER_WIN

    # 1) midpoint rule
    if abs(xc - 0.5) <= MID_TOL:
        return "center"

    # 2) center window overlap
    cfrac = _frac_overlap(fx1, fx2, cmin, cmax)
    if cfrac >= CENTER_FRAC:
        return "center"

    # 3) side bands (mirrored mapping per your webcam)
    lfrac = _frac_overlap(fx1, fx2, 0.0, LEFT_MAX)
    rfrac = _frac_overlap(fx1, fx2, RIGHT_MIN, 1.0)
    return "right" if lfrac >= rfrac else "left"

def _speak_top3(summary_text: str):
    """
    Improved TTS:
    - Combines multiple people in same side & level: "قدامك شخصين على مسافة قريبة"
    - Handles up to 2-3 items for clarity
    """
    global _last_tts_time, _last_tts_text
    if not _TTS_OK or not summary_text:
        return

    now = time.time()
    if summary_text == _last_tts_text and (now - _last_tts_time) < TTS_REPEAT_SAME:
        return
    if (now - _last_tts_time) < TTS_COOLDOWN:
        return

    # Parse items: label:level:side
    parsed = []
    for p in [s.strip() for s in summary_text.split(",") if s.strip()]:
        try:
            lab, lvl, side = [t.strip() for t in p.split(":")]
            parsed.append((lab, lvl, side))
        except ValueError:
            continue

    if not parsed:
        return

    # Prefer near -> medium; if none, allow one far
    chosen = [it for it in parsed if it[1] in ("near", "medium")]
    if not chosen:
        chosen = [parsed[0]]

    # Limit to top 3 items for speech clarity
    chosen = chosen[:3]

    # --- Smart grouping for people ---
    people = [c for c in chosen if c[0] == "person"]
    others = [c for c in chosen if c[0] != "person"]

    phrases = []
    if len(people) >= 2:
        # Try to group if same level and same side
        level_same = all(p[1] == people[0][1] for p in people)
        side_same  = all(p[2] == people[0][2] for p in people)
        count = len(people)
        num_ar = {2: "شخصين", 3: "٣ أشخاص"}

        if level_same and side_same:
            side_ar  = _AR_SIDE.get(people[0][2], people[0][2])
            level_ar = _AR_LEVEL.get(people[0][1], people[0][1])
            count_ar = num_ar.get(count, f"{count} أشخاص")
            phrases.append(f"{side_ar} {count_ar} على مسافة {level_ar}")
        else:
            # If different sides or levels, list individually
            for lab, lvl, side in people:
                side_ar  = _AR_SIDE.get(side, side)
                level_ar = _AR_LEVEL.get(lvl, lvl)
                name_ar  = _AR_LABEL.get(lab, lab)
                phrases.append(f"{side_ar} {name_ar} على مسافة {level_ar}")
    else:
        # Single person if any
        for lab, lvl, side in people:
            side_ar  = _AR_SIDE.get(side, side)
            level_ar = _AR_LEVEL.get(lvl, lvl)
            name_ar  = _AR_LABEL.get(lab, lab)
            phrases.append(f"{side_ar} {name_ar} على مسافة {level_ar}")

    # --- Add other obstacles (chair, table...) ---
    for lab, lvl, side in others[:2-len(phrases)]:
        side_ar  = _AR_SIDE.get(side, side)
        level_ar = _AR_LEVEL.get(lvl, lvl)
        name_ar  = _AR_LABEL.get(lab, lab)
        phrases.append(f"{side_ar} {name_ar} على مسافة {level_ar}")

    text = "، و".join(phrases)

    try:
        speak_submit(text)
    except Exception:
        pass

    _last_tts_text = summary_text
    _last_tts_time = now

def detect_and_rank(frame, model, device="cpu", conf=0.30, imgsz=640):
    global _last_spoken
    results = model(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    

    h, w = frame.shape[:2]
    rank = []  # (rscore, (x1,y1,x2,y2), label, side, level)

    # ── guide lines  ──
    mid_x = int(0.5 * w); lx = int(LEFT_MAX * w); rx = int(RIGHT_MIN * w)
    c1 = int(CENTER_WIN[0] * w); c2 = int(CENTER_WIN[1] * w); fy = int(FLOOR_Y * h)
    cv2.line(frame, (mid_x,0), (mid_x,h), (0,255,255),1)
    cv2.line(frame, (lx,0), (lx,h), (255,0,0),1); cv2.line(frame, (rx,0), (rx,h), (255,0,0),1)
    cv2.line(frame, (c1,0), (c1,h), (0,165,255),1); cv2.line(frame, (c2,0), (c2,h), (0,165,255),1)
    cv2.line(frame, (0,fy), (w,fy), (255,0,255),1)
    # ─────────────────────────────

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item()); confv = float(box.conf[0].item())
            label  = model.names.get(cls_id, str(cls_id))

            # bbox midpoint 
            x_mid = int((x1 + x2) / 2); cv2.line(frame, (x_mid,0), (x_mid,h), (255,255,255),1)

            # draw detection (green)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {confv:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if label in OBSTACLE:
                bottom = y2 / h
                if bottom < FLOOR_Y:
                    continue
                height_r = (y2 - y1) / h
                floor_norm = (bottom - FLOOR_Y) / (1.0 - FLOOR_Y + 1e-6)
                floor_norm = max(0.0, min(1.0, floor_norm))
                dscore = 0.7 * height_r + 0.3 * floor_norm  # 0..1

                # distance bins
                if dscore >= 0.70:   level = "near"
                elif dscore >= 0.40: level = "medium"
                else:                level = "far"

                side = _side_by_zones(x1, x2, w)

                # center corridor bias for ranking
                fx1, fx2 = x1 / w, x2 / w
                path_overlap = _frac_overlap(fx1, fx2, CENTER_WIN[0], CENTER_WIN[1])  # 0..1
                center_bonus = 0.05 if side == "center" else 0.0
                rscore = (1.0 - PATH_WEIGHT) * dscore + PATH_WEIGHT * (path_overlap + center_bonus)
                rscore = max(0.0, min(1.0, rscore))

                rank.append((rscore, (x1,y1,x2,y2), label, side, level))

    # =====  order near -> medium -> far, within each by rscore desc =====
    near   = [r for r in rank if r[4] == "near"]
    medium = [r for r in rank if r[4] == "medium"]
    far    = [r for r in rank if r[4] == "far"]
    near.sort(key=lambda t: t[0], reverse=True)
    medium.sort(key=lambda t: t[0], reverse=True)
    far.sort(key=lambda t: t[0], reverse=True)
    ordered = near + medium + far
    top = ordered[:3]

    # draw + summary
    summary = []

    # best 'person' to show YOUR side (by rscore)
    best_person = None
    for rscore, (x1,y1,x2,y2), label, side, level in ordered:
        if label == "person" and (best_person is None or rscore > best_person[0]):
            best_person = (rscore, side)

    for rscore, (x1,y1,x2,y2), label, side, level in top:
        color = SIDE_COLOR.get(side, (0, 0, 255))
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"⚠ {label} {level} {side}", (x1, y1-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        summary.append(f"{label}:{level}:{side}")

    # Top-right panel (now explicitly near->far)
    if top:
        panel_w = 280
        panel_h = 24 + 22 * len(top)
        x0 = max(0, w - panel_w - 10); y0 = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        cv2.putText(frame, "Top-3 (near -> far)", (x0 + 10, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        for i, (rscore, (x1,y1,x2,y2), label, side, level) in enumerate(top):
            y = y0 + 18 + 22 * (i + 1)
            col = SIDE_COLOR.get(side, (255, 255, 255))
            cv2.circle(frame, (x0 + 12, y - 6), 5, col, -1)
            # show level first as you asked
            cv2.putText(frame, f"{i+1}. {level}  {label} {side}",
                        (x0 + 26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if best_person is not None:
        _, you_side = best_person
        ycolor = SIDE_COLOR.get(you_side, (0, 0, 255))
        cv2.putText(frame, f"YOU: {you_side.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, ycolor, 3)
        
    summary_text = ", ".join(summary)

    # Only speak if there's at least one near/medium, and avoid repeating same phrase
    if any(("near" in s) or ("medium" in s) for s in summary):
        if summary_text != _last_spoken:
            _speak_top3(summary_text)
            _last_spoken = summary_text

    return frame, summary_text