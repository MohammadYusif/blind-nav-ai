# blind-nav-ai

**Arabic-speaking AI assistant for blind navigation using zero-shot vision-language models.**

## Repository Structure & Branches

* `main` — Stable integration point; holds scaffold and merged features.
* `ml-quick` — Quick track with real-time, low-latency models:

  * **PyDNet** for depth estimation (\~60 FPS)
  * **YOLOv8-Nano** for object detection
  * **BLIP-2 (tiny)** for on-device captioning
* `ml-best-base` — Best track with high-accuracy models:

  * **GroundingDINO** for open-vocabulary grounding
  * **MiDaS-small** for depth estimation (\~20–30 FPS)
  * **BLIP-2 (tiny)** for rich scene descriptions

## Usage Modes & Expected Output

### 🔹 Quick Mode (`ml-quick` branch)

**Pipeline:** PyDNet → YOLOv8-Nano → BLIP-2 (tiny) → Template-based TTS

**Visual Overlay:**

* Bounding boxes around detected objects
* Distance labels (e.g., “2.1 m”) beside each box

**Spoken Alerts:**
Succinct, templated sentences (up to 5 Hz). Example:

```
[Beep] “Chair one point five meters ahead.”
[Beep] “Person two point three meters ahead.”
```

### 🔹 Best-Base Mode (`ml-best-base` branch)

**Pipeline:** GroundingDINO → MiDaS-small → BLIP-2 (tiny) → Free-form planning → TTS

**Visual Overlay:**

* Precise region labels (open-vocabulary)
* Distance heatmap or per-object distance tags

**Spoken Narration:**
Rich, free-form descriptions. Example:

```
[Beep] “There’s a wooden chair one point five meters in front of you, on your left.”
[Beep] “A person is walking about two point three meters ahead.”
```

## Getting Started

1. Clone the repo and checkout your track:

   ```bash
   git clone https://github.com/MohammadYusif/blind-nav-ai.git
   # Quick mode
   git checkout ml-quick
   # Best-base mode
   git checkout ml-best-base
   ```
2. Follow the branch-specific README in `/ml-quick` or `/ml-best-base` for setup and demo instructions.

---
