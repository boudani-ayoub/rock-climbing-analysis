# 🧗 Climbing AI — Real-Time Climbing Analysis on RK3588

A real-time climbing analysis system built for the **Orange Pi 5 Plus (RK3588)** edge device. The system tracks climbers, detects holds, and provides live biomechanical feedback using computer vision — all running locally with no cloud dependency.

---

## Overview

This project combines pose estimation, multi-object tracking, and custom hold detection into a unified real-time pipeline. It was designed with edge deployment in mind from the start, targeting the RK3588 NPU via RKNN.

---

## Features

- **Climber tracking** — YOLO11m-pose + ByteTrack for stable per-climber IDs across frames
- **Wall isolation** — automatic wall region filter to ignore background movement
- **Auto-zoom crop** — dynamically crops around the active climber
- **Path trail** — visualizes the climber's movement history on screen
- **Triangle foot support** — detects when the foot is properly planted on a hold
- **Lock arm warning** — flags when the arm is in a locked/overextended position
- **Force bar** — estimates and displays load distribution in real time
- **Hold detection** — custom-trained YOLO11n model (mAP@50 = 0.85), runs once at session start and is cached since holds don't move
- **Grip confirmation** — requires 8 consecutive frames before confirming a grip to prevent false positives

---

## Project Structure

```
├── phase1_skeleton.py        # Pose estimation + ByteTrack IDs + wall filter + auto-zoom
├── phase2_features.py        # Reads keypoint JSON → path trail, triangle support, lock arm, force bar
├── phase3_train.py           # Fine-tuning YOLO11n hold detector
├── phase3_validate.py        # Hold detection + 8-frame grip confirmation
├── main.py                   # Unified real-time script — combines all phases
├── bytetrack_climbing.yaml   # Custom ByteTrack config (90-frame buffer, stable IDs)
└── main_rk3588.py            # (Phase 4) RK3588 deployment using RKNNLite API
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Pose estimation | YOLO11m-pose (dev) / YOLO11n-pose (RK3588) |
| Object tracking | ByteTrack |
| Hold detection | YOLO11n (custom fine-tuned) |
| Video processing | OpenCV |
| Language | Python |
| Target hardware | Orange Pi 5 Plus — Rockchip RK3588 |

---

## Key Design Decisions

**Pose model split** — YOLO11m is used during development for accuracy. YOLO11n is used for RK3588 deployment to fit within the NPU constraints.

**Hold detection is cached** — holds are detected once at the start of each session and stored. Since holds don't move during a climb, re-running detection every frame would waste compute.

**Grip confirmation uses 8 frames** — a single-frame detection is not enough to confirm a real grip. Requiring 8 consecutive frames eliminates false positives caused by hand passing near a hold.

**Target selection** — currently set to `largest` bounding box when multiple climbers are present. This can be changed based on use case.

---

## Remaining Work (Phase 4 — Deployment)

- Flash Ubuntu on the Orange Pi 5 Plus (requires MicroSD)
- Convert models to RKNN format:
  ```bash
  yolo export format=rknn name=rk3588
  ```
- Write `main_rk3588.py` using RKNNLite API instead of Ultralytics
- Transfer files via SSH and test on device

---

## Getting Started (Development — PC/GPU)

```bash
pip install ultralytics opencv-python supervision
python main.py
```

For RK3588 deployment, follow the [RKNN Toolkit 2 documentation](https://github.com/rockchip-linux/rknn-toolkit2).

---

## Hardware Target

**Orange Pi 5 Plus** powered by the Rockchip **RK3588** — a 6 TOPS NPU capable of running quantized YOLO models in real time at the edge.

---

## Author

Ayoub Boudani
