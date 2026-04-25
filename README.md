# Rock Climbing Analysis

Real-time AI-powered analysis system for rock climbing using computer vision and pose estimation. The system tracks climbers, evaluates technique, and provides live feedback and a detailed score report at the end of each climb.

---

## Features

### Detection & Tracking
- Real-time climber detection and multi-person tracking using YOLOv11 and ByteTrack
- Two-phase detection for tall walls:
  - **Phase 1** — full-frame detection to identify new climbers
  - **Phase 2** — follow crop that tracks each climber as they climb higher, keeping them large in the crop regardless of wall height
- Hold detection using a custom-trained YOLO model (grip holds and volume holds)
- Grip tracking — detects which holds each climber's hands and feet are touching
- Emergency zoom fallback for temporarily lost climbers

### Scoring Metrics
All scores are calculated per-frame and averaged across the full climb.

**Triangle Stability**
Evaluates the triangle formed by the climber's two wrists and one foot. Scored based on:
- Height ratio: how high the wrists are above the foot relative to body height
- Spread ratio: how wide the wrists are spread relative to body width

**Arm Technique**
Measures the percentage of frames where at least one arm is straight. Straight arms transfer load to the skeleton rather than muscles, reducing fatigue.

**Limb Load Distribution**
Estimates the load (kg) on each active contact point using inverse distance weighting from the center of gravity. Only limbs confirmed on holds are included.
- Left Hand / Right Hand
- Left Foot / Right Foot

**Overall Score**
Average of all 6 metrics: Triangle, Arm Technique, Left Hand, Right Hand, Left Foot, Right Foot.

### Live Feedback
- Per-climber stats panel showing triangle status, arm status, and live load per limb
- Live score badge with 6 metric bars updated every frame
- Red border alert when both arms are bent simultaneously for more than 3 seconds
- Colored path trail per climber showing movement history
- Hold visualization: idle / hand-touched / foot-touched / both

### Final Score Card
Displayed at the end of the video output:
- Diamond radar chart (4 axes: LH, RH, LF, RF) with limb score in center
- 6 metric bars: Triangle, Arm tech, Left Hand, Right Hand, Left Foot, Right Foot
- Total score line at the bottom

---

## Models

| Model | Purpose |
|---|---|
| `yolo11m.onnx` | Person detection and tracking |
| `yolo11m-pose.onnx` | Pose estimation (17 keypoints) |
| `runs/hold_detection/weights/best.onnx` | Hold detection (custom trained) |

ONNX models are required for inference. PyTorch `.pt` fallbacks are supported if ONNX files are not found.

---

## Requirements

```
pip install ultralytics opencv-python numpy onnxruntime
```

---

## Usage

```bash
# Run on a video file
python main_crop4.py --source "your_video.mp4" --weight 65 --output result.mp4

# Run on live camera
python main_crop4.py --weight 65

# Arguments
--source   Path to video file or camera index (default: 0)
--weight   Climber body weight in kg (required for load analysis)
--output   Path to save annotated output video (optional)
```

---

## Script Versions

| Script | Description |
|---|---|
| `main2.py` | Original base — detection, pose, path tracking |
| `main_crop1.py` | Adds emergency zoom fallback for lost climbers |
| `main_crop2.py` | Adds Phase 2 follow zoom for tall walls |
| `main_crop3.py` | Adds normalized triangle and arm tech scoring |
| `main_crop4.py` | Adds limb load analysis, diamond radar chart, red border alert |

---

## Hardware

Designed for deployment on **Orange Pi 5 Plus** with RK3588 NPU using RKNN models for real-time inference at 25-30 FPS. Development and testing on CPU.