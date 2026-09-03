# Orange Pi / RK3588 Setup Record

This file is a reproducibility checklist. Fill the exact accepted values during
on-device testing; do not replace unknown versions with guesses.

## Target environment

| Item | Accepted value |
|---|---|
| Board model | pending confirmation |
| RAM | pending |
| OS image and kernel | pending |
| RKNPU driver | pending |
| `rknn_toolkit_lite2` | pending |
| Model-conversion toolkit | pending |
| Python | pending |
| NumPy | pending |
| OpenCV and build options | pending |
| Camera and capture mode | pending |
| Cooling/power configuration | pending |

## Installation order

1. Update the board using the procedure appropriate for the selected OS image.
2. Verify the RKNPU driver version before selecting an RKNN Lite wheel.
3. Install the matching Rockchip `rknn_toolkit_lite2` wheel.
4. Install NumPy and an OpenCV build appropriate for the capture/output path.
5. Place the three accepted RKNN files where `rknn_finall.py` currently expects
   them, or create local links from those names to externally stored models.
6. Record package versions and model SHA-256 values before testing.

If OpenCV was built with GStreamer or board-specific hardware codecs, do not
overwrite it with `opencv-python`. Confirm build capabilities with
`cv2.getBuildInformation()` and save the relevant video-I/O section in the test
record.

## First smoke run

Start with a short recorded clip and explicit output path:

```bash
python rknn_finall.py \
  --source "path/to/short-normal-wall.mp4" \
  --weight 65 \
  --output "results/short-normal-wall.mp4"
```

Verify model loading, one ID, aligned skeleton, path continuity, hold detection,
contact confirmation, `N/A` handling, clean shutdown, and output readability
before attempting a long run.

## Live run

```bash
python rknn_finall.py --source 0 --weight 65
```

Check capture resolution and reported camera FPS. Live scoring uses elapsed wall
time, so compare displayed FPS with the camera rate and note whether the camera
buffers or drops frames.

## Performance record

Record at minimum:

- input resolution and codec;
- output resolution and codec;
- detection and pose intervals;
- mean, median, p95, and minimum FPS;
- per-stage read, inference, draw, and write times;
- NPU call count per frame;
- CPU/NPU temperature and clock behaviour;
- memory use at start and after the soak run; and
- recovery-call frequency.

The current OpenCV writer uses `mp4v`. Evaluate a hardware-backed GStreamer/MPP
writer only as a separate comparison, preserving the accepted CPU-writer result
as the baseline.
