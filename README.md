# Rock Climbing Analysis

Experimental computer-vision pipeline for analysing one climber on an indoor
wall. The current target is an Orange Pi 5-class board with the RK3588 NPU.

> **Project status:** [`rknn_finall.py`](rknn_finall.py) is the current
> integration candidate. Its components and execution paths have passed
> synthetic checks, but the complete file still needs recorded-video and live
> camera acceptance testing on the target board. Treat all technique and load
> outputs as experimental until that validation is complete.

## What the current pipeline does

- Detects and follows exactly one climber with a sticky ID that is not recycled
  during an attempt.
- Runs cropped 17-keypoint pose estimation and draws the skeleton.
- Preserves a jump-safe hip path and total path length.
- Detects climbing holds, confirms timed hand/foot contacts, and retains a
  separate history of holds touched during the session.
- Uses follow-crop and emergency-crop recovery when full-frame detection misses
  the climber.
- Produces four coaching metrics: arm technique, support/triangle, footwork and
  leg use, and fluency.
- Optionally displays a constrained 2D estimate of vertical support distribution.
- Can overlap NPU inference with drawing and video output through worker threads.

The technique number is a project-specific coaching score. It is not an IFSC
competition score and should not be presented as one.

## Coaching metrics

| Metric | Current interpretation |
|---|---|
| Arm technique | Sustained bent-arm time while the wrist is confirmed on a hold and the climber is static. A short grace period avoids penalising normal transitions. |
| Support / triangle | Confirmed support contacts, support-polygon area, and the estimated 2D centre of mass relative to that geometry. |
| Footwork / legs | Rapid same-hold foot readjustments plus leg extension during distinct upward movement episodes. |
| Fluency | Time spent in pauses longer than the configured grace period after the attempt begins. |

The overall score is the equal average of these four metrics only after every
metric has enough observations. Missing evidence remains `N/A`; it is not
silently replaced with a neutral score.

Load distribution is deliberately excluded from the score. The camera cannot
measure individual contact forces. The displayed result is a 2D quasi-static
vertical-support estimate with feasible ranges and an ambiguity label. See
[`docs/scoring.md`](docs/scoring.md) for the complete definitions and limits.

## Repository layout

```text
.
├── .github/workflows/cpu-checks.yml
├── .gitattributes
├── .gitignore
├── rknn_finall.py              # current single-file RK3588 candidate
├── README.md
├── requirements-rk3588.txt
├── requirements-dev.txt
├── docs/
│   ├── architecture.md
│   ├── model-conversion.md
│   ├── orange-pi-setup.md
│   ├── scoring.md
│   └── validation.md
├── models/
│   └── README.md               # model contracts; model binaries are untracked
├── scripts/
│   └── README.md               # planned benchmark and validation utilities
├── tests/
│   ├── README.md               # test scope and next-phase plan
│   └── test_core_logic.py      # CPU-only invariant checks
└── legacy/
    └── onnx-prototypes/        # unchanged historical CPU/ONNX snapshots
```

The active pipeline remains one file while hardware behaviour is being
stabilised. Splitting it into packages is intentionally deferred until after
acceptance testing, so a refactor cannot hide performance or tracking changes.

## Required models

The current constants expect these files in the working directory:

| File | Role | Required contract |
|---|---|---|
| `yolo11m.rknn` | person detection | COCO class `0` is person; confidence output contract must match the configured flag |
| `yolo11m-pose.rknn` | 17-keypoint pose | input must match `POSE_INPUT_SIZE`, currently 512×512 |
| `best.rknn` | holds and volumes | class `0` is grip, class `1` is volume |

Model binaries, calibration data, videos, and generated results are ignored by
Git. Record conversion details and hashes locally using the checklist in
[`docs/model-conversion.md`](docs/model-conversion.md).

## Environment

The RKNN Lite runtime must match the RK3588 driver and the toolkit version used
to build the models. Rockchip distributes it as a platform-specific wheel, so it
is documented separately rather than pretending that a generic PyPI dependency
is sufficient.

```bash
python -m pip install -r requirements-rk3588.txt
# Then install the matching rknn_toolkit_lite2 wheel for the board.
```

If OpenCV was built locally for GStreamer or hardware encoding, keep that build
instead of replacing it with the PyPI wheel. Record the final versions in
[`docs/orange-pi-setup.md`](docs/orange-pi-setup.md).

## Usage

Run commands from the repository root so the current model paths resolve.

```bash
# Recorded video
python rknn_finall.py --source "path/to/climb.mp4" --weight 65 --output result.mp4

# Live camera; omit --output for a preview window
python rknn_finall.py --source 0 --weight 65

# Technique analysis without load estimation
python rknn_finall.py --source "path/to/climb.mp4" --output result.mp4
```

| Argument | Meaning |
|---|---|
| `--source`, `-s` | Video path or camera index; default is camera `0`. |
| `--weight`, `-w` | Body mass in kilograms. Optional; enables only the load display. |
| `--output`, `-o` | Annotated video path. A video input defaults to `results/<input-name>`. |

## Performance status

Previous normal-wall attempts reached approximately **22 FPS** on the RK3588
setup. That number is a working observation, not a reproducible benchmark for
this final candidate. Detection and pose use alternating schedules by default;
at 22 live frames per second, pose evidence arrives at roughly 11 Hz.

This is generally sufficient for sustained posture and normal climbing
movements. Fast dynamic contacts may require a higher pose sampling rate. The
acceptance run must record resolution, model hashes, temperatures, encoder,
stage timings, output FPS, and pose sampling rate before a performance claim is
published.

## What remains before a release

- Validate recorded-video and live-camera paths on the Orange Pi.
- Confirm all input shapes and confidence-output contracts against the actual
  RKNN models.
- Build a labelled clip set for identity, pose, contact, and metric evaluation.
- Measure repeatability across multiple runs of the same climb.
- Compare load estimates with instrumented holds before making accuracy claims.
- Extend the CPU test suite and add the benchmark tools described in `scripts/`.
- Pin the exact versions that pass acceptance testing.
- Decide on a project licence before accepting external reuse or contributions.
- Tag an experimental release only after the validation checklist passes.

See [`docs/validation.md`](docs/validation.md) for the planned acceptance record.

## Historical implementations

The original CPU/ONNX development stages are preserved unchanged in
[`legacy/onnx-prototypes/`](legacy/onnx-prototypes/). They are reference
snapshots, not supported entry points. Git history preserves them, so they can
be removed from the default branch after the RKNN implementation has a validated
release.
