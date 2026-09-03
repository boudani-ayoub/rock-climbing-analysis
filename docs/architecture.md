# Architecture

## Current boundary

`rknn_finall.py` is intentionally a single-file integration candidate. The
project is still measuring correctness and performance on RK3588, so preserving
one reviewable execution path is more valuable than splitting the code early.

The file contains four broad layers:

1. RKNN adapters for person, pose, and hold models.
2. Single-climber tracking, crop recovery, and pose caching.
3. Contact, motion, path, scoring, and load-estimation state.
4. Inline or threaded rendering and video output.

## Runtime data flow

```text
video/camera frame
        │
        ├── scheduled person detection ──> sticky slot ID 1 ──> box cache
        │                                      │
        │                                      ├── follow-crop recovery
        │                                      └── emergency-crop recovery
        │
        └── scheduled crop pose ─────────> pose cache and pose age
                                               │
                                               ├── skeleton and hip path
                                               ├── timed hold contacts
                                               ├── movement phase
                                               ├── coaching metrics
                                               └── optional static load estimate
```

Hold detection runs over the first configured frames. Its RKNN model is then
released before the person and pose models are loaded, reducing simultaneous NPU
runtime pressure.

## Scheduling

Person detection and pose use separate configurable intervals. With both
intervals set to two, detection normally runs on odd frames and pose on even
frames. The first frame primes both caches. This keeps the usual main-loop frame
to one NPU call while retaining frequent pose updates.

Cached poses may be shown briefly for visual continuity. Timed state and scores
are updated only from a newly inferred, valid pose. Recorded video uses source
frame time; live input uses a monotonic elapsed-time clock.

## Identity invariant

The detector exposes at most one slot. Once ID 1 is acquired, that identity is
not recycled during the attempt. Detection candidates must pass the wall filter
before initial assignment. Normal matching uses overlap and distance gates;
successful crop recovery explicitly synchronises the sticky slot.

This design avoids an ID-number change, but it cannot prove biometric identity.
Testing must still include another person entering the wall region, occlusion,
crossing, and long detection loss.

## Contact-state invariant

Current contacts and session history are separate:

- A current contact requires a confident limb keypoint near a detected hold for
  the confirmation duration.
- Release also has a time threshold to suppress jitter.
- Missing pose evidence eventually changes the current contact to unknown.
- The history of previously touched holds remains available for the final report.

Unknown observations are excluded from timing and scoring.

## Threading

NPU inference and all mutable tracking/scoring state remain on the main thread.
The draw worker receives immutable frame snapshots and performs overlay drawing
and video writing. The bounded queue prevents unbounded memory growth and applies
backpressure when encoding cannot keep up.

The current `mp4v` writer is still a CPU/OpenCV path. Hardware encoding should be
considered only after the target OpenCV/GStreamer stack is recorded and tested.

## Later package boundary

After acceptance testing, the monolith can be separated without changing its
behaviour:

```text
src/climbing_analysis/
├── config.py
├── rknn_models.py
├── tracking.py
├── contacts.py
├── metrics.py
├── overlays.py
└── pipeline.py
```

That refactor should happen in its own change after the validated single-file
version is tagged, with output comparisons against the tagged baseline.
