# Validation Plan

## Current state

The current RKNN integration candidate has passed source-level, synthetic
component, and synthetic execution-path checks. Those checks cannot validate
model accuracy, camera behaviour, Orange Pi runtime compatibility, thermals, or
real climbing scores.

The reported result of approximately 22 FPS came from earlier normal-wall
attempts. Re-measure it with the committed candidate and a recorded environment
description before treating it as a repository benchmark.

## Test order

### 1. Model-contract smoke test

- Confirm every RKNN model loads with no runtime warnings.
- Confirm the configured input dimensions.
- Inspect real output shapes and confidence ranges.
- Confirm person and hold class IDs.
- Compare several decoded boxes/keypoints with the source model.

Stop here if any contract is uncertain. Scoring tests are meaningless with a
misdecoded model.

### 2. Recorded normal-wall acceptance

Use short labelled clips covering:

- one unobstructed climber;
- wide and compact body positions;
- feet and wrists close to neighbouring holds;
- temporary pose loss;
- temporary person-detection loss;
- another person visible below the wall; and
- another person entering the wall region.

Review skeleton alignment, path continuity, ID contamination, contact events,
unknown-state transitions, and the four score components separately.

### 3. Recovery and identity stress

Measure identity contamination rather than only checking that the numeric ID
stays equal to one. Include full occlusion, exit/re-entry, crossings, similar
clothing, and a false-positive box near the last climber position.

### 4. Live-camera acceptance

Repeat the normal-wall cases using camera input. Check elapsed-time metrics,
capture buffering, preview responsiveness, clean quit, and behaviour both with
and without an output file.

### 5. Performance and soak

Run long enough to reveal thermal throttling and queue backpressure. Record the
environment and timing fields from `orange-pi-setup.md`. Repeat with a fixed clip
so future versions can be compared fairly.

### 6. Metric validation

Create timestamped expert labels for:

- supported bent-arm intervals;
- confirmed limb/hold contacts;
- releases and same-hold readjustments;
- distinct upward moves and supporting-leg extension; and
- pauses after the attempt begins.

Compare event detection first. Only judge the combined 0–100 score after the
underlying events and coverage rules are acceptable.

### 7. Load-estimate study

Test force and moment constraints automatically. Accuracy evaluation requires
instrumented contacts or another defensible reference measurement. Until then,
evaluate only internal consistency, ambiguity ranges, and correct refusal during
motion or missing contact evidence.

## Result template

| Category | Measurement | Acceptance threshold | Result |
|---|---|---|---|
| Person tracking | identity-contaminated frames | define before test | pending |
| Pose | labelled keypoint accuracy | define before test | pending |
| Contacts | precision / recall / timing error | define before test | pending |
| Path | false jump count | define before test | pending |
| Arm events | interval precision / recall | define before test | pending |
| Foot events | placement/readjustment accuracy | define before test | pending |
| Leg events | upward-episode accuracy | define before test | pending |
| Fluency | pause-duration error | define before test | pending |
| Performance | sustained FPS and p95 frame time | define before test | pending |
| Stability | errors, memory growth, thermal throttle | define before test | pending |

Define thresholds before examining final results. Keep raw reports, model hashes,
and environment metadata together outside Git when they contain large media.

## Release gate

An experimental release is ready when:

- recorded and live paths pass their agreed thresholds;
- no model contract remains implicit;
- score coverage and `N/A` behaviour are documented with real examples;
- the same fixed clip produces repeatable results;
- the setup can be reproduced from a clean target board; and
- known limitations are written in the release notes.
