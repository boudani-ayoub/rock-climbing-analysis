# Scoring Model

## Purpose

The output is a coaching-oriented technique summary derived from 2D pose and
hold-contact evidence. It is not an official competition result, a medical
assessment, or a direct force measurement.

All time accumulation uses source-video time for files and monotonic elapsed time
for live input. A cached pose can be drawn but never adds scoring evidence.

## Overall score

The current overall score is the equal average of four pillars:

```text
overall = (arm + support + footwork + fluency) / 4
```

The overall value remains `N/A` until every pillar has enough coverage. This
prevents missing evidence from turning into an invented neutral score.

## Arm technique

An arm becomes eligible only when:

- its shoulder, elbow, and wrist angle is observable;
- that wrist has a confirmed hold contact; and
- the movement phase is static.

The arm is considered bent below the configured 150-degree threshold. A
continuous bend receives a one-second grace period, after which bent time adds a
penalty. The score is:

```text
100 × (1 - penalised eligible time / total eligible time)
```

At least one second of eligible arm-contact time is required. This metric should
be described as arm economy, not as a universal rule that every bent arm is bad.

## Support / triangle

Only confirmed wrist and ankle contacts participate. Unknown or estimated limb
locations do not create supports.

- Fewer than two contacts receive zero support quality.
- Two contacts receive partial quality of 0.5.
- Three or four contacts form a convex support polygon in the image plane.

For a polygon, quality combines a base value with normalised polygon area and
the estimated centre of mass relative to the polygon:

```text
quality = 0.4 + 0.3 × area_factor + 0.3 × com_factor
```

The area factor reaches one at a polygon area of eight percent of squared body
height. The centre-of-mass factor is one inside the polygon and falls with
distance outside it, reaching zero at 20 percent of body height.

The score is the time-weighted average of this quality during known static
support observations. At least one second is required. This is a front-camera
2D stability proxy; wall-normal depth and contact direction are unobserved.

## Footwork and leg use

This pillar can contain two submetrics:

1. **Foot placement control.** Confirmed foot acquisitions are counted after
   each foot's initial contact. Reacquiring the same hold within 1.2 seconds of
   release is treated as a readjustment. At least two placements are required.
2. **Leg contribution.** A distinct upward movement with a confirmed supporting
   foot is one opportunity. Sufficient supporting-knee extension during that
   episode counts as a leg drive. At least three opportunities are required.

If both submetrics have coverage, their mean is the pillar value. If only one
has coverage, that observed submetric is used. A leg-drive episode is counted
once rather than once per pose sample.

## Fluency

The attempt begins after the centre of mass has moved enough from its initial
position. Known static and moving observations then add to observed time. Static
pauses have a 2.5-second grace period; only the excess adds a penalty:

```text
100 × (1 - excess pause time / observed attempt time)
```

At least five seconds of observed attempt time is required.

## Geometric entropy

Geometric entropy of the smoothed centre-of-mass path is retained as a raw
diagnostic. Teleport-size path jumps are excluded, and only the longest
continuous path segment is used. It is not mapped into the overall score because
the project has not validated a population-specific 0–100 calibration.

## Vertical-support estimate

When body mass is provided, the optional load panel runs only for a static pose
with at least two confirmed contacts and adequate centre-of-mass coverage. It
enforces:

- non-negative vertical reactions;
- total vertical reaction equal to body weight; and
- zero projected moment about the estimated 2D centre of mass.

Multiple force distributions can satisfy those constraints. The panel therefore
shows a selected temporally smooth solution, each contact's feasible percentage
range, and whether the solution is constrained or ambiguous. The values must not
be called measured kilograms or used as a score.

## Validation still required

- Expert-labelled arm, contact, placement, pause, and leg-drive events.
- Repeatability across runs and camera positions.
- Sensitivity analysis for every threshold.
- Subgroup checks for body proportions and climbing styles.
- Instrumented-contact comparison for any load-estimation accuracy claim.

Every future formula change should receive a score-model version so reports from
different versions are not compared as if they were identical.
