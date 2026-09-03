# RKNN Model Contracts

## Expected files

The current code opens model filenames relative to the process working directory:

| File | Code input | Expected decoded output |
|---|---:|---|
| `yolo11m.rknn` | 640×640 | YOLO detection tensor with 80 class scores; class 0 is person |
| `yolo11m-pose.rknn` | 512×512 | YOLO pose tensor containing box, object confidence, and 17 `(x, y, confidence)` keypoints |
| `best.rknn` | 640×640 | YOLO detection tensor with two classes: grip and volume |

The filename does not prove an input size. Check the exported graph and RKNN
metadata before running it.

## Confidence contract

The code does not guess whether confidence tensors are probabilities or logits.
Set these constants to match the converted graphs:

```python
DETECT_OUTPUTS_ARE_LOGITS = False
POSE_OUTPUTS_ARE_LOGITS = False
```

Use `True` only when the matching final sigmoid was deliberately removed during
conversion. With `False`, values materially outside `[0, 1]` raise an error to
surface a contract mismatch instead of silently producing bad detections.

The hold detector currently shares the detection-output flag. Verify that its
conversion uses the same confidence convention as the person detector before
acceptance testing.

## INT8 calibration

Use representative frames from the real normal wall, lighting, camera distance,
clothing, body scales, holds, and expected occlusions. Keep calibration media
outside Git. Compare floating-point and INT8 models on the same labelled frames,
including:

- person recall and false detections;
- box drift during wide poses;
- keypoint confidence and localisation;
- wrist/ankle contact classification;
- identity recovery after missed detections; and
- per-model latency and thermal behaviour.

Do not accept an INT8 model from FPS alone. Define the maximum permitted accuracy
loss before running the comparison.

## Conversion record

Complete one row for every accepted model:

| Field | Person | Pose | Holds |
|---|---|---|---|
| Source model and version | pending | pending | pending |
| Source checkpoint hash | pending | pending | pending |
| Toolkit version | pending | pending | pending |
| Opset/export options | pending | pending | pending |
| Input size | pending | pending | pending |
| Quantisation | pending | pending | pending |
| Calibration-set version | pending | pending | pending |
| Output shape | pending | pending | pending |
| Probability or logits | pending | pending | pending |
| RKNN SHA-256 | pending | pending | pending |
| Accepted latency | pending | pending | pending |

Store the completed record in the validation report. Model binaries should stay
outside the repository unless their licences allow redistribution and Git LFS or
a release-asset policy is deliberately adopted.
