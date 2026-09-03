# Historical ONNX Prototypes

These files are unchanged development snapshots from the original CPU/ONNX
pipeline. They are retained for comparison while the RKNN candidate is being
validated; they are not current entry points and are not imported by
`rknn_finall.py`.

| File | Historical role |
|---|---|
| `phase1_skeleton.py` | early skeleton/detection experiment |
| `main.py` | original combined detection, pose, path, hold, and scoring pipeline |
| `main_crop1.py` | emergency crop recovery stage |
| `main_crop2_follow_crop.py` | follow-crop stage |
| `main_crop3_triangle_arm_score.py` | normalised triangle and arm-score stage |
| `main_crop4.py` | last repository CPU/ONNX stage with the old inverse-distance load model |

Their duplicated code and old scoring formulas should not be copied into new
work. After the RKNN pipeline receives a validated release tag, these snapshots
can be removed from the default branch because Git history will continue to
preserve them.
