# Models

Model binaries are intentionally not tracked in Git.

The current single-file pipeline expects these names in its working directory:

- `yolo11m.rknn`
- `yolo11m-pose.rknn`
- `best.rknn`

Until model paths become configurable after acceptance testing, keep the actual
binaries in an external local model store and copy or link the accepted versions
to those root-level names when running the project.

Do not commit source checkpoints, RKNN files, calibration frames, or third-party
weights without first checking redistribution licences and choosing an artefact
policy. Record each accepted model's origin, conversion settings, input/output
contract, toolkit version, and SHA-256 value using
[`../docs/model-conversion.md`](../docs/model-conversion.md).
