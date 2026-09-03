# Next Test Phase

`test_core_logic.py` contains CPU-only invariant tests using a fake RKNN import
and synthetic arrays. It currently covers confidence contracts, sticky identity,
path jumps, contact timing, support geometry, load constraints, leg episodes,
and score coverage.

The next test boundaries, after real model contracts are confirmed, are:

- ground-spotter filtering before initial slot assignment;
- follow-crop and emergency-crop identity recovery;
- pose-candidate association inside a tracked crop;
- live-time duration handling with a controlled clock;
- foot-release and same-hold readjustment events; and
- inline/threaded rendering parity using synthetic frames.

CPU CI should use a fake RKNN module and synthetic arrays. On-device acceptance
remains a separate recorded procedure because ordinary CI cannot validate the
RK3588 runtime or NPU performance.

Run the current CPU suite from the repository root:

```bash
python -m pytest -q
```
