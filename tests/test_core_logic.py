"""CPU-only invariant tests for the current single-file integration candidate."""

from collections import defaultdict
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline():
    """Import the module without requiring the board-only RKNN Lite wheel."""
    api = types.ModuleType("rknnlite.api")

    class DummyRKNNLite:
        NPU_CORE_AUTO = 0

    api.RKNNLite = DummyRKNNLite
    sys.modules.setdefault("rknnlite", types.ModuleType("rknnlite"))
    sys.modules["rknnlite.api"] = api

    path = Path(__file__).parents[1] / "rknn_finall.py"
    module = types.ModuleType("rknn_finall_under_test")
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


pipeline = _load_pipeline()


def test_confidence_contract_is_explicit():
    probabilities = np.array([0.1, 0.9], dtype=np.float32)
    np.testing.assert_allclose(
        pipeline._confidence_values(probabilities, False, "test"), probabilities
    )

    logits = np.array([0.0], dtype=np.float32)
    assert pipeline._confidence_values(logits, True, "test")[0] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="likely emits logits"):
        pipeline._confidence_values(np.array([-2.0], dtype=np.float32), False, "test")


def test_single_slot_is_not_recycled_after_a_long_gap():
    tracker = pipeline.RKNNDetectModel.__new__(pipeline.RKNNDetectModel)
    tracker.slot_boxes = {}
    tracker.slot_missed = defaultdict(int)
    tracker._next_id = 1

    ids = tracker._assign_slots(
        np.array([[10, 10, 100, 200]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
    )
    assert ids.tolist() == [1.0]

    for _ in range(100):
        tracker._assign_slots(np.empty((0, 4)), np.array([]))

    ids = tracker._assign_slots(
        np.array([[1000, 10, 1100, 200]], dtype=np.float32),
        np.array([0.99], dtype=np.float32),
    )
    assert ids.tolist() == [0.0]
    assert set(tracker.slot_boxes) == {1}


def test_path_does_not_add_a_teleport_segment():
    tracker = pipeline.PathTracker()

    def pose(x, y):
        return {
            "left_hip": (x - 5, y, 0.9),
            "right_hip": (x + 5, y, 0.9),
        }

    tracker.update(1, pose(10, 100))
    tracker.update(1, pose(20, 100))
    tracker.update(1, pose(500, 100))
    tracker.update(1, pose(510, 100))

    assert tracker.path_length(1) == pytest.approx(20.0)


def test_contact_requires_time_and_expires_when_pose_is_missing():
    holds = [{"box": (0, 0, 20, 20), "center": (10, 10), "class": 0, "conf": 1.0}]
    tracker = pipeline.GripTracker(holds)
    boxes = {1: (0, 0, 100, 200)}
    observed = {1: {"left_wrist": (10, 10, 0.9)}}

    tracker.update(observed, 0.0, boxes)
    assert tracker.current_contacts(1) == {}

    tracker.update(observed, 0.13, boxes)
    assert tracker.current_contacts(1) == {"left_wrist": 0}
    assert tracker.touched[0] == {"hand"}

    tracker.age_missing({1}, 0.50)
    assert tracker.current_contacts(1) == {}
    assert tracker.states[1]["left_wrist"]["status"] == "unknown"
    assert tracker.touched[0] == {"hand"}


def test_support_geometry_rewards_a_broad_supported_triangle():
    contacts = {"left_wrist", "right_wrist", "left_ankle"}
    broad = {
        "left_wrist": (0.0, 0.0, 0.9),
        "right_wrist": (100.0, 0.0, 0.9),
        "left_ankle": (50.0, 120.0, 0.9),
    }
    collapsed = {
        "left_wrist": (0.0, 0.0, 0.9),
        "right_wrist": (3.0, 0.0, 0.9),
        "left_ankle": (1.0, 4.0, 0.9),
    }

    good = pipeline.ClimbScorer._support_quality(
        broad, contacts, (50.0, 50.0), 150.0
    )
    poor = pipeline.ClimbScorer._support_quality(
        collapsed, contacts, (90.0, 90.0), 150.0
    )
    assert 0.0 <= poor < good <= 1.0


def _complete_pose():
    return {
        "nose": (50.0, 15.0, 0.95),
        "left_ear": (45.0, 15.0, 0.95),
        "right_ear": (55.0, 15.0, 0.95),
        "left_shoulder": (35.0, 35.0, 0.95),
        "right_shoulder": (65.0, 35.0, 0.95),
        "left_elbow": (20.0, 60.0, 0.95),
        "right_elbow": (80.0, 60.0, 0.95),
        "left_wrist": (0.0, 90.0, 0.95),
        "right_wrist": (100.0, 90.0, 0.95),
        "left_hip": (40.0, 90.0, 0.95),
        "right_hip": (60.0, 90.0, 0.95),
        "left_knee": (35.0, 125.0, 0.95),
        "right_knee": (65.0, 125.0, 0.95),
        "left_ankle": (20.0, 160.0, 0.95),
        "right_ankle": (80.0, 160.0, 0.95),
    }


def test_static_load_solution_obeys_force_and_projected_moment():
    pose = _complete_pose()
    com = pipeline.body_center_of_mass(pose)
    phase = {"state": "static", "com": com, "body_h": 160.0}
    contacts = {"left_wrist", "right_wrist", "left_ankle", "right_ankle"}
    analyzer = pipeline.LimbLoadAnalyzer(70)

    result = analyzer.analyze(1, pose, contacts, phase)

    assert result is not None
    assert result["certainty"] in {"constrained", "ambiguous"}
    assert "loads" not in result
    total = 70 * pipeline.GRAVITY
    assert sum(result["vertical_n"].values()) == pytest.approx(total, abs=1e-3)
    moment = sum(
        (result["positions"][name][0] - com[0]) * force
        for name, force in result["vertical_n"].items()
    )
    assert moment == pytest.approx(0.0, abs=1e-2)


def test_leg_drive_is_counted_once_per_upward_episode():
    scorer = pipeline.ClimbScorer(300)
    box = (0, 0, 100, 100)

    def pose(kind):
        ankle = (1.0, 1.0, 0.9) if kind == "bent" else (0.0, 2.0, 0.9)
        return {
            "left_hip": (0.0, 0.0, 0.9),
            "left_knee": (0.0, 1.0, 0.9),
            "left_ankle": ankle,
        }

    def update(timestamp, com_y, kind):
        scorer.update(
            timestamp,
            pose(kind),
            box,
            {},
            "unknown",
            {"state": "moving", "com": (50.0, com_y), "body_h": 100.0},
            [],
            {"left_ankle"},
        )

    update(0.0, 100.0, "bent")
    update(0.1, 98.0, "straight")
    update(0.2, 96.0, "straight")
    assert (scorer.leg_opportunities, scorer.leg_drives) == (1, 1)

    update(0.6, 96.0, "bent")
    update(0.7, 94.0, "straight")
    assert (scorer.leg_opportunities, scorer.leg_drives) == (2, 2)


def test_overall_remains_unknown_until_all_pillars_have_coverage():
    scorer = pipeline.ClimbScorer(300)
    overall, breakdown = scorer.compute()
    assert overall is None
    assert all(breakdown[key] is None for key in ("arm", "support", "footwork", "fluency"))

    scorer.arm_eligible = 2.0
    scorer.arm_penalty = 0.2
    scorer.support_total = 2.0
    scorer.support_sum = 1.6
    scorer.placements = 2
    scorer.adjustments = 0
    scorer.leg_opportunities = 3
    scorer.leg_drives = 3
    scorer.observed_time = 10.0
    scorer.pause_excess = 1.0

    overall, breakdown = scorer.compute()
    assert breakdown == {
        "arm": 90,
        "support": 80,
        "footwork": 100,
        "fluency": 90,
        "triangle": 80,
        "ge_raw": None,
        "placements": 2,
        "adjustments": 0,
        "leg_drives": 3,
        "leg_opportunities": 3,
        "coverage": {
            "arm_support_sec": 2.0,
            "support_sec": 2.0,
            "observed_sec": 10.0,
        },
    }
    assert overall == 90
