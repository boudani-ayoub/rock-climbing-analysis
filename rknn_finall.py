"""
Climbing AI — final single-climber RK3588 pipeline
==================================================
Based on the last tested normal-wall threaded pipeline. Person detection,
cropped skeleton estimation, the sticky one-person track, recovery, the hip
path, and threaded video rendering are retained. Contact state, technique
metrics, and the optional load display are estimate-aware: unknown evidence
remains unknown and estimated load is never graded.

Usage:
    python rknn_finall.py --source "climbing training data/1.mp4" --weight 30
    python rknn_finall.py --source 0   # live camera
"""

import cv2
import numpy as np
import argparse
import math
import time
import threading
import queue as queue_module
from pathlib import Path
from collections import defaultdict

try:
    from rknnlite.api import RKNNLite
except ImportError:
    raise ImportError("rknn-toolkit-lite2 required — install on Orange Pi")

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════

TRACK_MODEL    = "yolo11m.rknn"
POSE_MODEL     = "yolo11m-pose.rknn"
HOLD_MODEL     = "best.rknn"

DEFAULT_SOURCE   = 0
BODY_WEIGHT_KG   = None

INPUT_SIZE       = 640
# Pose runs on a smaller input than person/hold. NPU cost scales with input area:
# (512/640)^2 ≈ 0.64, so ~36% less pose compute. Person/hold stay at 640 for accuracy.
# The pose RKNN model MUST be rebuilt at this size (export yolo11n-pose at imgsz=512).
POSE_INPUT_SIZE  = 512
CONF_THRESH      = 0.03   # very low — RKNN post-sigmoid scores are small
NMS_THRESH       = 0.45
PERSON_CLASS     = 0

# Explicit model-output contracts. Set a flag True only for an RKNN exported
# from a graph whose final confidence Sigmoid was removed. Silent range-based
# guessing corrupted ordinary logits in [-5, 5].
DETECT_OUTPUTS_ARE_LOGITS = False
POSE_OUTPUTS_ARE_LOGITS   = False

# The session follows exactly one climber. A sticky slot prevents accumulated
# metrics from being handed to a new ID after a temporary detection gap.
MAX_TRACKED_PERSONS = 1
# Keep several detection candidates through the filter even though only 1 slot is
# tracked. This lets the slot matcher see BOTH climbers and lock the single slot to
# the nearest one (sticky), instead of the filter pre-picking whichever looks biggest
# this frame — which is what made the single ID flip between the two people.
CANDIDATE_KEEP = 5
# Lock gate: a tracked slot only re-matches a detection within this radius of its
# last position. Tight when just seen; grows with missed frames (climber may have
# moved while briefly undetected). Stops the single slot from jumping to a far-away
# second climber when the locked one is momentarily missed.
LOCK_GATE_BASE = 130   # px radius when slot was seen last frame
LOCK_GATE_STEP = 45    # extra px allowed per missed frame
LOCK_GATE_MAX  = 320   # cap so it never accepts an absurd jump
# ── Split inference schedules ──
# Person box changes slowly → detect rarely.
# Body joints change fast → pose must refresh often, or skeleton "floats".
DETECT_EVERY  = 2    # detect often so the box never goes stale (fixes floating skeleton)
POSE_EVERY    = 2    # pose refresh cadence
POSE_DRAW_MAX_AGE  = 4  # short display grace; stale poses are never scored
POSE_SCORE_MAX_AGE = 0  # only score/contact on fully fresh pose (no stale data)
# A pose can be "fresh" (just inferred) but still junk if the crop was stale or
# empty — the model returns low-confidence noise. Require a real, confident torso
# before drawing skeleton/triangle/diagram or scoring.
POSE_VALID_CONF    = 0.45  # a keypoint counts as "real" only above this confidence
POSE_VALID_MIN_KPS = 6     # need at least this many real keypoints
CACHE_MISS_TOL = 20  # keep last box if detection misses temporarily
# Follow-crop recovery runs an EXTRA person-detection to re-find a missed climber.
# Doing it every missed frame is the #1 cause of FPS dips (npu_calls 2-3 per frame).
# The cached box carries the climber meanwhile, so only attempt recovery every Nth
# missed detect-frame. Higher = fewer dips but slightly staler box when missed.
RECOVERY_EVERY = 3

# ── Visualization toggles ──
# Agent wants triangle technique + weight/load overlays visible.
# These draw ONLY on fresh pose; stale frames never show them.
DRAW_WEIGHT_TRIANGLE = True
DRAW_WEIGHT_DIAGRAM  = True  # per-limb force dots/text over the body
# Overlap NPU inference with CPU drawing on a worker thread. Validated: ~14→~20 FPS,
# scores/IDs match the inline path. Set False to fall back to proven inline drawing.
THREADED_DRAW = True
MIN_TRACK_W       = 45
MIN_TRACK_H       = 90
MIN_TRACK_AREA    = 5000
MIN_TRACK_ASPECT  = 0.55   # h/w lower bound
MAX_TRACK_ASPECT  = 5.50   # h/w upper bound

GROUND_FILTER    = 0.85   # reject people whose box-center is in the bottom 15% (standing spotters, non-climbers)
MIN_PERSON_H     = 90
KP_CONF          = 0.03
CROP_PAD         = 1.0

HOLD_DETECT_FRAMES = 5
HOLD_CONF          = 0.03   # low for RKNN
HOLD_NMS_DIST      = 30
# Contact requires a CONFIDENT hand/foot keypoint — much higher than the global
# KP_CONF (0.03) used for drawing. A jittery low-confidence wrist/ankle can land
# anywhere (e.g. near top holds) and falsely mark them touched; this gate blocks that.
GRIP_KP_CONF       = 0.40

# Contacts use source-video time, never processing speed.
CONTACT_CONFIRM_SEC = 0.12
CONTACT_RELEASE_SEC = 0.15
CONTACT_UNKNOWN_SEC = 0.35
CONTACT_RADIUS_BODY = 0.06


ARM_STRAIGHT_ANGLE = 150
ARM_BEND_GRACE_SEC = 1.0
LOCK_SECONDS     = 3.0
GRAVITY          = 9.81

STATIC_COM_SPEED_BODY_S  = 0.18
STATIC_LIMB_SPEED_BODY_S = 0.45
STATIC_CONFIRM_SEC       = 0.25
FLUENCY_PAUSE_GRACE_SEC  = 2.5
FOOT_ADJUST_WINDOW_SEC   = 1.2

LOST_FRAMES_TRIGGER  = 3
ZOOM_CROP_HALF_W     = 300
ZOOM_CROP_HALF_H     = 400
ZOOM_EXPAND_PER_SEC  = 50

FOLLOW_CROP_PADDING  = 2.0
FOLLOW_CROP_MIN_HALF = 60
FOLLOW_MIN_FRAMES    = 5

# ═══════════════════════════════════════════════════════
#  RKNN ADAPTERS — mimic Ultralytics result objects
# ═══════════════════════════════════════════════════════

class NP:
    """Numpy wrapper that acts like a PyTorch tensor."""
    def __init__(self, arr):
        self._d = np.array(arr) if not isinstance(arr, np.ndarray) else arr
    def cpu(self): return self
    def numpy(self): return self._d
    def sum(self): return NP(np.array(self._d.sum()))
    def __float__(self): return float(self._d.flat[0] if self._d.ndim > 0 else self._d)
    def __int__(self): return int(self._d.flat[0] if self._d.ndim > 0 else self._d)
    def __len__(self): return len(self._d)
    def __getitem__(self, i):
        r = self._d[i]
        return NP(r) if isinstance(r, np.ndarray) else r

class FakeBoxes:
    """Mimics ultralytics Boxes object."""
    def __init__(self, xyxy, conf, cls, track_id=None):
        self.xyxy = NP(xyxy.astype(np.float32))
        self.conf = NP(conf.astype(np.float32))
        self.cls  = NP(cls.astype(np.float32))
        self.id   = NP(track_id.astype(np.float32)) if track_id is not None else None
        self._n   = len(conf)

    def __len__(self): return self._n
    def __iter__(self):
        for i in range(self._n):
            yield FakeBoxes(
                self.xyxy._d[i:i+1], self.conf._d[i:i+1], self.cls._d[i:i+1],
                self.id._d[i:i+1] if self.id is not None else None)
    def __getitem__(self, i):
        return FakeBoxes(
            self.xyxy._d[i:i+1], self.conf._d[i:i+1], self.cls._d[i:i+1],
            self.id._d[i:i+1] if self.id is not None else None)

class FakeKeypoints:
    """Mimics ultralytics Keypoints object."""
    def __init__(self, xy, conf):
        self.xy   = NP(xy.astype(np.float32))    # (N, 17, 2)
        self.conf = NP(conf.astype(np.float32))   # (N, 17)

class FakeResult:
    """Mimics ultralytics Results object."""
    def __init__(self, boxes=None, keypoints=None):
        self.boxes     = boxes
        self.keypoints = keypoints

# ═══════════════════════════════════════════════════════
#  RKNN INFERENCE HELPERS
# ═══════════════════════════════════════════════════════

def letterbox(frame, size=640):
    """Resize with center padding → (padded_img, scale, pad_w, pad_h)."""
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(frame, (nw, nh))
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_h = (size - nh) // 2
    pad_w = (size - nw) // 2
    padded[pad_h:pad_h+nh, pad_w:pad_w+nw] = img
    return padded, scale, pad_w, pad_h

_INFER_CALLS = 0   # global NPU inference counter (for diagnosing slow frames)

def rknn_infer(rknn, frame, size=INPUT_SIZE):
    """Letterbox + RGB + batch dim + RKNN inference."""
    global _INFER_CALLS
    _INFER_CALLS += 1
    img, scale, pad_w, pad_h = letterbox(frame, size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_4d  = np.expand_dims(img_rgb, 0)
    outputs = rknn.inference(inputs=[img_4d])
    return outputs, scale, pad_w, pad_h

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _confidence_values(values, are_logits, label):
    """Apply the declared model contract and reject obvious contract mistakes."""
    values = values.astype(np.float32)
    if are_logits:
        return _sigmoid(values)
    if values.size and (float(values.min()) < -0.02 or float(values.max()) > 1.02):
        raise ValueError(
            f"{label} contains values outside [0,1]. The RKNN likely emits logits; "
            f"set the matching *_OUTPUTS_ARE_LOGITS flag to True.")
    return values


def decode_boxes(raw, num_classes, conf_thresh, classes_filter=None):
    """Decode YOLO output → (xyxy, scores, class_ids) in 640x640 space.

    raw: [1, 4+num_classes, N] numpy array.
    Returns arrays in xyxy format, filtered by confidence.
    """
    out = raw[0]                            # (4+nc, N)
    if out.shape[0] <= out.shape[1]:
        out = out.T                         # (N, 4+nc)

    boxes_xywh   = out[:, :4].astype(np.float32)
    class_scores = out[:, 4:4 + num_classes].astype(np.float32)
    class_scores = _confidence_values(
        class_scores, DETECT_OUTPUTS_ARE_LOGITS, "Detection confidence")

    class_ids  = np.argmax(class_scores, axis=1)
    max_scores = np.amax(class_scores, axis=1)

    # Filter by confidence
    mask = max_scores >= conf_thresh
    if classes_filter is not None:
        mask &= np.isin(class_ids, classes_filter)

    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids  = class_ids[mask]

    if len(boxes_xywh) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])

    # xywh → xyxy
    xyxy = np.zeros_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # NMS — cv2.dnn.NMSBoxes expects [x1, y1, width, height], not center-format
    nms_rects = np.column_stack([
        xyxy[:, 0],
        xyxy[:, 1],
        xyxy[:, 2] - xyxy[:, 0],
        xyxy[:, 3] - xyxy[:, 1],
    ]).astype(np.float32)
    idxs = cv2.dnn.NMSBoxes(
        nms_rects.tolist(), max_scores.tolist(), conf_thresh, NMS_THRESH)
    if len(idxs) == 0:
        return np.empty((0, 4)), np.array([]), np.array([])
    idxs = np.array(idxs).flatten()

    return xyxy[idxs], max_scores[idxs], class_ids[idxs]

def unpad_boxes(xyxy, scale, pad_w, pad_h, frame_w, frame_h):
    """Convert boxes from 640x640 letterbox space to original frame space."""
    xyxy = xyxy.copy().astype(np.float32)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_w) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_h) / scale
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, frame_w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, frame_h)
    return xyxy

def filter_person_candidates(xyxy, scores, class_ids, max_people=MAX_TRACKED_PERSONS):
    """Reject tiny/partial/person-like junk boxes and keep only top people."""
    if len(xyxy) == 0:
        return xyxy, scores, class_ids

    xyxy = xyxy.astype(np.float32)
    scores = scores.astype(np.float32)
    class_ids = class_ids.astype(np.float32)

    bw = xyxy[:, 2] - xyxy[:, 0]
    bh = xyxy[:, 3] - xyxy[:, 1]
    area = bw * bh
    aspect = bh / np.maximum(bw, 1.0)

    mask = (
        (bw >= MIN_TRACK_W) &
        (bh >= MIN_TRACK_H) &
        (area >= MIN_TRACK_AREA) &
        (aspect >= MIN_TRACK_ASPECT) &
        (aspect <= MAX_TRACK_ASPECT)
    )

    xyxy = xyxy[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(xyxy) == 0:
        return xyxy, scores, class_ids

    # Rank by confidence and size. This removes tiny false positives but keeps real bodies.
    bw = xyxy[:, 2] - xyxy[:, 0]
    bh = xyxy[:, 3] - xyxy[:, 1]
    area = bw * bh
    rank = scores * np.sqrt(np.maximum(area, 1.0))

    keep = np.argsort(rank)[-max_people:][::-1]
    return xyxy[keep], scores[keep], class_ids[keep]

# ═══════════════════════════════════════════════════════
#  RKNN MODEL WRAPPERS — behave like YOLO(...)
# ═══════════════════════════════════════════════════════

class RKNNDetectModel:
    """YOLO-style detector with one sticky, attempt-owned tracking slot."""

    def __init__(self, model_path, core_mask=None):
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load {model_path}")
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_AUTO
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init runtime: {model_path}")

        # A single sticky slot. The score belongs to one attempt and one person;
        # it must not be silently handed to a newcomer after a detection gap.
        # ByteTrack keeps minting new IDs with weak RKNN detections, so we do not use it here.
        self.tracker = None
        self.slot_boxes = {}
        self.slot_missed = defaultdict(int)
        self._next_id = 1
        print(f"  RKNN loaded: {model_path}")

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / max(1.0, area_a + area_b - inter)

    def _center_dist(self, a, b):
        acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)

    def _assign_slots(self, xyxy, scores):
        """Assign detections to stable IDs 1..MAX_TRACKED_PERSONS only."""
        ids = np.zeros(len(xyxy), dtype=np.float32)
        if len(xyxy) == 0:
            for sid in list(self.slot_missed.keys()):
                self.slot_missed[sid] += 1
            return ids

        order = np.argsort(scores)[::-1]
        used_slots = set()
        used_dets = set()

        # First: match detections to existing slots by GLOBAL nearest pairing,
        # but only within each slot's LOCK GATE. The gate is tight when the slot
        # was just seen and grows with missed frames. This is what stops the swap:
        # a far-away second climber is outside the gate, so it can never capture
        # the locked slot even when the locked climber is briefly missed.
        candidates = []
        for di in range(len(xyxy)):
            box = xyxy[di]
            for sid, prev_box in self.slot_boxes.items():
                iou = self._iou(box, prev_box)
                cd  = self._center_dist(box, prev_box)
                gate = min(LOCK_GATE_BASE + self.slot_missed.get(sid, 0) * LOCK_GATE_STEP,
                           LOCK_GATE_MAX)
                if iou >= 0.05 or cd <= gate:
                    cost = cd - iou * 200.0   # distance dominates, IOU breaks ties
                    candidates.append((cost, di, sid))

        candidates.sort(key=lambda c: c[0])
        for cost, di, sid in candidates:
            if di in used_dets or sid in used_slots:
                continue
            ids[di] = float(sid)
            self.slot_boxes[sid] = xyxy[di].copy()
            self.slot_missed[sid] = 0
            used_slots.add(sid)
            used_dets.add(di)

        # Second: assign a detection only when the single slot has never been used.
        # Once acquired, ID 1 remains owned by this attempt.
        free_slots = [sid for sid in range(1, MAX_TRACKED_PERSONS + 1)
                      if sid not in used_slots and sid not in self.slot_boxes]
        for di in order:
            if di in used_dets:
                continue
            if not free_slots:
                break

            sid = free_slots.pop(0)
            ids[di] = float(sid)
            self.slot_boxes[sid] = xyxy[di].copy()
            self.slot_missed[sid] = 0
            used_slots.add(sid)
            used_dets.add(di)

        # Age unmatched slots for the expanding spatial gate and recovery trigger.
        for sid in list(self.slot_boxes.keys()):
            if sid not in used_slots:
                self.slot_missed[sid] += 1
                # ID 1 is not recycled during an attempt. Emergency recovery
                # continues searching for the same climber.

        return ids

    def track(self, frame, classes=None, persist=True,
              tracker=None, verbose=False, device=None):
        """Detect + track → list of FakeResult (mimics YOLO.track)."""
        outputs, scale, pad_w, pad_h = rknn_infer(self.rknn, frame)
        h, w = frame.shape[:2]

        xyxy, scores, class_ids = decode_boxes(
            outputs[0], 80, CONF_THRESH, classes)

        if len(xyxy) > 0:
            xyxy = unpad_boxes(xyxy, scale, pad_w, pad_h, w, h)
            if classes is not None and PERSON_CLASS in classes:
                xyxy, scores, class_ids = filter_person_candidates(
                    xyxy, scores, class_ids, CANDIDATE_KEEP)
                # Filter before the sticky slot is assigned. Otherwise a ground
                # spotter could permanently own ID 1 and lock out the climber.
                wall_mask=np.asarray([on_wall(box,h) for box in xyxy],dtype=bool)
                xyxy,scores,class_ids=xyxy[wall_mask],scores[wall_mask],class_ids[wall_mask]

        if len(xyxy) == 0:
            self._assign_slots(xyxy, scores)
            return [FakeResult(boxes=FakeBoxes(
                np.empty((0,4)), np.array([]), np.array([]), np.array([])))]

        # Stable hard-limited slot IDs: only 1..MAX_TRACKED_PERSONS.
        ids = self._assign_slots(xyxy, scores)
        keep = ids > 0

        if not np.any(keep):
            return [FakeResult(boxes=FakeBoxes(
                np.empty((0,4)), np.array([]), np.array([]), np.array([])))]

        boxes = FakeBoxes(
            xyxy[keep],
            scores[keep],
            class_ids[keep].astype(np.float32),
            ids[keep].astype(np.float32)
        )

        return [FakeResult(boxes=boxes)]

    def sync_single_slot(self, box):
        """Synchronize the sticky slot after successful emergency recovery."""
        self.slot_boxes[1] = np.asarray(box, dtype=np.float32).copy()
        self.slot_missed[1] = 0

    def __call__(self, frame, verbose=False, classes=None, device=None,
                 conf=None):
        """Detect without tracking → list of FakeResult (for follow/zoom)."""
        outputs, scale, pad_w, pad_h = rknn_infer(self.rknn, frame)
        h, w = frame.shape[:2]

        thresh = conf if conf is not None else CONF_THRESH
        xyxy, scores, class_ids = decode_boxes(
            outputs[0], 80, thresh, classes)

        if len(xyxy) > 0:
            xyxy = unpad_boxes(xyxy, scale, pad_w, pad_h, w, h)
            if classes is not None and PERSON_CLASS in classes:
                xyxy, scores, class_ids = filter_person_candidates(
                    xyxy, scores, class_ids, CANDIDATE_KEEP)

        boxes = FakeBoxes(xyxy, scores, class_ids.astype(np.float32))
        return [FakeResult(boxes=boxes)]

    def release(self):
        self.rknn.release()


class RKNNPoseModel:
    """Drop-in replacement for YOLO pose model."""

    def __init__(self, model_path, core_mask=None):
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load {model_path}")
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_AUTO
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init runtime: {model_path}")
        print(f"  RKNN loaded: {model_path}")

    def __call__(self, frame, verbose=False, device=None):
        """Pose estimation → list of FakeResult (mimics YOLO(crop))."""
        outputs, scale, pad_w, pad_h = rknn_infer(self.rknn, frame, POSE_INPUT_SIZE)
        h, w = frame.shape[:2]

        raw = outputs[0]  # (1, 56, 8400) for pose
        out = raw[0]       # (56, 8400)
        if out.shape[0] < out.shape[1]:
            out = out.T    # (8400, 56)

        boxes  = out[:, :4].astype(np.float32)
        scores = _confidence_values(
            out[:, 4], POSE_OUTPUTS_ARE_LOGITS, "Pose object confidence")

        mask = scores >= CONF_THRESH
        out    = out[mask]
        scores = scores[mask]

        if len(out) == 0:
            return [FakeResult(keypoints=None)]

        # The crop is centred on the tracked box. Select one central, confident
        # pose before materialising keypoints, so a person at a crop edge cannot
        # replace the tracked climber's skeleton.
        kept_boxes = boxes[mask]
        bw = np.maximum(kept_boxes[:, 2], 1.0)
        bh = np.maximum(kept_boxes[:, 3], 1.0)
        cx = kept_boxes[:, 0]
        cy = kept_boxes[:, 1]
        centre_d2 = ((cx - POSE_INPUT_SIZE / 2) / POSE_INPUT_SIZE) ** 2 + \
                    ((cy - POSE_INPUT_SIZE / 2) / POSE_INPUT_SIZE) ** 2
        rank = scores * np.sqrt(bw * bh) * np.exp(-2.0 * centre_d2)
        best = int(np.argmax(rank))
        kp_data = out[best, 5:56].astype(np.float32)
        if len(kp_data) < 51:
            return [FakeResult(keypoints=None)]
        kp_data = kp_data.reshape(17, 3)
        xy = kp_data[:, :2].copy()
        conf = _confidence_values(
            kp_data[:, 2], POSE_OUTPUTS_ARE_LOGITS, "Pose keypoint confidence")
        xy[:, 0] = (xy[:, 0] - pad_w) / scale
        xy[:, 1] = (xy[:, 1] - pad_h) / scale

        kps = FakeKeypoints(
            np.array([xy], dtype=np.float32),
            np.array([conf], dtype=np.float32),
        )
        return [FakeResult(keypoints=kps)]

    def release(self):
        self.rknn.release()


class RKNNHoldModel:
    """Drop-in replacement for YOLO hold detection model."""

    def __init__(self, model_path, num_classes=2, core_mask=None):
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load {model_path}")
        if core_mask is None:
            core_mask = RKNNLite.NPU_CORE_AUTO
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError(f"Failed to init runtime: {model_path}")
        self.num_classes = num_classes
        print(f"  RKNN loaded: {model_path}")

    def __call__(self, frame, verbose=False, conf=None, device=None):
        """Hold detection → list of FakeResult."""
        outputs, scale, pad_w, pad_h = rknn_infer(self.rknn, frame)
        h, w = frame.shape[:2]
        thresh = conf if conf is not None else HOLD_CONF

        xyxy, scores, class_ids = decode_boxes(
            outputs[0], self.num_classes, thresh)

        if len(xyxy) > 0:
            xyxy = unpad_boxes(xyxy, scale, pad_w, pad_h, w, h)

        boxes = FakeBoxes(xyxy, scores, class_ids.astype(np.float32))
        return [FakeResult(boxes=boxes)]

    def release(self):
        self.rknn.release()

# ═══════════════════════════════════════════════════════
#  COLORS
# ═══════════════════════════════════════════════════════

ID_COLORS = [
    (100,220,255),(100,255,150),(255,150,100),
    (180,100,255),(100,180,255),(255,100,180),
]
def id_color(pid): return ID_COLORS[(pid-1) % len(ID_COLORS)]

COL_PATH      = (255, 60,  255)
PATH_COLORS = [
    (0,255,255),(255,50,255),(50,255,50),(255,100,50),
    (50,100,255),(255,255,50),(50,255,200),(200,50,255),
]
_path_color_map  = {}
_path_color_next = [0]
def path_color(pid):
    if pid not in _path_color_map:
        _path_color_map[pid] = PATH_COLORS[_path_color_next[0] % len(PATH_COLORS)]
        _path_color_next[0] += 1
    return _path_color_map[pid]

COL_TRI_GOOD  = (50,  220,  50)
COL_TRI_OK    = (50,  200, 220)
COL_TRI_WARN  = (50,   50, 255)
COL_HOLD_IDLE = (160, 160, 160)
COL_HOLD_HAND = (0,   165, 255)
COL_HOLD_FOOT = (255, 100,  50)

# ═══════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════

def dist(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def blend_poly(frame, pts, color, alpha):
    """Alpha-fill a polygon, blending ONLY its bounding box instead of copying the
    whole frame. A full-frame frame.copy()+addWeighted is ~2.7M pixel ops at 1280x720;
    localizing it to the shape's bbox cuts that by 10-50x. Called per climber per frame."""
    pts = np.asarray(pts, np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    ov = roi.copy()
    cv2.fillPoly(ov, [pts - [x0, y0]], color)
    cv2.addWeighted(ov, alpha, roi, 1.0 - alpha, 0, roi)

def blend_rect(frame, x0, y0, x1, y1, color, alpha):
    """Alpha-fill a rectangle region, blending only that region."""
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(frame.shape[1], int(x1)); y1 = min(frame.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    ov = np.empty_like(roi); ov[:] = color
    cv2.addWeighted(ov, alpha, roi, 1.0 - alpha, 0, roi)

def angle3(a, b, c):
    if not all([a, b, c]): return None
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0: return None
    return math.degrees(math.acos(max(-1., min(1., (ba[0]*bc[0]+ba[1]*bc[1])/mag))))

def tri_area(p1, p2, p3):
    return abs((p1[0]*(p2[1]-p3[1]) +
                p2[0]*(p3[1]-p1[1]) +
                p3[0]*(p1[1]-p2[1])) / 2)

def kp(keypoints, name):
    k = keypoints.get(name)
    if k and k[2] >= KP_CONF:
        return (k[0], k[1])
    return None

KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]

def extract_kps(kpts_xy, kpts_conf, ox=0, oy=0):
    """Works with both real tensors and NP wrappers."""
    if hasattr(kpts_xy, 'cpu'):
        pts  = kpts_xy.cpu().numpy()
    elif isinstance(kpts_xy, np.ndarray):
        pts = kpts_xy
    else:
        pts = np.array(kpts_xy)
    if hasattr(kpts_conf, 'cpu'):
        conf = kpts_conf.cpu().numpy()
    elif isinstance(kpts_conf, np.ndarray):
        conf = kpts_conf
    else:
        conf = np.array(kpts_conf)
    return {KP_NAMES[i]: (float(pts[i][0])+ox,
                           float(pts[i][1])+oy,
                           float(conf[i]))
            for i in range(17)}

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
CRITICAL = {9, 10, 15, 16}

def pose_is_valid(kps):
    """A pose is real only if it has a confident torso plus enough confident
    keypoints. Rejects low-confidence noise from stale/empty crops, which is
    what causes the 'floating triangle' drawn away from the climber."""
    if not kps:
        return False
    n_conf = sum(1 for v in kps.values() if v[2] >= POSE_VALID_CONF)
    if n_conf < POSE_VALID_MIN_KPS:
        return False
    ls = kps.get("left_shoulder");  rs = kps.get("right_shoulder")
    lh = kps.get("left_hip");       rh = kps.get("right_hip")
    sh_ok  = (ls and ls[2] >= POSE_VALID_CONF) or (rs and rs[2] >= POSE_VALID_CONF)
    hip_ok = (lh and lh[2] >= POSE_VALID_CONF) or (rh and rh[2] >= POSE_VALID_CONF)
    return bool(sh_ok and hip_ok)

# ═══════════════════════════════════════════════════════
#  WALL FILTER
# ═══════════════════════════════════════════════════════

def on_wall(box, H):
    x1,y1,x2,y2 = box
    return (y2-y1) >= MIN_PERSON_H and ((y1+y2)/2) < H * GROUND_FILTER

# ═══════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════

def draw_skeleton(frame, keypoints, color):
    pts = list(keypoints.values())
    if len(pts) < 17: return
    for p1, p2 in SKELETON:
        k1, k2 = pts[p1], pts[p2]
        if k1[2] >= KP_CONF and k2[2] >= KP_CONF:
            cv2.line(frame, (int(k1[0]),int(k1[1])),
                     (int(k2[0]),int(k2[1])), color, 2, cv2.LINE_AA)
    for i, k in enumerate(pts):
        if k[2] >= KP_CONF:
            cx, cy = int(k[0]), int(k[1])
            if i in CRITICAL:
                cv2.circle(frame, (cx,cy), 7, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx,cy), 7, (255,255,255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx,cy), 4, (255,255,255), -1, cv2.LINE_AA)

def draw_person_box(frame, box, pid, color):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
    lbl = f"ID {pid}"
    tw, th = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.rectangle(frame, (x1,y1-th-6), (x1+tw+6, y1), color, -1)
    cv2.putText(frame, lbl, (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20,20,20), 1)

def draw_info_panel(frame, frame_num, fps, active_ids, device_name):
    n = len(active_ids)
    blend_rect(frame, 10, 10, 260, 68+n*18, (15,15,15), 0.65)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Frame {frame_num}  |  {fps:.1f} FPS",
                (18,30), font, 0.42, (200,200,200), 1)
    cv2.putText(frame, f"Device: {device_name}", (18,48), font, 0.38, (100,220,255), 1)
    cv2.putText(frame, f"Climbers on wall: {n}", (18,66), font, 0.42, (100,255,150), 1)
    y = 84
    for pid in sorted(active_ids):
        cv2.putText(frame, f"  ID {pid}", (18,y), font, 0.38, id_color(pid), 1)
        y += 18

# ═══════════════════════════════════════════════════════
#  MODULE 1 — PATH TRACKING
# ═══════════════════════════════════════════════════════

def body_center_of_mass(kps):
    """Return a generic-adult, mass-weighted 2D body centre when coverage is adequate."""
    def point(name, threshold=POSE_VALID_CONF):
        v=kps.get(name)
        return np.array(v[:2],dtype=np.float64) if v and v[2]>=threshold else None
    def along(a,b,r):
        pa,pb=point(a),point(b)
        return None if pa is None or pb is None else pa+r*(pb-pa)
    parts=[]
    def add(mass,p):
        if p is not None: parts.append((mass,p))
    ls,rs,lh,rh=(point(n) for n in ("left_shoulder","right_shoulder","left_hip","right_hip"))
    if all(p is not None for p in (ls,rs,lh,rh)):
        add(.4346,((ls+rs)/2+(lh+rh)/2)/2)
    head=point("nose")
    if head is None:
        le,re=point("left_ear"),point("right_ear")
        head=None if le is None or re is None else (le+re)/2
    add(.0694,head)
    for side in ("left","right"):
        add(.0271,along(f"{side}_shoulder",f"{side}_elbow",.5772))
        add(.0162,along(f"{side}_elbow",f"{side}_wrist",.4574))
        add(.0061,point(f"{side}_wrist"))
        add(.1416,along(f"{side}_hip",f"{side}_knee",.4095))
        add(.0433,along(f"{side}_knee",f"{side}_ankle",.4459))
        add(.0137,point(f"{side}_ankle"))
    observed=sum(m for m,_ in parts)
    if observed<.70: return None
    c=sum((m*p for m,p in parts),np.zeros(2))/observed
    return float(c[0]),float(c[1])


class MotionPhaseTracker:
    """Classify only fresh observations; cached frames add no timing evidence."""
    JOINTS=("left_wrist","right_wrist","left_ankle","right_ankle")
    def __init__(self):
        self.previous={}
        self.ema=defaultdict(lambda:{"com":0.0,"limb":0.0})
        self.static_since=defaultdict(lambda:None)
    def update(self,pid,kps,box,source_time):
        body_h=max(1.0,float(box[3]-box[1]))
        com=body_center_of_mass(kps)
        current={"time":source_time,"com":com,"joints":{}}
        for name in self.JOINTS:
            v=kps.get(name)
            if v and v[2]>=GRIP_KP_CONF: current["joints"][name]=(v[0],v[1])
        prev=self.previous.get(pid); self.previous[pid]=current
        unknown={"state":"unknown","com":com,"com_speed":None,"limb_speed":None,"body_h":body_h}
        if prev is None or com is None or prev["com"] is None: return unknown
        dt=source_time-prev["time"]
        if dt<=1e-6 or dt>.5:
            self.static_since[pid]=None; return unknown
        cs=dist(com,prev["com"])/body_h/dt
        speeds=[dist(p,prev["joints"][n])/body_h/dt for n,p in current["joints"].items() if n in prev["joints"]]
        ls=max(speeds) if speeds else 0.0
        e=self.ema[pid]; e["com"]=.45*cs+.55*e["com"]; e["limb"]=.45*ls+.55*e["limb"]
        quiet=e["com"]<=STATIC_COM_SPEED_BODY_S and e["limb"]<=STATIC_LIMB_SPEED_BODY_S
        if quiet:
            if self.static_since[pid] is None: self.static_since[pid]=source_time
            state="static" if source_time-self.static_since[pid]>=STATIC_CONFIRM_SEC else "transition"
        else:
            self.static_since[pid]=None; state="moving"
        return {"state":state,"com":com,"com_speed":e["com"],"limb_speed":e["limb"],"body_h":body_h}


class PathTracker:
    """Climber-following hip path with incremental, jump-safe length."""
    MAX_SEGMENT=150.0
    def __init__(self):
        self.history=defaultdict(list); self.lengths=defaultdict(float)
    def update(self,pid,kps):
        lh,rh=kp(kps,"left_hip"),kp(kps,"right_hip")
        if not lh or not rh: return None
        pt=(int((lh[0]+rh[0])/2),int((lh[1]+rh[1])/2)); h=self.history[pid]
        if not h: h.append(pt)
        else:
            d=dist(h[-1],pt)
            if d>5:
                if d<=self.MAX_SEGMENT: self.lengths[pid]+=d
                h.append(pt)
        return pt
    def draw(self,frame,pid,color=None):
        pts=self.history[pid]; col=color if color is not None else COL_PATH
        for i in range(1,len(pts)):
            if dist(pts[i-1],pts[i])<=self.MAX_SEGMENT:
                cv2.line(frame,pts[i-1],pts[i],col,3,cv2.LINE_AA)
        if pts:
            cv2.circle(frame,pts[-1],6,COL_PATH,-1,cv2.LINE_AA)
            cv2.circle(frame,pts[-1],6,(255,255,255),1,cv2.LINE_AA)
    def path_length(self,pid): return self.lengths[pid]
    def highest_point(self,pid):
        pts=self.history[pid]; return min(pts,key=lambda p:p[1]) if pts else None

# ═══════════════════════════════════════════════════════
#  MODULE 2 — TRIANGLE SUPPORT
# ═══════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════
#  MODULE 3 — LOCK ARM WARNING
# ═══════════════════════════════════════════════════════

class LockArmDetector:
    """Warn only for a sustained, supported bend during a static phase."""
    def __init__(self,fps):
        self.thresh=LOCK_SECONDS
        self.frames=defaultdict(lambda:{"left":0.0,"right":0.0})
        self.last_time=defaultdict(lambda:{"left":None,"right":None})
    def update(self,pid,kps,source_time=None,fresh=False,support_joints=None,phase_state="unknown"):
        status={"left":"unknown","right":"unknown","warning":False}; support_joints=support_joints or set()
        for side,(sh,el,wr) in {"left":("left_shoulder","left_elbow","left_wrist"),
                                "right":("right_shoulder","right_elbow","right_wrist")}.items():
            ang=angle3(kp(kps,sh),kp(kps,el),kp(kps,wr))
            if ang is not None: status[side]="bent" if ang<ARM_STRAIGHT_ANGLE else "straight"
            if not fresh or source_time is None: continue
            last=self.last_time[pid][side]
            dt=0.0 if last is None else max(0.0,min(.25,source_time-last))
            self.last_time[pid][side]=source_time
            active=ang is not None and ang<ARM_STRAIGHT_ANGLE and wr in support_joints and phase_state=="static"
            self.frames[pid][side]=self.frames[pid][side]+dt if active else 0.0
        status["warning"]=any(v>=self.thresh for v in self.frames[pid].values())
        return status
    def draw(self,frame,pid,kps,arm_status):
        draw_lock_overlay(frame,kps,arm_status,self.frames[pid],self.thresh)

# ═══════════════════════════════════════════════════════
#  MODULE 4 — LIMB LOAD ANALYSIS
# ═══════════════════════════════════════════════════════

class LimbLoadAnalyzer:
    """Constrained 2D quasi-static vertical-support estimate.

    It enforces vertical force and projected moment balance, reports feasible
    ranges, and refuses unsuitable observations. It is separate from scoring
    because this is not a 3D force measurement.
    """
    JOINT_TO_LIMB={"left_wrist":"left_hand","right_wrist":"right_hand",
                   "left_ankle":"left_foot","right_ankle":"right_foot"}
    def __init__(self,weight_kg):
        self.weight_kg=float(weight_kg); self.previous={}
    @staticmethod
    def _vertices(xs,total):
        n=len(xs); vertices=[]; tol=1e-7
        for i,x in enumerate(xs):
            if abs(x)<=tol:
                f=np.zeros(n); f[i]=total; vertices.append(f)
        for i in range(n):
            for j in range(i+1,n):
                den=xs[j]-xs[i]
                if abs(den)<=tol: continue
                fi=total*xs[j]/den; fj=-total*xs[i]/den
                if fi>=-tol and fj>=-tol:
                    f=np.zeros(n); f[i],f[j]=max(0.0,fi),max(0.0,fj); vertices.append(f)
        unique=[]
        for f in vertices:
            if not any(np.allclose(f,q,atol=1e-5) for q in unique): unique.append(f)
        return unique
    @staticmethod
    def _closest_feasible(xs,total,prior):
        n=len(xs); A0=np.vstack([np.ones(n),np.asarray(xs)]); b=np.array([total,0.0])
        best=None; best_cost=float("inf")
        for mask in range(1,1<<n):
            idx=[i for i in range(n) if mask&(1<<i)]
            A=A0[:,idx]; p=prior[idx]
            f=p+A.T@np.linalg.pinv(A@A.T)@(b-A@p)
            if np.linalg.norm(A@f-b)>max(1e-4,total*1e-5) or np.any(f< -1e-6): continue
            full=np.zeros(n); full[idx]=np.maximum(f,0.0)
            cost=float(np.sum((full-prior)**2))
            if cost<best_cost: best,best_cost=full,cost
        return best
    def analyze(self,pid,kps,active_contacts,phase_info):
        if not active_contacts or phase_info.get("state")!="static": return None
        com=phase_info.get("com") or body_center_of_mass(kps)
        body_h=max(1.0,phase_info.get("body_h",1.0))
        if com is None or len(active_contacts)<2: return None
        names=[]; positions={}
        for joint,name in self.JOINT_TO_LIMB.items():
            if joint not in active_contacts: continue
            p=kp(kps,joint)
            if p is not None: names.append(name); positions[name]=p
        if len(names)<2: return None
        total=self.weight_kg*GRAVITY
        xs=[(positions[n][0]-com[0])/body_h for n in names]
        vertices=self._vertices(xs,total)
        if not vertices: return None
        prior_map=self.previous.get(pid,{})
        prior=np.array([prior_map.get(n,total/len(names)) for n in names])
        forces=self._closest_feasible(xs,total,prior)
        if forces is None: return None
        self.previous[pid]={n:float(forces[i]) for i,n in enumerate(names)}
        arr=np.asarray(vertices); lo,hi=arr.min(axis=0),arr.max(axis=0)
        span=float(np.mean((hi-lo)/total))
        certainty="constrained" if span<=.20 else "ambiguous"
        vertical={n:float(forces[i]) for i,n in enumerate(names)}
        shares={n:100*v/total for n,v in vertical.items()}
        ranges={n:(100*lo[i]/total,100*hi[i]/total) for i,n in enumerate(names)}
        return {"status":"2D static estimate","certainty":certainty,"cog":com,
                "vertical_n":vertical,"shares_pct":shares,"ranges_pct":ranges,
                "positions":positions}
    def draw_body_diagram(self,frame,kps,load_data,color):
        if not load_data: return
        for name,pos in load_data["positions"].items():
            share=load_data["shares_pct"][name]; lo,hi=load_data["ranges_pct"][name]
            px,py=int(pos[0]),int(pos[1])
            col=(50,200,220) if load_data["certainty"]=="constrained" else (80,150,220)
            cv2.circle(frame,(px,py),10,col,-1,cv2.LINE_AA)
            cv2.putText(frame,f"{share:.0f}% [{lo:.0f}-{hi:.0f}]",(px+12,py+4),
                        cv2.FONT_HERSHEY_SIMPLEX,.32,col,1,cv2.LINE_AA)
        cx,cy=map(int,load_data["cog"]); cv2.circle(frame,(cx,cy),5,(255,255,0),-1,cv2.LINE_AA)

# ═══════════════════════════════════════════════════════
#  SCORING SYSTEM
# ═══════════════════════════════════════════════════════

class ClimbScorer:
    """Four independent coaching metrics; load estimates are never included."""
    def __init__(self,frame_height):
        self.H=frame_height; self.last_time=None
        self.arm_eligible=self.arm_penalty=0.0
        self.arm_bent_run={"left":0.0,"right":0.0}
        self.support_total=self.support_sum=0.0
        self.placements=self.adjustments=0; self.foot_initialized=set(); self.last_release={}
        self.leg_opportunities=self.leg_drives=0; self.prev_knee={}
        self.prev_com=None; self.upward_active=False; self.upward_last=None
        self.upward_had_leg_drive=False
        self.observed_time=self.pause_run=self.pause_excess=0.0
        self.attempt_started=False; self.first_com=None
        self.com_path=[]; self.com_segments=[self.com_path]

    @staticmethod
    def _support_quality(kps,active_contacts,com,body_h):
        """Score confirmed base-of-support geometry in the camera plane.

        Two contacts receive partial credit. With three or four contacts, both
        polygon area and the CoM position affect the result. This is a 2D
        coaching proxy, not a claim of full 3D mechanical stability.
        """
        points=[]
        for joint in active_contacts:
            raw=kps.get(joint)
            if raw and len(raw)>=3 and raw[2]>=GRIP_KP_CONF:
                points.append((float(raw[0]),float(raw[1])))
        if len(points)<2: return 0.0
        if len(points)==2: return .5
        poly=cv2.convexHull(np.asarray(points,dtype=np.float32))
        area=float(cv2.contourArea(poly))
        area_factor=max(0.0,min(1.0,area/max(.08*body_h*body_h,1.0)))
        signed=float(cv2.pointPolygonTest(poly,(float(com[0]),float(com[1])),True))
        com_factor=1.0 if signed>=0 else max(0.0,1.0+signed/max(.20*body_h,1.0))
        return .4+.3*area_factor+.3*com_factor

    def update(self,source_time,kps,box,arm_status,support_status,phase_info,
               contact_events,active_contacts,load_data=None):
        dt=0.0 if self.last_time is None else max(0.0,min(.25,source_time-self.last_time))
        self.last_time=source_time
        com=phase_info.get("com"); body_h=max(1.0,phase_info.get("body_h",box[3]-box[1]))
        previous_com=self.prev_com
        if com is not None:
            if self.first_com is None: self.first_com=com
            if not self.com_path:
                self.com_path.append((source_time,com))
            else:
                jump=dist(self.com_path[-1][1],com)
                if jump>body_h*.75:
                    self.com_path=[]; self.com_segments.append(self.com_path)
                    self.com_path.append((source_time,com))
                elif jump>body_h*.02:
                    self.com_path.append((source_time,com))
            if dist(self.first_com,com)>body_h*.05: self.attempt_started=True
        state=phase_info.get("state","unknown")
        if state=="static" and dt>0:
            if support_status in ("good","ok","warn"):
                self.support_total+=dt
                q=self._support_quality(kps,active_contacts,com,body_h)
                self.support_sum+=dt*q
            for side,wrist in (("left","left_wrist"),("right","right_wrist")):
                ang=angle3(kp(kps,f"{side}_shoulder"),kp(kps,f"{side}_elbow"),kp(kps,wrist))
                if wrist not in active_contacts or ang is None:
                    self.arm_bent_run[side]=0.0; continue
                self.arm_eligible+=dt
                if ang<ARM_STRAIGHT_ANGLE:
                    self.arm_bent_run[side]+=dt
                    if self.arm_bent_run[side]>ARM_BEND_GRACE_SEC: self.arm_penalty+=dt
                else: self.arm_bent_run[side]=0.0
        for event in contact_events:
            joint=event["joint"]
            if joint not in ("left_ankle","right_ankle"): continue
            if event["type"]=="released":
                self.last_release[joint]=(source_time,event["hold"])
            elif event["type"]=="acquired":
                if joint not in self.foot_initialized:
                    self.foot_initialized.add(joint); continue
                self.placements+=1; previous=self.last_release.get(joint)
                if previous and source_time-previous[0]<=FOOT_ADJUST_WINDOW_SEC and previous[1]==event["hold"]:
                    self.adjustments+=1
        knee_now={}
        for side,ankle in (("left","left_ankle"),("right","right_ankle")):
            knee_now[side]=angle3(kp(kps,f"{side}_hip"),kp(kps,f"{side}_knee"),kp(kps,ankle))
        upward=(dt>0 and state=="moving" and com is not None and previous_com is not None and
                (previous_com[1]-com[1])/body_h/dt>.08)
        if upward:
            supporting=[s for s,a in (("left","left_ankle"),("right","right_ankle")) if a in active_contacts]
            if supporting:
                if (not self.upward_active or self.upward_last is None or
                        source_time-self.upward_last>.30):
                    self.leg_opportunities+=1
                    self.upward_active=True; self.upward_had_leg_drive=False
                leg_drive=any(knee_now[s] is not None and s in self.prev_knee and
                              (knee_now[s]-self.prev_knee[s])/dt>15.0 for s in supporting)
                if leg_drive and not self.upward_had_leg_drive:
                    self.leg_drives+=1
                    self.upward_had_leg_drive=True
                self.upward_last=source_time
        elif (self.upward_active and self.upward_last is not None and
              source_time-self.upward_last>.30):
            self.upward_active=False; self.upward_had_leg_drive=False
        self.prev_knee={s:a for s,a in knee_now.items() if a is not None}
        self.prev_com=com
        if self.attempt_started and dt>0 and state in ("static","moving"):
            self.observed_time+=dt
            if state=="static":
                self.pause_run+=dt
                if self.pause_run>FLUENCY_PAUSE_GRACE_SEC: self.pause_excess+=dt
            else: self.pause_run=0.0
    def _geometric_entropy(self):
        segment=max(self.com_segments,key=len)
        if len(segment)<10: return None
        pts=[p for _,p in segment]; smooth=[]
        for i in range(len(pts)):
            seg=pts[max(0,i-2):min(len(pts),i+3)]
            smooth.append((sum(p[0] for p in seg)/len(seg),sum(p[1] for p in seg)/len(seg)))
        lengths=[dist(smooth[i-1],smooth[i]) for i in range(1,len(smooth))]
        lm=sum(d for d in lengths if d<=150)
        if lm<=1: return None
        hull=cv2.convexHull(np.asarray(smooth,dtype=np.float32)); ch=cv2.arcLength(hull,True)
        return None if ch<=1 else max(0.0,math.log(2.0*lm/ch))
    def compute(self):
        arm=None if self.arm_eligible<1.0 else max(0.0,100.0*(1.0-self.arm_penalty/self.arm_eligible))
        support=None if self.support_total<1.0 else 100.0*self.support_sum/self.support_total
        foot=[]
        if self.placements>=2:
            foot.append(max(0.0,100.0*(1.0-self.adjustments/max(1,self.placements))))
        if self.leg_opportunities>=3:
            foot.append(100.0*self.leg_drives/self.leg_opportunities)
        footwork=None if not foot else sum(foot)/len(foot)
        fluency=None if self.observed_time<5.0 else max(0.0,100.0*(1.0-self.pause_excess/self.observed_time))
        raw={"arm":arm,"support":support,"footwork":footwork,"fluency":fluency}
        overall=None if any(v is None for v in raw.values()) else round(sum(raw.values())/4)
        bd={k:(None if v is None else round(v)) for k,v in raw.items()}
        bd.update({"triangle":bd["support"],"ge_raw":self._geometric_entropy(),
                   "placements":self.placements,"adjustments":self.adjustments,
                   "leg_drives":self.leg_drives,"leg_opportunities":self.leg_opportunities,
                   "coverage":{"arm_support_sec":round(self.arm_eligible,1),
                               "support_sec":round(self.support_total,1),
                               "observed_sec":round(self.observed_time,1)}})
        return overall,bd
    def draw_live(self,frame,pid,id_color,badge_index=0):
        overall,_=self.compute(); draw_score_badge(frame,pid,overall,id_color,badge_index)
    def draw_final(self,frame,pid,id_color,card_index=0,total_cards=1):
        overall,bd=self.compute(); H,W=frame.shape[:2]
        cw,ch=min(420,W-30),min(410,H-30); x1,y1=(W-cw)//2,(H-ch)//2
        blend_rect(frame,x1,y1,x1+cw,y1+ch,(12,12,18),.94)
        cv2.rectangle(frame,(x1,y1),(x1+cw,y1+ch),(60,60,70),1)
        f=cv2.FONT_HERSHEY_SIMPLEX; total="--" if overall is None else str(overall)
        cv2.putText(frame,f"CLIMBER {pid}  TECHNIQUE {total}",(x1+18,y1+35),f,.72,id_color,2)
        labels=(("Arm use","arm"),("Support / triangle","support"),
                ("Footwork / legs","footwork"),("Fluency","fluency"))
        y=y1+82
        for label,key in labels:
            value=bd[key]; text="N/A" if value is None else str(value)
            cv2.putText(frame,label,(x1+20,y),f,.48,(210,210,210),1)
            cv2.putText(frame,text,(x1+cw-65,y),f,.55,
                        (180,200,255) if value is not None else (110,110,110),2)
            y+=52
        cv2.putText(frame,f"Placements {bd['placements']}  readjustments {bd['adjustments']}",
                    (x1+20,y+10),f,.38,(180,180,180),1)
        ge=bd["ge_raw"]; ge_text="N/A" if ge is None else f"{ge:.3f}"
        cv2.putText(frame,f"GE diagnostic: {ge_text}",(x1+20,y+38),f,.38,(150,150,170),1)
        cv2.putText(frame,"Estimated load is separate and is not graded.",
                    (x1+20,y1+ch-20),f,.34,(130,180,210),1)

# ═══════════════════════════════════════════════════════
#  STATS PANEL + LEGEND
# ═══════════════════════════════════════════════════════

def draw_stats_panel(frame,pid,arm_status,load_data,tri_status,path_len,
                     color,force_analyzer,panel_index=0,score_data=None):
    PW,PH,GAP=210,250,8; W=frame.shape[1]
    px=W-PW-4; py=4+panel_index*(PH+GAP)
    blend_rect(frame,px,py,px+PW,py+PH,(15,15,15),.75)
    cv2.rectangle(frame,(px,py),(px+3,py+PH),color,-1)
    f=cv2.FONT_HERSHEY_SIMPLEX; x,y=px+8,py+15
    cv2.putText(frame,f"Climber ID {pid}",(x,y),f,.38,color,1); y+=17
    tri_c={"good":COL_TRI_GOOD,"ok":COL_TRI_OK,"warn":COL_TRI_WARN,"unknown":(120,120,120)}
    tri_l={"good":"3+ contacts","ok":"2 contacts","warn":"<2 contacts","unknown":"unknown"}
    cv2.putText(frame,f"Support: {tri_l.get(tri_status,'unknown')}",(x,y),f,.32,tri_c.get(tri_status,(120,120,120)),1); y+=16
    for side in ("left","right"):
        state=arm_status.get(side,"unknown")
        col=(50,50,255) if state=="bent" else (50,220,50) if state=="straight" else (120,120,120)
        cv2.putText(frame,f"{side[0].upper()} arm: {state}",(x,y),f,.30,col,1); y+=14
    if score_data:
        cv2.line(frame,(x,y),(px+PW-6,y),(60,60,60),1); y+=12
        for label,key in (("Arm","arm"),("Support","support"),("Foot/leg","footwork"),("Flow","fluency")):
            value=score_data.get(key); text="N/A" if value is None else str(value)
            cv2.putText(frame,f"{label}: {text}",(x,y),f,.30,(190,190,205),1); y+=13
    cv2.line(frame,(x,y),(px+PW-6,y),(60,60,60),1); y+=11
    cv2.putText(frame,"Estimated vertical support:",(x,y),f,.29,(180,180,180),1); y+=13
    if load_data:
        for limb,label in (("left_hand","LH"),("right_hand","RH"),("left_foot","LF"),("right_foot","RF")):
            if limb in load_data["shares_pct"]:
                share=load_data["shares_pct"][limb]; lo,hi=load_data["ranges_pct"][limb]
                cv2.putText(frame,f"{label}: {share:.0f}%  range {lo:.0f}-{hi:.0f}",
                            (x,y),f,.27,(80,180,220),1); y+=12
        cv2.putText(frame,f"solution: {load_data['certainty']}",(x,y),f,.27,(130,150,180),1); y+=12
    else:
        cv2.putText(frame,"N/A (needs static contacts)",(x,y),f,.27,(100,100,100),1); y+=13
    cv2.putText(frame,f"Path: {path_len:.0f}px",(x,min(py+PH-8,y+5)),f,.30,(200,200,200),1)

def draw_legend(frame):
    H = frame.shape[0]
    items = [(COL_HOLD_IDLE,"Detected hold"),(COL_HOLD_HAND,"Hand hold"),(COL_HOLD_FOOT,"Foot hold")]
    y = H - 80
    for col,lbl in items:
        cv2.circle(frame,(25,y),7,col,-1,cv2.LINE_AA)
        cv2.putText(frame,lbl,(38,y+5),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1); y+=22

# ═══════════════════════════════════════════════════════
#  FOLLOW ZOOM + EMERGENCY ZOOM
# ═══════════════════════════════════════════════════════

class FollowZoomTracker:
    def __init__(self):
        self.detected_frames = defaultdict(int)
        self.follow_box = {}
    def is_following(self, pid):
        return self.detected_frames[pid] >= FOLLOW_MIN_FRAMES
    def update_seen(self, pid, hip_pos, box_size=None):
        self.detected_frames[pid] += 1
        if hip_pos:
            cx, cy = hip_pos
            if box_size:
                bw, bh = box_size
                half_w = max(FOLLOW_CROP_MIN_HALF, int(bw*FOLLOW_CROP_PADDING))
                half_h = max(FOLLOW_CROP_MIN_HALF, int(bh*FOLLOW_CROP_PADDING))
            else:
                half_w, half_h = max(FOLLOW_CROP_MIN_HALF,150), max(FOLLOW_CROP_MIN_HALF,200)
            self.follow_box[pid] = (cx-half_w, cy-half_h, cx+half_w, cy+half_h)
    def get_crop_box(self, pid, W, H):
        if pid not in self.follow_box: return None
        x1,y1,x2,y2 = self.follow_box[pid]
        return (max(0,int(x1)), max(0,int(y1)), min(W,int(x2)), min(H,int(y2)))
    def detect_in_crop(self, pid, track_model, frame, W, H, device=None):
        box = self.get_crop_box(pid, W, H)
        if box is None: return []
        cx1,cy1,cx2,cy2 = box
        if cx2-cx1 < 50 or cy2-cy1 < 50: return []
        crop = frame[cy1:cy2, cx1:cx2]
        results = track_model(crop, verbose=False, classes=[PERSON_CLASS], device=device)
        found = []
        for r in results:
            if r.boxes is None: continue
            for b in r.boxes:
                conf = float(b.conf[0])
                if conf < 0.15: continue
                bx1,by1,bx2,by2 = b.xyxy[0].cpu().numpy()
                found.append(((bx1+cx1,by1+cy1,bx2+cx1,by2+cy1), conf))
        return found

class EmergencyZoomTracker:
    def __init__(self, fps):
        self.fps = fps; self.lost_frames = defaultdict(int)
    def update_seen(self, pid): self.lost_frames[pid] = 0
    def update_lost(self, pid): self.lost_frames[pid] += 1
    def should_zoom(self, pid): return self.lost_frames[pid] >= LOST_FRAMES_TRIGGER
    def zoom_crop_size(self, pid):
        extra = max(0, self.lost_frames[pid]-LOST_FRAMES_TRIGGER)
        extra_px = int((extra/max(self.fps,1))*ZOOM_EXPAND_PER_SEC)
        return (ZOOM_CROP_HALF_W+extra_px, ZOOM_CROP_HALF_H+extra_px)
    def try_recover(self, pid, track_model, frame, W, H, last_path_pos, device=None):
        if last_path_pos is None: return None
        cx, cy = last_path_pos
        hw, hh = self.zoom_crop_size(pid)
        zx1,zy1 = max(0,int(cx-hw)), max(0,int(cy-hh))
        zx2,zy2 = min(W,int(cx+hw)), min(H,int(cy+hh))
        if zx2-zx1 < 50 or zy2-zy1 < 50: return None
        zoom_crop = frame[zy1:zy2, zx1:zx2]
        results = track_model(zoom_crop, verbose=False, classes=[PERSON_CLASS], device=device)
        best_box, best_conf = None, 0.0
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.15: continue
                if conf > best_conf:
                    best_conf = conf
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                    best_box = np.array([x1+zx1,y1+zy1,x2+zx1,y2+zy1])
        return best_box

# ═══════════════════════════════════════════════════════
#  HOLD DETECTION + GRIP TRACKER
# ═══════════════════════════════════════════════════════

def detect_holds(hold_model, cap, W, H, device=None):
    print(f"  Detecting holds on first {HOLD_DETECT_FRAMES} frames...")
    raw = []
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    for _ in range(HOLD_DETECT_FRAMES):
        ret, frame = cap.read()
        if not ret: break
        results = hold_model(frame, verbose=False, conf=HOLD_CONF, device=device)
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                raw.append({"center":((x1+x2)/2,(y1+y2)/2),
                    "box":(int(x1),int(y1),int(x2),int(y2)),
                    "class":int(box.cls[0]), "conf":float(box.conf[0])})
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    raw = sorted(raw, key=lambda h: h["conf"], reverse=True)
    holds = []
    for h in raw:
        if not any(dist(h["center"],k["center"]) < HOLD_NMS_DIST for k in holds):
            holds.append(h)
    grips = sum(1 for h in holds if h["class"]==0)
    volumes = sum(1 for h in holds if h["class"]==1)
    print(f"  Found {len(holds)} holds ({grips} grips, {volumes} volumes)")
    return holds

class GripTracker:
    """Per-person, per-limb current contacts plus persistent session history."""
    LIMB_MAP={"left_wrist":"hand","right_wrist":"hand",
              "left_ankle":"foot","right_ankle":"foot"}
    def __init__(self,holds):
        self.holds=holds; self.touched=[set() for _ in holds]
        self.states=defaultdict(dict); self._static=None; self._mask=None
    @staticmethod
    def _rect_distance(point,box):
        x,y=point; x1,y1,x2,y2=box
        return math.hypot(max(x1-x,0.0,x-x2),max(y1-y,0.0,y-y2))
    def _state(self,pid,joint):
        if joint not in self.states[pid]:
            self.states[pid][joint]={"status":"unknown","candidate":None,
                "candidate_since":None,"current":None,"away_since":None,"last_confident":None}
        return self.states[pid][joint]
    def update(self,all_kps,source_time=None,boxes=None):
        if source_time is None: raise ValueError("GripTracker.update requires source-video time")
        boxes=boxes or {}; events=defaultdict(list)
        for pid,kps in all_kps.items():
            box=boxes.get(pid); body_h=max(1.0,float(box[3]-box[1])) if box is not None else 300.0
            radius=min(65.0,max(10.0,CONTACT_RADIUS_BODY*body_h))
            for joint,limb_type in self.LIMB_MAP.items():
                state=self._state(pid,joint); raw=kps.get(joint)
                if raw is None or len(raw)<3 or raw[2]<GRIP_KP_CONF:
                    state["candidate"]=None; state["candidate_since"]=None
                    last=state["last_confident"]
                    if last is None or source_time-last>CONTACT_UNKNOWN_SEC: state["status"]="unknown"
                    continue
                state["last_confident"]=source_time; point=(raw[0],raw[1])
                distances=[self._rect_distance(point,h["box"]) for h in self.holds]
                best=int(np.argmin(distances)) if distances else None
                near=best is not None and distances[best]<=radius
                if near:
                    state["away_since"]=None
                    if state["current"]==best:
                        state["status"]="confirmed"; state["candidate"]=None; state["candidate_since"]=None
                    elif state["candidate"]!=best:
                        state["candidate"]=best; state["candidate_since"]=source_time; state["status"]="uncertain"
                    elif source_time-state["candidate_since"]>=CONTACT_CONFIRM_SEC:
                        old=state["current"]
                        if old is not None:
                            events[pid].append({"type":"released","joint":joint,"hold":old,"time":source_time})
                        state["current"]=best; state["status"]="confirmed"
                        state["candidate"]=None; state["candidate_since"]=None
                        self.touched[best].add(limb_type)
                        events[pid].append({"type":"acquired","joint":joint,"hold":best,"time":source_time})
                else:
                    state["candidate"]=None; state["candidate_since"]=None
                    if state["current"] is None: state["status"]="free"
                    elif state["away_since"] is None:
                        state["away_since"]=source_time; state["status"]="uncertain"
                    elif source_time-state["away_since"]>=CONTACT_RELEASE_SEC:
                        old=state["current"]; state["current"]=None
                        state["away_since"]=None; state["status"]="free"
                        events[pid].append({"type":"released","joint":joint,"hold":old,"time":source_time})
        return events
    def age_missing(self,pids,source_time):
        """Expire contacts when a whole pose observation is unavailable."""
        for pid in pids:
            for state in self.states.get(pid,{}).values():
                last=state["last_confident"]
                if last is None or source_time-last>CONTACT_UNKNOWN_SEC:
                    state["status"]="unknown"; state["current"]=None
                    state["candidate"]=None; state["candidate_since"]=None
                    state["away_since"]=None
    def current_contacts(self,pid):
        return {j:s["current"] for j,s in self.states.get(pid,{}).items()
                if s["status"]=="confirmed" and s["current"] is not None}
    def support_geometry(self,pid,kps):
        contacts=self.current_contacts(pid); points=[]
        for joint in self.LIMB_MAP:
            if joint in contacts:
                raw=kps.get(joint)
                if raw and raw[2]>=GRIP_KP_CONF: points.append((int(raw[0]),int(raw[1]),joint))
        states=list(self.states.get(pid,{}).values())
        evidence_known=(len(states)==4 and all(
            s["status"] in ("confirmed","free") for s in states))
        if not evidence_known: return "unknown",points,points
        if len(points)>=3:
            triples=[(points[i],points[j],points[k]) for i in range(len(points))
                     for j in range(i+1,len(points)) for k in range(j+1,len(points))]
            best=max(triples,key=lambda t:tri_area(t[0],t[1],t[2]))
            return "good",points,list(best)
        if len(points)==2: return "ok",points,points
        return "warn",points,points
    def draw(self,frame):
        draw_holds_snapshot(frame,self,[set(s) for s in self.touched])
    def session_report(self):
        return {"total_holds":len(self.holds),
                "hand_holds":sum(1 for t in self.touched if "hand" in t),
                "foot_holds":sum(1 for t in self.touched if "foot" in t)}

# ═══════════════════════════════════════════════════════
#  THREADED VIDEO WRITER
# ═══════════════════════════════════════════════════════
class ThreadedWriter:
    """Offload video encoding to a background thread. On RK3588 the CPU encode
    costs ~23ms/frame; running it on a worker thread takes it off the main loop's
    critical path so inference and drawing aren't blocked waiting on it.
    The writer is a pure sink (no shared tracker state), so this is race-free."""
    def __init__(self, path, fourcc, fps, size, maxsize=8):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        # Bounded queue gives backpressure: if encoding falls behind, write()
        # blocks instead of growing memory without limit.
        self.q = queue_module.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            self.writer.write(item)

    def write(self, frame):
        # Copy so the main loop can keep mutating its frame buffer safely.
        self.q.put(frame.copy())

    def isOpened(self):
        return self.writer.isOpened()

    def release(self):
        self.q.put(None)        # sentinel → worker drains remaining frames then exits
        self.thread.join()
        self.writer.release()


# ═══════════════════════════════════════════════════════
#  SNAPSHOT-FED DRAWERS (for THREADED_DRAW)
#  These draw ONLY from immutable data passed in — never from live tracker/scorer
#  state — so they are safe to call on the draw worker thread.
# ═══════════════════════════════════════════════════════
def draw_path_points(frame, pts, color):
    MAX_SEGMENT = 150
    for i in range(1, len(pts)):
        if dist(pts[i-1], pts[i]) <= MAX_SEGMENT:
            cv2.line(frame, pts[i-1], pts[i], color, 3, cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 6, COL_PATH, -1, cv2.LINE_AA)
        cv2.circle(frame, pts[-1], 6, (255,255,255), 1, cv2.LINE_AA)

def draw_score_badge(frame,pid,overall,id_color,badge_index=0):
    W=frame.shape[1]; bx=W-78-badge_index*72; by=frame.shape[0]-52
    blend_rect(frame,bx,by,bx+66,by+48,(15,15,15),.72)
    cv2.rectangle(frame,(bx,by),(bx+66,by+2),id_color,-1)
    f=cv2.FONT_HERSHEY_SIMPLEX
    col=(120,120,120) if overall is None else (50,220,50) if overall>=75 else (180,180,50) if overall>=50 else (50,50,255)
    cv2.putText(frame,f"ID{pid}",(bx+4,by+16),f,.35,id_color,1)
    cv2.putText(frame,"--" if overall is None else str(overall),(bx+10,by+40),f,.7,col,2)

def draw_lock_overlay(frame, kps, arm_status, frames_side, thresh):
    for side, el_name in [("left","left_elbow"),("right","right_elbow")]:
        el = kp(kps, el_name)
        if el is None: continue
        ex, ey = int(el[0]), int(el[1])
        f = frames_side.get(side, 0)
        if f > 0:
            pct = min(f/thresh, 1.0)
            warning = pct >= 1.0
            color = (50,50,255) if warning else (50,180,255)
            radius = 26 if warning else int(14+pct*6)
            thickness = 3 if warning else 2
            cv2.ellipse(frame, (ex,ey), (radius,radius), -90, 0, int(pct*360), color, thickness, cv2.LINE_AA)
            if warning:
                cv2.putText(frame, "Stretch arm!", (ex+radius+5, ey+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,255), 2, cv2.LINE_AA)
    left_bent = arm_status.get("left","unknown")=="bent"
    right_bent = arm_status.get("right","unknown")=="bent"
    if arm_status.get("warning") and left_bent and right_bent:
        H, W = frame.shape[0], frame.shape[1]
        cv2.rectangle(frame, (0,0), (W,H), (0,0,255), 8)
        cv2.putText(frame, "BOTH ARMS BENT — STRAIGHTEN UP!",
                    (W//2-220,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

def draw_holds_snapshot(frame, grip_tracker, touched):
    # Build the static idle-hold layer once (immutable after; worker is sole accessor
    # in threaded mode). 'touched' is an immutable COPY passed in the packet.
    if grip_tracker._static is None:
        layer = np.zeros_like(frame)
        for hold in grip_tracker.holds:
            if hold["class"] == 1:
                x1,y1,x2,y2 = hold["box"]
                cv2.rectangle(layer,(x1,y1),(x2,y2),(120,120,60),1)
            else:
                cx, cy = int(hold["center"][0]), int(hold["center"][1])
                cv2.circle(layer,(cx,cy),10,COL_HOLD_IDLE,1,cv2.LINE_AA)
        grip_tracker._static = layer
        grip_tracker._mask = layer.any(axis=2)
    frame[grip_tracker._mask] = grip_tracker._static[grip_tracker._mask]
    for idx, hold in enumerate(grip_tracker.holds):
        if hold["class"] == 1: continue
        t = touched[idx]
        if not t: continue
        cx, cy = int(hold["center"][0]), int(hold["center"][1])
        if "hand" in t and "foot" in t:
            cv2.circle(frame,(cx,cy),16,COL_HOLD_FOOT,-1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),10,COL_HOLD_HAND,-1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),16,(255,255,255),1,cv2.LINE_AA)
        elif "hand" in t:
            cv2.circle(frame,(cx,cy),14,COL_HOLD_HAND,-1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),14,(255,255,255),1,cv2.LINE_AA)
        elif "foot" in t:
            cv2.circle(frame,(cx,cy),14,COL_HOLD_FOOT,-1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),14,(255,255,255),1,cv2.LINE_AA)


def draw_support_geometry(frame, points, triangle, status):
    """Draw confirmed contact geometry only; never extrapolate a missing limb."""
    col = {"good": COL_TRI_GOOD, "ok": COL_TRI_OK,
           "warn": COL_TRI_WARN}.get(status, (120,120,120))
    if len(triangle) >= 3:
        poly = np.asarray([(p[0],p[1]) for p in triangle], dtype=np.int32)
        blend_poly(frame, poly, col, 0.13)
        cv2.polylines(frame, [poly], True, col, 2, cv2.LINE_AA)
    elif len(triangle) == 2:
        cv2.line(frame, triangle[0][:2], triangle[1][:2], col, 2, cv2.LINE_AA)
    for x,y,_ in points:
        cv2.circle(frame, (x,y), 7, col, 2, cv2.LINE_AA)


def render_overlays(frame, packet):
    """Draw a whole frame from an immutable snapshot packet. Safe on any thread."""
    fa = packet["force_analyzer"]
    for it in packet["items"]:
        pid = it["pid"]; color = it["color"]
        draw_person_box(frame, it["box"], pid, color)
        if it["pose_draw_ok"]:
            kps = it["kps"]
            draw_skeleton(frame, kps, color)
            draw_path_points(frame, it["path_pts"], path_color(pid))
            if DRAW_WEIGHT_TRIANGLE:
                draw_support_geometry(frame, it["support_points"],
                                      it["support_triangle"], it["tri_status"])
            draw_lock_overlay(frame, kps, it["arm_status"], it["lock_frames"], packet["lock_thresh"])
            if fa and it["load_data"] and DRAW_WEIGHT_DIAGRAM:
                fa.draw_body_diagram(frame, kps, it["load_data"], color)
        else:
            draw_path_points(frame, it["path_pts"], path_color(pid))
        draw_stats_panel(frame, pid, it["arm_status"], it["load_data"], it["tri_status"],
                         it["path_len"], color, fa, panel_index=it["panel_index"],
                         score_data=it["score_data"])
        draw_score_badge(frame, pid, it["score_overall"], color, it["badge_index"])

    for it in packet["lost_items"]:
        pid = it["pid"]; col = it["color"]
        draw_path_points(frame, it["path_pts"], path_color(pid))
        if it["last_stats"] is not None:
            _, _, _, path_l = it["last_stats"]
            draw_stats_panel(frame, pid, {}, None, "unknown", path_l, col, fa,
                             panel_index=it["panel_index"])
        draw_score_badge(frame, pid, it["score_overall"], col, it["panel_index"])

    if packet["grip_tracker"] is not None:
        draw_holds_snapshot(frame, packet["grip_tracker"], packet["grip_touched"])
        draw_legend(frame)

    draw_info_panel(frame, packet["frame_num"], packet["fps"], packet["sorted_pids"], packet["device_name"])


class DrawWorker:
    """Runs render_overlays + video write on a background thread so the main thread's
    NPU inference overlaps with the CPU drawing of the previous frame. The packet is
    immutable, so no tracker/scorer state is shared across threads."""
    def __init__(self, writer, maxsize=3):
        self.writer = writer
        self.q = queue_module.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            frame, packet = item
            render_overlays(frame, packet)
            if self.writer:
                self.writer.write(frame)

    def submit(self, frame, packet):
        self.q.put((frame, packet))

    def release(self):
        self.q.put(None)
        self.thread.join()


# ═══════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════

def run(source, body_weight_kg=None, output_path=None):
    print("\n" + "="*55)
    print("  Climbing AI — final single-climber RK3588 pipeline")
    print("="*55)

    device_name = "RK3588 NPU"

    print("\nPreparing RKNN models...")
    track_model = None
    pose_model  = None
    hold_model  = None

    print(f"  Body weight  : {body_weight_kg} kg" if body_weight_kg
          else "  Body weight  : not set — force analysis disabled")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open source: {source}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_live = source in (0, "0")
    print(f"\n  Source  : {'Live camera' if is_live else source}")
    print(f"  Size    : {W}x{H} @ {fps:.1f}fps")

    writer = None
    draw_worker = None
    if output_path:
        if THREADED_DRAW:
            # Threaded draw: a plain writer owned by the draw worker (which is itself
            # a background thread, so its writes don't block the main loop).
            raw_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            draw_worker = DrawWorker(raw_writer)
            print(f"  Saving  : {output_path}  [THREADED_DRAW on — no live preview]")
        else:
            writer = ThreadedWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            print(f"  Saving  : {output_path}")

    holds, grip_tracker = [], None
    if Path(HOLD_MODEL).exists():
        print("\nLoading hold model...")
        hold_model = RKNNHoldModel(HOLD_MODEL)
        print("\nDetecting holds...")
        holds = detect_holds(hold_model, cap, W, H)
        grip_tracker = GripTracker(holds)
        print("  Releasing hold model to free NPU memory...")
        hold_model.release()
        hold_model = None
    else:
        print("  Hold model: NOT FOUND — hold detection disabled")

    print("\nLoading person + pose models...")
    track_model = RKNNDetectModel(TRACK_MODEL)
    pose_model  = RKNNPoseModel(POSE_MODEL)

    path_tracker   = PathTracker()
    phase_tracker  = MotionPhaseTracker()
    phase_cache    = {}
    lock_detector  = LockArmDetector(fps)
    force_analyzer = LimbLoadAnalyzer(body_weight_kg) if body_weight_kg else None
    scorers        = {}
    last_colors    = {}
    last_stats     = {}
    zoom_tracker   = EmergencyZoomTracker(fps)
    follow_tracker = FollowZoomTracker()

    # ── Split-schedule caches ──
    # box_cache: {pid: bxyxy}  — refreshed on detect frames (slow-changing)
    # pose_cache: {pid: kps}   — refreshed on pose frames (fast-changing)
    # pose_age: {pid: frames since last fresh pose}
    box_cache = {}
    pose_cache = {}
    pose_age = defaultdict(lambda: 999)
    box_miss_count = 0
    recovery_miss = defaultdict(int)  # per-climber missed-detect counter for recovery throttle

    def run_pose_on_box(bxyxy):
        """Crop around a box and return best keypoints (frame coords) or {}."""
        x1, y1, x2, y2 = map(int, bxyxy)
        px1 = max(0, x1 - int((x2-x1)*CROP_PAD))
        py1 = max(0, y1 - int((y2-y1)*CROP_PAD))
        px2 = min(W, x2 + int((x2-x1)*CROP_PAD))
        py2 = min(H, y2 + int((y2-y1)*CROP_PAD))
        crop = frame[py1:py2, px1:px2]
        if crop.size == 0:
            return {}
        pose_results = pose_model(crop, verbose=False)
        best_kps, best_sum = None, -1
        for pr in pose_results:
            if pr.keypoints is None:
                continue
            for pi in range(len(pr.keypoints.xy)):
                s = float(pr.keypoints.conf[pi].sum())
                if s > best_sum:
                    best_sum = s
                    best_kps = extract_kps(
                        pr.keypoints.xy[pi], pr.keypoints.conf[pi], ox=px1, oy=py1)
        return best_kps if best_kps else {}

    print("\nStarting... (press Q to quit)\n")
    frame_num = 0; fps_cnt = 0; fps_timer = time.time(); disp_fps = 0.0
    live_clock_start = time.monotonic()
    # ── Stage timers (read / inference / draw / write) to find the real bottleneck ──
    t_read = t_infer = t_draw = t_write = 0.0
    detect_interval=max(1,int(DETECT_EVERY)); pose_interval=max(1,int(POSE_EVERY))

    while True:
        _t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1
        t_read += time.perf_counter() - _t0
        _t1 = time.perf_counter()
        _calls_at_start = _INFER_CALLS   # to count NPU calls this frame

        # With the default intervals, detect and pose run on alternating frames so
        # the main loop normally carries one NPU call. Frame 1 primes both caches.
        do_detect = (frame_num == 1) or ((frame_num-1) % detect_interval == 0)
        do_pose   = (frame_num == 1) or (frame_num % pose_interval == 0)

        # ════════════════════════════════════════════════
        #  STEP 1 — Person detection (sparse) → box_cache
        # ════════════════════════════════════════════════
        if do_detect:
            track_results = track_model.track(frame, classes=[PERSON_CLASS], persist=True)
            new_boxes = {}

            for result in track_results:
                if result.boxes is None or result.boxes.id is None:
                    continue
                for box in result.boxes:
                    pid   = int(box.id[0])
                    bxyxy = box.xyxy[0].cpu().numpy()
                    if not on_wall(bxyxy, H):
                        x1,y1,x2,y2 = map(int, bxyxy)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(60,60,60),1)
                        continue
                    new_boxes[pid] = bxyxy

            # Phase 2: follow-crop recovery for known climbers not seen full-frame
            for known_pid in list(scorers.keys()):
                if known_pid in new_boxes:
                    recovery_miss[known_pid] = 0   # detected normally → reset throttle
                    continue
                if not follow_tracker.is_following(known_pid): continue
                # Throttle: the cached follow box carries the climber for a few frames,
                # so only spend an extra detection every RECOVERY_EVERY missed frames.
                recovery_miss[known_pid] += 1
                if recovery_miss[known_pid] % RECOVERY_EVERY != 0:
                    continue
                found = follow_tracker.detect_in_crop(known_pid, track_model, frame, W, H)
                if not found: continue
                best_box, _ = max(found, key=lambda x: x[1])
                bxyxy = np.array(best_box)
                if not on_wall(bxyxy, H): continue
                new_boxes[known_pid] = bxyxy
                track_model.sync_single_slot(bxyxy)
                crop_box = follow_tracker.get_crop_box(known_pid, W, H)
                if crop_box:
                    col = path_color(known_pid)
                    cv2.rectangle(frame,(crop_box[0],crop_box[1]),(crop_box[2],crop_box[3]),col,1,cv2.LINE_AA)
                    cv2.putText(frame,"FOLLOW",(crop_box[0]+4,crop_box[1]+16),cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)

            # Actual detections this frame (NOT cached) — used for lost/recovery logic.
            actual_detected_pids = set(new_boxes.keys())

            if new_boxes:
                box_cache = new_boxes
                box_miss_count = 0
            else:
                box_miss_count += 1
                if box_miss_count > CACHE_MISS_TOL:
                    box_cache = {}
        else:
            # Non-detect frame: no fresh detections.
            actual_detected_pids = set()

        # ════════════════════════════════════════════════
        #  STEP 2 — Pose (frequent) on current boxes → pose_cache
        # ════════════════════════════════════════════════
        if do_pose and box_cache:
            # The detector exposes at most one slot, so pose always follows it.
            targets = sorted(box_cache.keys())[:1]
            for pid in targets:
                kps = run_pose_on_box(box_cache[pid])
                if kps:
                    pose_cache[pid] = kps
                    pose_age[pid] = 0
                else:
                    pose_age[pid] += 1
            # Age every climber not refreshed this round.
            for pid in list(pose_cache.keys()):
                if pid not in targets:
                    pose_age[pid] += 1
        else:
            for pid in list(pose_cache.keys()):
                pose_age[pid] += 1

        # ════════════════════════════════════════════════
        #  STEP 3 — Build wall_climbers (pid, box, kps) from caches
        # ════════════════════════════════════════════════
        _frame_infer_ms = (time.perf_counter() - _t1) * 1000.0
        _frame_calls = _INFER_CALLS - _calls_at_start
        if _frame_infer_ms > 62 and not is_live:
            print(f"    SLOW frame {frame_num}: infer={_frame_infer_ms:.0f}ms  "
                  f"npu_calls={_frame_calls}  detect={int(do_detect)} pose={int(do_pose)}")
        t_infer += time.perf_counter() - _t1
        _t2 = time.perf_counter()
        wall_climbers = []
        for pid, bxyxy in box_cache.items():
            kps = pose_cache.get(pid, {})
            wall_climbers.append((pid, bxyxy, kps))


        # ── Lost/recovery — ONLY on detect frames ──
        # On non-detect frames we only have cached boxes, so treating them as
        # "detections" would corrupt lost/recovery state.
        if do_detect:
            # ── Update zoom tracker (actual detections only) ──
            for kp_id in list(scorers.keys()):
                if kp_id in actual_detected_pids: zoom_tracker.update_seen(kp_id)
                else: zoom_tracker.update_lost(kp_id)

            # ── Emergency zoom recovery ──
            for kp_id in list(scorers.keys()):
                if kp_id in actual_detected_pids: continue
                if not zoom_tracker.should_zoom(kp_id): continue
                last_pos = path_tracker.history[kp_id][-1] if path_tracker.history[kp_id] else None
                if last_pos is None: continue

                recovered_box = zoom_tracker.try_recover(kp_id, track_model, frame, W, H, last_pos)
                if recovered_box is not None and on_wall(recovered_box, H):
                    best_kps = run_pose_on_box(recovered_box)
                    box_cache[kp_id] = recovered_box
                    track_model.sync_single_slot(recovered_box)
                    if best_kps:
                        pose_cache[kp_id] = best_kps
                        pose_age[kp_id] = 0
                    wall_climbers=[c for c in wall_climbers if c[0]!=kp_id]
                    wall_climbers.append((kp_id, recovered_box, best_kps))
                    zoom_tracker.update_seen(kp_id)

                    zx, zy = int(last_pos[0]), int(last_pos[1])
                    hw, hh = zoom_tracker.zoom_crop_size(kp_id)
                    cv2.rectangle(frame,(max(0,zx-hw),max(0,zy-hh)),(min(W,zx+hw),min(H,zy+hh)),(0,200,255),1,cv2.LINE_AA)
                    cv2.putText(frame,"ZOOM RECOVERY",(max(0,zx-hw)+4,max(0,zy-hh)+16),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,200,255),1)

        # ── Per-climber modules (the detector exposes only ID 1) ───────
        active_wall = wall_climbers
        sorted_pids = sorted([c[0] for c in active_wall])
        all_known_pids = sorted(set(list(scorers.keys()) + sorted_pids))

        # Update time-sensitive state exactly once per fresh pose. Cached draw poses
        # preserve the display but cannot confirm contacts, duration, or movement.
        source_time = (time.monotonic()-live_clock_start if is_live else
                       max(0.0,(frame_num-1)/max(fps,1e-6)))
        fresh_kps, fresh_boxes = {}, {}
        for pid, bxyxy, kps in active_wall:
            if pose_age.get(pid, 999) == 0 and pose_is_valid(kps):
                fresh_kps[pid] = kps
                fresh_boxes[pid] = bxyxy
        contact_events = defaultdict(list)
        if grip_tracker and fresh_kps:
            contact_events = grip_tracker.update(fresh_kps, source_time, fresh_boxes)
        if grip_tracker:
            contact_pids=set(grip_tracker.states)|set(scorers)|set(box_cache)
            grip_tracker.age_missing(contact_pids-set(fresh_kps),source_time)
        for pid, kps in fresh_kps.items():
            phase_cache[pid] = phase_tracker.update(
                pid, kps, fresh_boxes[pid], source_time)

        # ════════════════════════════════════════════════
        #  THREADED DRAW PATH — state updates here on the main thread, drawing on the
        #  worker. Mirrors the inline loop's STATE logic exactly, then snapshots
        #  everything the drawing needs into an immutable packet. The inline path
        #  below (the proven one) is used unchanged when THREADED_DRAW is off.
        # ════════════════════════════════════════════════
        if THREADED_DRAW and draw_worker is not None:
            render_items = []
            for pid, bxyxy, kps in wall_climbers:
                color = id_color(pid)
                badge_index = all_known_pids.index(pid) if pid in all_known_pids else 0
                panel_index = badge_index
                last_colors[pid] = color
                if pid not in scorers: scorers[pid] = ClimbScorer(H)

                path_len = path_tracker.path_length(pid)
                age = pose_age.get(pid, 999)
                valid = pose_is_valid(kps)
                pose_draw_ok  = (age <= POSE_DRAW_MAX_AGE) and valid
                pose_score_ok = (age <= POSE_SCORE_MAX_AGE) and valid
                arm_status, load_data, tri_status, _ = last_stats.get(
                    pid, ({}, None, "unknown", path_len))

                if pose_draw_ok:
                    if pose_score_ok:
                        path_tracker.update(pid, kps)
                        hip_pos = path_tracker.history[pid][-1] if path_tracker.history[pid] else None
                        if hip_pos:
                            box_size = (int(bxyxy[2]-bxyxy[0]), int(bxyxy[3]-bxyxy[1]))
                            follow_tracker.update_seen(pid, hip_pos, box_size=box_size)
                    path_len = path_tracker.path_length(pid)
                    if grip_tracker:
                        tri_status, support_points, support_triangle = \
                            grip_tracker.support_geometry(pid, kps)
                        active_contacts = set(grip_tracker.current_contacts(pid))
                    else:
                        tri_status, support_points, support_triangle = "unknown", [], []
                        active_contacts = set()
                    phase = phase_cache.get(pid, {"state":"unknown", "com":None,
                        "body_h":max(1.0,float(bxyxy[3]-bxyxy[1]))})
                    arm_status = lock_detector.update(
                        pid, kps, source_time, pose_score_ok, active_contacts,
                        phase.get("state", "unknown"))
                    if pose_score_ok:
                        load_data = (force_analyzer.analyze(
                            pid, kps, active_contacts, phase) if force_analyzer else None)
                        hip_pos = path_tracker.history[pid][-1] if path_tracker.history[pid] else None
                        scorers[pid].update(
                            source_time, kps, bxyxy, arm_status, tri_status, phase,
                            contact_events.get(pid, []), active_contacts, load_data)
                        last_stats[pid] = (arm_status, load_data, tri_status, path_len)

                overall, _bd = scorers[pid].compute()
                lock_frames_copy = dict(lock_detector.frames.get(pid, {}))
                panel_arm = arm_status if pose_draw_ok else {}
                panel_load = load_data if pose_draw_ok else None
                panel_tri = tri_status if pose_draw_ok else "unknown"
                render_items.append({
                    "pid": pid,
                    "box": bxyxy.copy() if hasattr(bxyxy, "copy") else bxyxy,
                    "kps": kps, "color": color,
                    "panel_index": panel_index, "badge_index": badge_index,
                    "pose_draw_ok": pose_draw_ok, "tri_status": panel_tri,
                    "arm_status": panel_arm, "load_data": panel_load, "path_len": path_len,
                    "support_points": support_points if pose_draw_ok else [],
                    "support_triangle": support_triangle if pose_draw_ok else [],
                    "path_pts": list(path_tracker.history[pid]),   # immutable copy
                    "lock_frames": lock_frames_copy,               # immutable copy
                    "score_overall": overall, "score_data": _bd,
                })

            lost_items = []
            for pid in all_known_pids:
                if pid not in sorted_pids:
                    col = last_colors.get(pid, id_color(pid))
                    i = all_known_pids.index(pid)
                    overall, _ = scorers[pid].compute()
                    lost_items.append({
                        "pid": pid, "color": col, "panel_index": i,
                        "path_pts": list(path_tracker.history[pid]),
                        "last_stats": last_stats.get(pid, None),
                        "score_overall": overall,
                    })

            grip_touched_copy = [set(s) for s in grip_tracker.touched] if grip_tracker else None

            fps_cnt += 1
            if time.time()-fps_timer >= 1.0:
                disp_fps = fps_cnt/(time.time()-fps_timer); fps_cnt=0; fps_timer=time.time()

            packet = {
                "items": render_items, "lost_items": lost_items,
                "grip_tracker": grip_tracker, "grip_touched": grip_touched_copy,
                "force_analyzer": force_analyzer, "lock_thresh": lock_detector.thresh,
                "sorted_pids": sorted_pids, "frame_num": frame_num,
                "fps": disp_fps, "device_name": device_name,
            }
            t_draw += time.perf_counter() - _t2
            _t3 = time.perf_counter()
            draw_worker.submit(frame, packet)      # worker draws + writes
            t_write += time.perf_counter() - _t3

            if frame_num % 100 == 0 and not is_live:
                pct = frame_num/total*100 if total > 0 else 0
                n = 100.0
                print(f"  [{pct:5.1f}%] frame {frame_num:5d}  ids={len(sorted_pids)}  fps={disp_fps:.1f}"
                      f"  | read={t_read/n*1000:.0f}ms infer={t_infer/n*1000:.0f}ms "
                      f"build={t_draw/n*1000:.0f}ms submit={t_write/n*1000:.0f}ms")
                t_read = t_infer = t_draw = t_write = 0.0
            continue   # skip the inline proven draw path below

        for pid, bxyxy, kps in wall_climbers:
            color = id_color(pid)
            badge_index = all_known_pids.index(pid) if pid in all_known_pids else 0
            panel_index = badge_index
            last_colors[pid] = color
            if pid not in scorers: scorers[pid] = ClimbScorer(H)

            # Box + path are always safe to draw (slow-changing).
            draw_person_box(frame, bxyxy, pid, color)
            path_len = path_tracker.path_length(pid)

            age = pose_age.get(pid, 999)
            valid = pose_is_valid(kps)
            pose_draw_ok  = (age <= POSE_DRAW_MAX_AGE) and valid
            pose_score_ok = (age <= POSE_SCORE_MAX_AGE) and valid

            # default carried-over stats (used if not scoring this frame)
            arm_status, load_data, tri_status, _ = last_stats.get(
                pid, ({}, None, "unknown", path_len))

            if pose_draw_ok:
                # Pose fresh enough to DISPLAY skeleton + overlays.
                draw_skeleton(frame, kps, color)

                # Path + follow-crop must only use FRESHLY inferred pose, not cached
                # draw-poses, or path explodes and follow-crop drifts.
                if pose_score_ok:
                    path_tracker.update(pid, kps)
                    hip_pos = path_tracker.history[pid][-1] if path_tracker.history[pid] else None
                    if hip_pos:
                        box_size = (int(bxyxy[2]-bxyxy[0]), int(bxyxy[3]-bxyxy[1]))
                        follow_tracker.update_seen(pid, hip_pos, box_size=box_size)

                path_tracker.draw(frame, pid, color=path_color(pid))
                path_len = path_tracker.path_length(pid)
                hip_pos = path_tracker.history[pid][-1] if path_tracker.history[pid] else None

                if grip_tracker:
                    tri_status, support_points, support_triangle = \
                        grip_tracker.support_geometry(pid, kps)
                    active_contacts = set(grip_tracker.current_contacts(pid))
                else:
                    tri_status, support_points, support_triangle = "unknown", [], []
                    active_contacts = set()
                if DRAW_WEIGHT_TRIANGLE:
                    draw_support_geometry(frame, support_points, support_triangle, tri_status)
                phase = phase_cache.get(pid, {"state":"unknown", "com":None,
                    "body_h":max(1.0,float(bxyxy[3]-bxyxy[1]))})
                arm_status = lock_detector.update(
                    pid, kps, source_time, pose_score_ok, active_contacts,
                    phase.get("state", "unknown"))
                lock_detector.draw(frame, pid, kps, arm_status)

                # SCORING only on freshly inferred pose (no stale pollution).
                if pose_score_ok:
                    load_data = (force_analyzer.analyze(
                        pid, kps, active_contacts, phase) if force_analyzer else None)
                    if load_data and DRAW_WEIGHT_DIAGRAM:
                        force_analyzer.draw_body_diagram(frame, kps, load_data, color)
                    hip_pos = path_tracker.history[pid][-1] if path_tracker.history[pid] else None
                    scorers[pid].update(
                        source_time, kps, bxyxy, arm_status, tri_status, phase,
                        contact_events.get(pid, []), active_contacts, load_data)
                    last_stats[pid] = (arm_status, load_data, tri_status, path_len)
            else:
                # Pose stale — path only, reuse last stats, no scoring.
                path_tracker.draw(frame, pid, color=path_color(pid))

            _overall, _bd = scorers[pid].compute()
            panel_arm = arm_status if pose_draw_ok else {}
            panel_load = load_data if pose_draw_ok else None
            panel_tri = tri_status if pose_draw_ok else "unknown"
            draw_stats_panel(frame, pid, panel_arm, panel_load, panel_tri, path_len,
                             color, force_analyzer, panel_index=panel_index,
                             score_data=_bd)
            scorers[pid].draw_live(frame, pid, color, badge_index)

        # ── Persist lost climbers ─────────────────────
        for pid in all_known_pids:
            if pid not in sorted_pids:
                col = last_colors.get(pid, id_color(pid))
                i = all_known_pids.index(pid)
                path_tracker.draw(frame, pid, color=path_color(pid))
                if pid in last_stats:
                    _,_,_,path_l = last_stats[pid]
                    draw_stats_panel(frame, pid, {}, None, "unknown", path_l, col,
                                     force_analyzer, panel_index=i)
                scorers[pid].draw_live(frame, pid, col, i)

        if grip_tracker:
            grip_tracker.draw(frame)
            draw_legend(frame)

        fps_cnt += 1
        if time.time()-fps_timer >= 1.0:
            disp_fps = fps_cnt/(time.time()-fps_timer); fps_cnt=0; fps_timer=time.time()
        draw_info_panel(frame, frame_num, disp_fps, sorted_pids, device_name)

        t_draw += time.perf_counter() - _t2
        _t3 = time.perf_counter()
        if writer: writer.write(frame)
        cv2.imshow("Climbing AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): print("\nStopped by user."); break
        t_write += time.perf_counter() - _t3

        if frame_num % 100 == 0 and not is_live:
            pct = frame_num/total*100 if total > 0 else 0
            n = 100.0
            print(f"  [{pct:5.1f}%] frame {frame_num:5d}  ids={len(sorted_pids)}  fps={disp_fps:.1f}"
                  f"  | read={t_read/n*1000:.0f}ms infer={t_infer/n*1000:.0f}ms "
                  f"draw={t_draw/n*1000:.0f}ms write={t_write/n*1000:.0f}ms")
            t_read = t_infer = t_draw = t_write = 0.0

    # Stop the draw worker thread first (threaded mode) so the final card can be
    # written directly and the queue is fully drained.
    if draw_worker is not None:
        draw_worker.release()

    # ── Final score card ──────────────────────────────
    target_writer = writer if writer else (draw_worker.writer if draw_worker else None)
    if target_writer is not None and scorers:
        final_frame = np.zeros((H,W,3), dtype=np.uint8)
        MIN_PATH_PX = 300
        final_pids = sorted([p for p in scorers if path_tracker.path_length(p)>=MIN_PATH_PX])
        if not final_pids: final_pids = sorted(scorers.keys())
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(final_frame,"CLIMB COMPLETE",(W//2-140,H//2-210),font,1.0,(200,200,200),2)
        for i,pid in enumerate(final_pids):
            col = last_colors.get(pid, id_color(pid))
            scorers[pid].draw_final(final_frame, pid, col, card_index=i, total_cards=len(final_pids))
        for _ in range(int(fps*3)): target_writer.write(final_frame)
        if writer:   # live preview only exists in inline mode
            cv2.imshow("Climbing AI", final_frame); cv2.waitKey(2000)

    cap.release()
    if writer: writer.release()
    if draw_worker is not None: draw_worker.writer.release()
    cv2.destroyAllWindows()

    # ── Release RKNN models ───────────────────────────
    track_model.release()
    pose_model.release()
    if hold_model: hold_model.release()

    # ── Session summary ───────────────────────────────
    print("\n" + "="*55)
    print("  Session Summary")
    print("="*55)
    for pid in sorted(p for p in path_tracker.history if path_tracker.path_length(p)>=300):
        print(f"\n  Climber ID {pid}:")
        print(f"    Path length : {path_tracker.path_length(pid):.0f} px")
        hp = path_tracker.highest_point(pid)
        if hp: print(f"    Highest Y   : {hp[1]} px from top")
        if pid in scorers:
            overall, bd = scorers[pid].compute()
            shown = "N/A" if overall is None else f"{overall} / 100"
            print(f"    OVERALL      : {shown}")
            print(f"    Arm use      : {bd['arm']}")
            print(f"    Support/tri  : {bd['support']}")
            print(f"    Footwork/legs: {bd['footwork']}")
            print(f"    Fluency      : {bd['fluency']}")
            if bd["ge_raw"] is not None:
                print(f"    GE diagnostic: {bd['ge_raw']:.3f}")
    if grip_tracker:
        r = grip_tracker.session_report()
        print(f"\n  Hold report: {r['total_holds']} total, {r['hand_holds']} hand, {r['foot_holds']} foot")
    if output_path: print(f"\n  Video saved → {output_path}")

# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Climbing AI final single-climber RK3588 pipeline")
    parser.add_argument("--source", "-s", default=DEFAULT_SOURCE)
    parser.add_argument("--weight", "-w", type=float, default=BODY_WEIGHT_KG)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    output_path = args.output
    if output_path is None and isinstance(source, str):
        results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
        output_path = str(results_dir / Path(source).name)
    run(source, body_weight_kg=args.weight, output_path=output_path)
