"""
Climbing AI — main_crop3: Triangle & Arm Tech Scoring
======================================================
Builds on main_crop2 with improved scoring:
  - Triangle score: normalized by body size (Method 3)
  - Arm tech score: per-frame straight arm percentage
  - Balance and efficiency metrics removed

Usage:
    python main_crop3_triangle_arm_score.py --source "your_video.MOV" --weight 45
    python main_crop3_triangle_arm_score.py                            # live camera
"""

import cv2
import numpy as np
import argparse
import math
import time
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════

# ── Models (ONNX versions for GPU/NPU acceleration) ───
TRACK_MODEL  = "yolo11m.onnx"
POSE_MODEL   = "yolo11m-pose.onnx"
HOLD_MODEL   = "runs/hold_detection/weights/best.onnx"

# ── Fallback to .pt if .onnx not found ────────────────
TRACK_MODEL_PT = "yolo11m.pt"
POSE_MODEL_PT  = "yolo11m-pose.pt"
HOLD_MODEL_PT  = "runs/hold_detection/weights/best.pt"

# ── Source ────────────────────────────────────────────
DEFAULT_SOURCE   = 0          # 0 = first camera, or path to video file

# ── Climber selection ─────────────────────────────────
# "largest" = biggest bounding box (closest to camera)
# "highest" = highest on the wall
# None      = track all climbers
TARGET_SELECTION = None

# ── Body weight ───────────────────────────────────────
BODY_WEIGHT_KG   = None       # set via --weight or hardcode here

# ── Wall filter ───────────────────────────────────────
GROUND_FILTER    = 0.97       # tall walls — climbers reach near the top
MIN_PERSON_H     = 15         # allow small detections on tall walls

# ── Keypoint confidence ───────────────────────────────
KP_CONF          = 0.25
CROP_PAD         = 0.20

# ── Hold detection ────────────────────────────────────
HOLD_DETECT_FRAMES   = 5
HOLD_CONF            = 0.35
HOLD_NMS_DIST        = 30
GRIP_RADIUS          = 50
GRIP_CONFIRM_FRAMES  = 8

# ── Triangle support ──────────────────────────────────
# Triangle scoring — Method 3 (normalized by body size)
TRI_HEIGHT_RATIO_GOOD  = 0.35   # wrist midpoint must be this far above ankle (relative to body height)
TRI_HEIGHT_RATIO_WARN  = 0.15   # below this = warn
TRI_SPREAD_RATIO_GOOD  = 0.25   # wrist spread must be this wide (relative to body width)
TRI_SPREAD_RATIO_WARN  = 0.10   # below this = warn

# ── Lock arm ──────────────────────────────────────────
LOCK_ANGLE       = 110
LOCK_SECONDS     = 3.0

GRAVITY          = 9.81

# ── Emergency zoom fallback ───────────────────────────
# When a known climber is not detected for this many frames,
# crop around their last known position and retry detection
LOST_FRAMES_TRIGGER  = 3     # frames not detected before zoom kicks in
MIN_SEPARATION       = 250   # px — min distance between climbers to allow zoom/remap
ZOOM_CROP_HALF_W     = 300   # half-width of the emergency zoom crop
ZOOM_CROP_HALF_H     = 400   # half-height of the emergency zoom crop
ZOOM_EXPAND_PER_SEC  = 50    # expand crop by this many px/sec if still not found

# ── Follow zoom (Phase 2) ─────────────────────────────────────────────
# Once a climber is first detected, a follow crop tracks them every frame
# so they stay large in the crop regardless of how high they climb
FOLLOW_CROP_PADDING  = 2.0   # crop = last bounding box size x this multiplier
FOLLOW_CROP_MIN_HALF = 60    # minimum half-size of crop in pixels
FOLLOW_MIN_FRAMES    = 5     # frames needed before switching to follow mode

# ═══════════════════════════════════════════════════════
#  DEVICE SETUP
# ═══════════════════════════════════════════════════════

def setup_device():
    """Use CPU for inference."""
    print("\n  Device: CPU")
    return "cpu", "CPU"

def load_model(onnx_path, pt_path, task, device):
    """
    Load ONNX model if available, otherwise fall back to .pt.
    """
    if Path(onnx_path).exists():
        print(f"  Loading ONNX : {onnx_path}")
        return YOLO(onnx_path, task=task)
    elif Path(pt_path).exists():
        print(f"  Loading PT   : {pt_path}  (run export to get ONNX)")
        return YOLO(pt_path)
    else:
        raise FileNotFoundError(
            f"Model not found.\n"
            f"  ONNX: {onnx_path}\n"
            f"  PT  : {pt_path}"
        )

# ═══════════════════════════════════════════════════════
#  COLORS (BGR)
# ═══════════════════════════════════════════════════════

ID_COLORS = [
    (100,220,255),(100,255,150),(255,150,100),
    (180,100,255),(100,180,255),(255,100,180),
]
def id_color(pid): return ID_COLORS[(pid-1) % len(ID_COLORS)]

COL_PATH      = (255, 60,  255)  # fallback single-climber color

# Distinct path colors per climber, assigned in order of first appearance
PATH_COLORS = [
    (0,   255, 255),   # yellow
    (255,  50, 255),   # magenta
    (50,  255,  50),   # green
    (255, 100,  50),   # orange
    (50,  100, 255),   # red
    (255, 255,  50),   # cyan
    (50,  255, 200),   # lime
    (200,  50, 255),   # purple
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
    pts  = kpts_xy.cpu().numpy()
    conf = kpts_conf.cpu().numpy()
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

# ═══════════════════════════════════════════════════════
#  WALL FILTER + TARGET SELECTION
# ═══════════════════════════════════════════════════════

def on_wall(box, H):
    x1,y1,x2,y2 = box
    return (y2-y1) >= MIN_PERSON_H and ((y1+y2)/2) < H * GROUND_FILTER

def select_target(wall_climbers):
    if not wall_climbers or TARGET_SELECTION is None:
        return None
    if TARGET_SELECTION == "largest":
        return max(wall_climbers,
                   key=lambda c: (c[1][2]-c[1][0])*(c[1][3]-c[1][1]))[0]
    if TARGET_SELECTION == "highest":
        return min(wall_climbers,
                   key=lambda c: (
                       c[2].get("left_hip",(0,9999,0))[1] +
                       c[2].get("right_hip",(0,9999,0))[1]) / 2)[0]
    return None

# ═══════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════

def draw_skeleton(frame, keypoints, color):
    pts = list(keypoints.values())
    if len(pts) < 17: return   # skip if pose failed
    for p1, p2 in SKELETON:
        k1, k2 = pts[p1], pts[p2]
        if k1[2] >= KP_CONF and k2[2] >= KP_CONF:
            cv2.line(frame,
                     (int(k1[0]), int(k1[1])),
                     (int(k2[0]), int(k2[1])),
                     color, 2, cv2.LINE_AA)
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
    n  = len(active_ids)
    ov = frame.copy()
    cv2.rectangle(ov, (10,10), (260, 68+n*18), (15,15,15), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Frame {frame_num}  |  {fps:.1f} FPS",
                (18,30), font, 0.42, (200,200,200), 1)
    cv2.putText(frame, f"Device: {device_name}",
                (18,48), font, 0.38, (100,220,255), 1)
    cv2.putText(frame, f"Climbers on wall: {n}",
                (18,66), font, 0.42, (100,255,150), 1)
    y = 84
    for pid in sorted(active_ids):
        cv2.putText(frame, f"  ID {pid}", (18,y),
                    font, 0.38, id_color(pid), 1)
        y += 18

# ═══════════════════════════════════════════════════════
#  MODULE 1 — PATH TRACKING
# ═══════════════════════════════════════════════════════

class PathTracker:
    def __init__(self):
        self.history = defaultdict(list)

    def update(self, pid, kps):
        lh = kp(kps, "left_hip")
        rh = kp(kps, "right_hip")
        if lh and rh:
            pt = (int((lh[0]+rh[0])/2), int((lh[1]+rh[1])/2))
            h  = self.history[pid]
            if not h or dist(h[-1], pt) > 5:
                h.append(pt)
            return pt
        return None

    def draw(self, frame, pid, color=None):
        pts = self.history[pid]
        col = color if color is not None else COL_PATH
        MAX_SEGMENT = 150  # px — skip line if two points are too far apart
        for i in range(1, len(pts)):
            if dist(pts[i-1], pts[i]) <= MAX_SEGMENT:
                cv2.line(frame, pts[i-1], pts[i], col, 3, cv2.LINE_AA)
        if pts:
            cv2.circle(frame, pts[-1], 6, COL_PATH, -1, cv2.LINE_AA)
            cv2.circle(frame, pts[-1], 6, (255,255,255), 1, cv2.LINE_AA)

    def path_length(self, pid):
        pts = self.history[pid]
        return sum(dist(pts[i-1], pts[i]) for i in range(1, len(pts)))

    def highest_point(self, pid):
        pts = self.history[pid]
        return min(pts, key=lambda p: p[1]) if pts else None

# ═══════════════════════════════════════════════════════
#  MODULE 2 — TRIANGLE SUPPORT
# ═══════════════════════════════════════════════════════

def _estimate_point(kps, joint, parent, grandparent, body_height):
    """Return joint keypoint, or estimate it by extending the limb direction."""
    p = kp(kps, joint)
    if p: return p
    par  = kp(kps, parent)
    gpar = kp(kps, grandparent)
    if par and gpar:
        dx = par[0] - gpar[0]
        dy = par[1] - gpar[1]
        length = math.hypot(dx, dy)
        if length > 0:
            scale = (body_height * 0.20) / length
            return (par[0] + dx * scale, par[1] + dy * scale)
    return None

def draw_triangle(frame, kps, bxyxy=None):
    """
    Triangle = 2 wrists (hands) + 1 ankle (foot on hold).
    Scoring uses Method 3 — normalized by body size:
      - height_ratio: how high wrist midpoint is above the foot (relative to body height)
      - spread_ratio: how wide the two wrists are spread (relative to body width)
    Falls back to elbow/knee estimation if wrist/ankle not detected.
    """
    # Body size for normalization
    if bxyxy is not None:
        body_h = max(1, float(bxyxy[3] - bxyxy[1]))
        body_w = max(1, float(bxyxy[2] - bxyxy[0]))
    else:
        body_h, body_w = 200, 80   # fallback defaults

    # Get wrists with estimation fallback
    lh = _estimate_point(kps, "left_wrist",  "left_elbow",  "left_shoulder",  body_h)
    rh = _estimate_point(kps, "right_wrist", "right_elbow", "right_shoulder", body_h)

    # Get ankles with estimation fallback
    la = _estimate_point(kps, "left_ankle",  "left_knee",  "left_hip",  body_h)
    ra = _estimate_point(kps, "right_ankle", "right_knee", "right_hip", body_h)

    if not lh or not rh:
        return "unknown"

    feet = [f for f in [la, ra] if f]
    if not feet:
        return "unknown"

    p1, p2 = lh, rh
    # Pick the foot with lowest Y (highest on wall = most likely on a hold)
    foot = min(feet, key=lambda f: f[1])

    # ── Method 3 scoring ──────────────────────────────────────────────
    wrist_mid_y  = (p1[1] + p2[1]) / 2
    height_ratio = (foot[1] - wrist_mid_y) / body_h   # positive = wrists above foot
    spread_ratio = dist(p1, p2) / body_w

    height_good  = height_ratio >= TRI_HEIGHT_RATIO_GOOD
    height_warn  = height_ratio <  TRI_HEIGHT_RATIO_WARN
    spread_good  = spread_ratio >= TRI_SPREAD_RATIO_GOOD
    spread_warn  = spread_ratio <  TRI_SPREAD_RATIO_WARN

    if height_good and spread_good:
        status = "good"
    elif height_warn or spread_warn:
        status = "warn"
    else:
        status = "ok"

    color = {"good":COL_TRI_GOOD,"ok":COL_TRI_OK,
             "warn":COL_TRI_WARN}.get(status, (120,120,120))

    # Draw triangle
    tri = np.array([[int(p[0]),int(p[1])] for p in [p1,p2,foot]], np.int32)
    ov  = frame.copy()
    cv2.fillPoly(ov, [tri], color)
    cv2.addWeighted(ov, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [tri], True, color, 2, cv2.LINE_AA)
    for p in [p1, p2, foot]:
        cv2.circle(frame, (int(p[0]),int(p[1])), 6, color, -1, cv2.LINE_AA)
    if status == "warn":
        cx = int((p1[0]+p2[0]+foot[0])/3)
        cy = int((p1[1]+p2[1]+foot[1])/3)
        cv2.putText(frame, "TOO NARROW", (cx-45,cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TRI_WARN, 2)
    return status

# ═══════════════════════════════════════════════════════
#  MODULE 3 — LOCK ARM WARNING
# ═══════════════════════════════════════════════════════

class LockArmDetector:
    def __init__(self, fps):
        self.thresh  = int(LOCK_SECONDS * fps)
        self.frames  = defaultdict(lambda: {"left":0,"right":0})

    def update(self, pid, kps):
        status = {"left":"unknown","right":"unknown","warning":False}
        sides  = {
            "left":  ("left_shoulder","left_elbow","left_wrist"),
            "right": ("right_shoulder","right_elbow","right_wrist"),
        }
        for side, (sh,el,wr) in sides.items():
            ang = angle3(kp(kps,sh), kp(kps,el), kp(kps,wr))
            if ang is None:
                status[side] = "unknown"
                self.frames[pid][side] = 0
            elif ang < LOCK_ANGLE:
                self.frames[pid][side] += 1
                status[side] = "bent"
            else:
                self.frames[pid][side] = 0
                status[side] = "straight"
        if (self.frames[pid]["left"]  >= self.thresh or
            self.frames[pid]["right"] >= self.thresh):
            status["warning"] = True
        return status

    def draw(self, frame, pid, kps, arm_status):
        for side, el_name in [("left","left_elbow"),("right","right_elbow")]:
            el = kp(kps, el_name)
            if el is None: continue
            ex, ey  = int(el[0]), int(el[1])
            f       = self.frames[pid][side]
            if f > 0:
                pct     = min(f / self.thresh, 1.0)
                warning = pct >= 1.0
                color   = (50, 50, 255) if warning else (50, 180, 255)
                # Circle grows from radius 14 → 26 when warning fires
                radius  = 26 if warning else int(14 + pct * 6)
                thickness = 3 if warning else 2
                cv2.ellipse(frame, (ex, ey), (radius, radius), -90,
                            0, int(pct * 360), color, thickness, cv2.LINE_AA)
                # Show "Stretch arm!" label next to elbow when warning fires
                if warning:
                    lbl_x = ex + radius + 5
                    lbl_y = ey + 5
                    cv2.putText(frame, "Stretch arm!",
                                (lbl_x, lbl_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (50, 50, 255), 2, cv2.LINE_AA)
        if arm_status.get("warning"):
            H = frame.shape[0]
            cv2.rectangle(frame, (0, H-50), (frame.shape[1], H), (0, 0, 180), -1)
            cv2.putText(frame,
                        "WARNING: Lock arm — straighten your arms!",
                        (15, H-18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

# ═══════════════════════════════════════════════════════
#  MODULE 4 — FORCE ANALYSIS
# ═══════════════════════════════════════════════════════

class ForceAnalyzer:
    def __init__(self, weight_kg):
        self.weight_n = weight_kg * GRAVITY

    def analyze(self, kps):
        lw   = kp(kps,"left_wrist")
        rw   = kp(kps,"right_wrist")
        lhip = kp(kps,"left_hip")
        rhip = kp(kps,"right_hip")
        if not all([lw,rw,lhip,rhip]): return None
        cog_x  = (lhip[0]+rhip[0]) / 2
        dl, dr = abs(cog_x-lw[0]), abs(cog_x-rw[0])
        tot    = dl + dr
        if tot < 1: return None
        return {
            "pct_left":  round(dr/tot*100, 1),
            "pct_right": round(dl/tot*100, 1),
        }

    def draw_bar(self, frame, data, px, py, bar_w=164):
        if data is None:
            cv2.putText(frame, "Force: N/A", (px,py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)
            return
        lw = int(bar_w * data["pct_left"] / 100)
        cv2.rectangle(frame, (px,py), (px+bar_w,py+12), (60,60,60), -1)
        cv2.rectangle(frame, (px,py), (px+lw,py+12), (255,100,100), -1)
        cv2.rectangle(frame, (px+lw,py), (px+bar_w,py+12), (100,100,255), -1)
        cv2.line(frame, (px+lw,py), (px+lw,py+12), (255,255,255), 1)
        f = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"L {data['pct_left']:.0f}%",
                    (px,py-3), f, 0.38, (255,130,130), 1)
        cv2.putText(frame, f"{data['pct_right']:.0f}% R",
                    (px+bar_w-42,py-3), f, 0.38, (130,130,255), 1)

# ═══════════════════════════════════════════════════════
#  SCORING SYSTEM
# ═══════════════════════════════════════════════════════

class ClimbScorer:
    """Live score (0-100). Overall = Triangle 50% + Arm Technique 50%."""

    def __init__(self, frame_height):
        self.H           = frame_height
        # triangle
        self.tri_good    = 0
        self.tri_total   = 0
        # arm — per frame straight arm tracking
        self.arm_straight = 0   # frames with at least one straight arm
        self.arm_total    = 0   # total frames with arm data

    def update(self, tri_status, arm_status, force_data, hip_pos, path_length):
        # Triangle — per frame average
        if tri_status != "unknown":
            self.tri_total += 1
            if tri_status in ("good", "ok"):
                self.tri_good += 1

        # Arm technique — count frames with at least one straight arm
        left  = arm_status.get("left",  "unknown")
        right = arm_status.get("right", "unknown")
        if left != "unknown" or right != "unknown":
            self.arm_total += 1
            if left == "straight" or right == "straight":
                self.arm_straight += 1

    def compute(self):
        """Returns (overall_score, breakdown_dict) — all 0-100."""

        # Triangle — % of frames with good/ok triangle
        if self.tri_total > 0:
            tri_score = (self.tri_good / self.tri_total) * 100
        else:
            tri_score = 50.0

        # Arm technique — % of frames with at least one straight arm
        if self.arm_total > 0:
            arm_score = (self.arm_straight / self.arm_total) * 100
        else:
            arm_score = 50.0

        # Overall = 50% triangle + 50% arm
        overall = round(tri_score * 0.50 + arm_score * 0.50)

        return overall, {
            "triangle": round(tri_score),
            "arm":      round(arm_score),
        }

    def draw_live(self, frame, pid, id_color, badge_index=0):
        """Draw live score badge. badge_index offsets position for multiple climbers."""
        overall, breakdown = self.compute()
        W, H = frame.shape[1], frame.shape[0]

        score_color = (50,220,50) if overall >= 75 else (50,200,220) if overall >= 50 else (50,50,255)

        badge_w = 150
        px = W - 10 - badge_w * (badge_index + 1)
        py = H - 180

        ov = frame.copy()
        cv2.rectangle(ov, (px, py), (px+badge_w, H-10), (15,15,15), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

        cv2.rectangle(frame, (px, py), (px+4, H-10), id_color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(overall), (px+12, py+62),
                    font, 2.2, score_color, 4, cv2.LINE_AA)
        cv2.putText(frame, "/100", (px+12, py+82),
                    font, 0.5, (180,180,180), 1)
        cv2.putText(frame, f"ID {pid}", (px+12, py+100),
                    font, 0.4, id_color, 1)

        # Mini breakdown bars — only TRI and ARM
        items = [
            ("TRI", breakdown["triangle"], COL_TRI_GOOD),
            ("ARM", breakdown["arm"],      (100,220,255)),
        ]
        bx, by = px+12, py+108
        bar_w  = 100
        for label, val, col in items:
            cv2.putText(frame, label, (bx, by+9), font, 0.3, (160,160,160), 1)
            filled = int(bar_w * val / 100)
            cv2.rectangle(frame, (bx+30, by), (bx+30+bar_w, by+8), (50,50,50), -1)
            cv2.rectangle(frame, (bx+30, by), (bx+30+filled, by+8), col, -1)
            by += 14

    def draw_final(self, frame, pid, id_color, card_index=0, total_cards=1):
        """Draw final score card. Cards are placed side by side for multiple climbers."""
        overall, bd = self.compute()
        H, W = frame.shape[:2]

        # Card is 420px wide — center all cards together horizontally
        card_w   = 420
        gap      = 20
        total_w  = total_cards * card_w + (total_cards - 1) * gap
        start_x  = (W - total_w) // 2
        cx       = start_x + card_index * (card_w + gap) + card_w // 2
        card_x1  = cx - card_w // 2
        card_x2  = cx + card_w // 2
        card_y1  = H // 2 - 190
        card_y2  = H // 2 + 210

        ov = frame.copy()
        cv2.rectangle(ov, (card_x1, card_y1), (card_x2, card_y2), (10,10,10), -1)
        cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y1+4), id_color, -1)
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (70,70,70), 1)

        font        = cv2.FONT_HERSHEY_SIMPLEX
        score_color = (50,220,50) if overall >= 75 else (50,200,220) if overall >= 50 else (50,50,255)

        cv2.putText(frame, f"Climber ID {pid}",
                    (card_x1+14, card_y1+26), font, 0.55, id_color, 1)
        cv2.putText(frame, str(overall),
                    (cx-38, card_y1+100), font, 3.0, score_color, 6, cv2.LINE_AA)
        cv2.putText(frame, "/ 100",
                    (cx-28, card_y1+122), font, 0.6, (160,160,160), 1)

        rows = [
            ("Triangle",    bd["triangle"], True),
            ("Arm tech",    bd["arm"],      True),
        ]
        y = card_y1 + 142
        bar_w = 90
        for label, val, is_section in rows:
            col  = (220,220,220) if is_section else (160,160,160)
            sz   = 0.42 if is_section else 0.36
            cv2.putText(frame, label, (card_x1+14, y), font, sz, col, 1)
            bar_x  = card_x1 + 160
            filled = int(bar_w * val / 100)
            cv2.rectangle(frame, (bar_x, y-8), (bar_x+bar_w, y), (50,50,50), -1)
            bar_col = (50,220,50) if val >= 75 else (50,200,220) if val >= 50 else (50,50,255)
            cv2.rectangle(frame, (bar_x, y-8), (bar_x+filled, y), bar_col, -1)
            cv2.putText(frame, str(val),
                        (bar_x+bar_w+6, y), font, 0.36, col, 1)
            y += 20 if is_section else 17

# ═══════════════════════════════════════════════════════
#  STATS PANEL
# ═══════════════════════════════════════════════════════

def draw_stats_panel(frame, pid, arm_status, force_data,
                     tri_status, path_len, color, force_analyzer,
                     panel_index=0):
    # Panel width 170px, stacked vertically — panel_index offsets each one
    PANEL_W = 170
    PANEL_H = 158
    GAP     = 8
    W       = frame.shape[1]
    px      = W - PANEL_W - 4
    py      = 4 + panel_index * (PANEL_H + GAP)

    ov = frame.copy()
    cv2.rectangle(ov, (px, py), (px+PANEL_W, py+PANEL_H), (15,15,15), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (px, py), (px+3, py+PANEL_H), color, -1)

    f = cv2.FONT_HERSHEY_SIMPLEX
    x, y = px+8, py+14
    cv2.putText(frame, f"Climber ID {pid}", (x,y), f, 0.38, color, 1)
    y += 17
    tri_c = {"good":COL_TRI_GOOD,"ok":COL_TRI_OK,
              "warn":COL_TRI_WARN,"unknown":(120,120,120)}
    tri_l = {"good":"STABLE","ok":"OK","warn":"NARROW!","unknown":"---"}
    cv2.putText(frame, f"Tri: {tri_l.get(tri_status,'---')}",
                (x,y), f, 0.34, tri_c.get(tri_status,(120,120,120)), 1)
    y += 15
    for side in ["left","right"]:
        s   = arm_status.get(side,"unknown")
        col = (50,50,255) if s=="bent" else (50,220,50) if s=="straight" else (120,120,120)
        cv2.putText(frame, f"{side[0].upper()} arm: {s}", (x,y), f, 0.32, col, 1)
        y += 14
    cv2.line(frame, (x,y), (px+PANEL_W-6,y), (60,60,60), 1); y += 8
    cv2.putText(frame, "Load:", (x,y), f, 0.32, (180,180,180), 1); y += 12
    if force_analyzer:
        force_analyzer.draw_bar(frame, force_data, x, y, bar_w=130)
    y += 22
    cv2.line(frame, (x,y), (px+PANEL_W-6,y), (60,60,60), 1); y += 8
    cv2.putText(frame, f"Path: {path_len:.0f}px", (x,y), f, 0.32, (200,200,200), 1)

def draw_legend(frame):
    H = frame.shape[0]
    items = [(COL_HOLD_IDLE,"Detected hold"),
             (COL_HOLD_HAND,"Hand hold"),
             (COL_HOLD_FOOT,"Foot hold")]
    y = H - 80
    for col, lbl in items:
        cv2.circle(frame, (25,y), 7, col, -1, cv2.LINE_AA)
        cv2.putText(frame, lbl, (38,y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        y += 22

# ═══════════════════════════════════════════════════════
#  EMERGENCY ZOOM TRACKER
# ═══════════════════════════════════════════════════════

class FollowZoomTracker:
    """
    Phase 2 tracker — maintains a dynamic follow crop centered on each
    known climber's last hip position. Crop size adapts to bounding box size.
    """

    def __init__(self):
        self.detected_frames = defaultdict(int)  # { pid: total frames detected }
        self.follow_box      = {}                # { pid: last known (x1,y1,x2,y2) crop in full frame }

    def is_following(self, pid):
        """True once a climber has been seen enough frames to trust."""
        return self.detected_frames[pid] >= FOLLOW_MIN_FRAMES

    def update_seen(self, pid, hip_pos, box_size=None):
        """Call when climber was detected.
        hip_pos:  (cx, cy) center of gravity in full frame.
        box_size: (w, h) of last detected bounding box — used for dynamic crop.
        """
        self.detected_frames[pid] += 1
        if hip_pos:
            cx, cy = hip_pos
            # Dynamic crop size based on last known bounding box
            if box_size:
                bw, bh   = box_size
                half_w   = max(FOLLOW_CROP_MIN_HALF, int(bw * FOLLOW_CROP_PADDING))
                half_h   = max(FOLLOW_CROP_MIN_HALF, int(bh * FOLLOW_CROP_PADDING))
            else:
                half_w   = max(FOLLOW_CROP_MIN_HALF, 150)
                half_h   = max(FOLLOW_CROP_MIN_HALF, 200)
            self.follow_box[pid] = (
                cx - half_w,
                cy - half_h,
                cx + half_w,
                cy + half_h,
            )

    def get_crop_box(self, pid, W, H):
        """Returns (x1, y1, x2, y2) of the follow crop in full frame coords."""
        if pid not in self.follow_box:
            return None
        x1, y1, x2, y2 = self.follow_box[pid]
        return (
            max(0, int(x1)),
            max(0, int(y1)),
            min(W, int(x2)),
            min(H, int(y2)),
        )

    def detect_in_crop(self, pid, track_model, frame, W, H, device):
        """
        Run detection inside the follow crop.
        Returns list of (box_xyxy_fullframe, score) for persons found.
        """
        box = self.get_crop_box(pid, W, H)
        if box is None:
            return []
        cx1, cy1, cx2, cy2 = box
        if cx2 - cx1 < 50 or cy2 - cy1 < 50:
            return []

        crop    = frame[cy1:cy2, cx1:cx2]
        results = track_model(crop, verbose=False, classes=[0], device=device)

        found = []
        for r in results:
            if r.boxes is None: continue
            for b in r.boxes:
                conf = float(b.conf[0])
                if conf < 0.15: continue
                bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                # Remap to full frame
                found.append((
                    (bx1+cx1, by1+cy1, bx2+cx1, by2+cy1),
                    conf
                ))
        return found

class EmergencyZoomTracker:
    """Fallback — after LOST_FRAMES_TRIGGER missed frames, crops around last known position and retries detection."""

    def __init__(self, fps):
        self.fps         = fps
        self.lost_frames = defaultdict(int)  # { pid: consecutive frames lost }

    def update_seen(self, pid):
        """Call when pid was successfully detected this frame."""
        self.lost_frames[pid] = 0

    def update_lost(self, pid):
        """Call when pid was NOT detected this frame."""
        self.lost_frames[pid] += 1

    def should_zoom(self, pid):
        """Returns True after LOST_FRAMES_TRIGGER consecutive missed frames."""
        return self.lost_frames[pid] >= LOST_FRAMES_TRIGGER

    def zoom_crop_size(self, pid):
        """
        Returns (half_w, half_h) for the zoom crop.
        Expands slightly the longer the person has been lost.
        """
        extra_frames = max(0, self.lost_frames[pid] - LOST_FRAMES_TRIGGER)
        extra_px     = int((extra_frames / max(self.fps, 1)) * ZOOM_EXPAND_PER_SEC)
        return (ZOOM_CROP_HALF_W + extra_px,
                ZOOM_CROP_HALF_H + extra_px)

    def try_recover(self, pid, track_model, frame, W, H,
                    last_path_pos, device):
        """
        Attempt to detect the lost climber by zooming into their last
        known position. Returns (box_xyxy, keypoints) or None if not found.

        last_path_pos: (cx, cy) — last hip midpoint from PathTracker
        """
        if last_path_pos is None:
            return None

        cx, cy   = last_path_pos
        hw, hh   = self.zoom_crop_size(pid)

        # Build zoom crop around last known position
        zx1 = max(0, int(cx - hw))
        zy1 = max(0, int(cy - hh))
        zx2 = min(W, int(cx + hw))
        zy2 = min(H, int(cy + hh))

        if zx2 - zx1 < 50 or zy2 - zy1 < 50:
            return None   # crop too small

        zoom_crop = frame[zy1:zy2, zx1:zx2]

        # Run detection on zoom crop
        results = track_model(zoom_crop, verbose=False,
                              classes=[0], device=device)

        best_box  = None
        best_conf = 0.0

        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.15: continue
                if conf > best_conf:
                    best_conf = conf
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                    best_box = np.array([
                        x1 + zx1, y1 + zy1,
                        x2 + zx1, y2 + zy1
                    ])

        return best_box   # None if nobody found, box_xyxy if found

# ═══════════════════════════════════════════════════════
#  HOLD DETECTION + GRIP TRACKER
# ═══════════════════════════════════════════════════════

def detect_holds(hold_model, cap, W, H, device):
    print(f"  Detecting holds on first {HOLD_DETECT_FRAMES} frames...")
    raw = []
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    for _ in range(HOLD_DETECT_FRAMES):
        ret, frame = cap.read()
        if not ret: break
        results = hold_model(frame, verbose=False,
                             conf=HOLD_CONF, device=device)
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                raw.append({
                    "center": ((x1+x2)/2,(y1+y2)/2),
                    "box":    (int(x1),int(y1),int(x2),int(y2)),
                    "class":  int(box.cls[0]),
                    "conf":   float(box.conf[0]),
                })
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    raw   = sorted(raw, key=lambda h: h["conf"], reverse=True)
    holds = []
    for h in raw:
        if not any(dist(h["center"],k["center"]) < HOLD_NMS_DIST for k in holds):
            holds.append(h)

    grips   = sum(1 for h in holds if h["class"]==0)
    volumes = sum(1 for h in holds if h["class"]==1)
    print(f"  Found {len(holds)} holds ({grips} grips, {volumes} volumes)")
    return holds

class GripTracker:
    def __init__(self, holds):
        self.holds    = holds
        # Status is now a SET of limb types that have touched this hold
        # e.g. set() = idle, {"hand"} = hand only,
        #      {"foot"} = foot only, {"hand","foot"} = both
        self.touched  = [set() for _ in holds]
        self.counter  = defaultdict(lambda: defaultdict(int))
        self.limb_map = {
            "left_wrist":"hand","right_wrist":"hand",
            "left_ankle":"foot","right_ankle":"foot",
        }

    def update(self, all_kps):
        active = defaultdict(set)
        for kps in all_kps.values():
            for limb, ltype in self.limb_map.items():
                pos = kp(kps, limb)
                if pos is None: continue
                for idx, hold in enumerate(self.holds):
                    if dist(pos, hold["center"]) < GRIP_RADIUS:
                        active[idx].add(limb)

        for idx in range(len(self.holds)):
            limbs = active.get(idx, set())
            if limbs:
                for limb in limbs:
                    ltype = self.limb_map[limb]
                    if ltype in self.touched[idx]:
                        continue  # already confirmed this type
                    self.counter[idx][limb] += 1
                    if self.counter[idx][limb] >= GRIP_CONFIRM_FRAMES:
                        self.touched[idx].add(ltype)
            else:
                # Only reset counters for limb types not yet confirmed
                for limb in list(self.counter[idx].keys()):
                    ltype = self.limb_map[limb]
                    if ltype not in self.touched[idx]:
                        self.counter[idx][limb] = 0

    def draw(self, frame):
        for idx, hold in enumerate(self.holds):
            cx, cy = int(hold["center"][0]), int(hold["center"][1])
            if hold["class"] == 1:
                x1,y1,x2,y2 = hold["box"]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(120,120,60),1)
                continue
            t = self.touched[idx]
            if "hand" in t and "foot" in t:
                # Both — draw two concentric circles
                cv2.circle(frame,(cx,cy),16,COL_HOLD_FOOT,-1,cv2.LINE_AA)
                cv2.circle(frame,(cx,cy),10,COL_HOLD_HAND,-1,cv2.LINE_AA)
                cv2.circle(frame,(cx,cy),16,(255,255,255),1,cv2.LINE_AA)
            elif "hand" in t:
                cv2.circle(frame,(cx,cy),14,COL_HOLD_HAND,-1,cv2.LINE_AA)
                cv2.circle(frame,(cx,cy),14,(255,255,255),1,cv2.LINE_AA)
            elif "foot" in t:
                cv2.circle(frame,(cx,cy),14,COL_HOLD_FOOT,-1,cv2.LINE_AA)
                cv2.circle(frame,(cx,cy),14,(255,255,255),1,cv2.LINE_AA)
            else:
                cv2.circle(frame,(cx,cy),10,COL_HOLD_IDLE,1,cv2.LINE_AA)

    def session_report(self):
        return {
            "total_holds": len(self.holds),
            "hand_holds":  sum(1 for t in self.touched if "hand" in t),
            "foot_holds":  sum(1 for t in self.touched if "foot" in t),
            "both_holds":  sum(1 for t in self.touched if "hand" in t and "foot" in t),
        }

# ═══════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════

def run(source, body_weight_kg=None, output_path=None):
    print("\n" + "="*55)
    print("  Climbing AI — Real-Time Analysis")
    print("="*55)

    # ── Detect best device ────────────────────────────
    device, device_name = setup_device()

    # ── Load models ───────────────────────────────────
    print("\nLoading models...")
    track_model = load_model(TRACK_MODEL, TRACK_MODEL_PT, "detect", device)
    pose_model  = load_model(POSE_MODEL,  POSE_MODEL_PT,  "pose",   device)

    hold_model  = None
    hold_onnx   = Path(HOLD_MODEL)
    hold_pt     = Path(HOLD_MODEL_PT)
    if hold_onnx.exists():
        print(f"  Loading ONNX : {HOLD_MODEL}")
        hold_model = YOLO(HOLD_MODEL, task="detect")
    elif hold_pt.exists():
        print(f"  Loading PT   : {HOLD_MODEL_PT}")
        hold_model = YOLO(HOLD_MODEL_PT)
    else:
        print("  Hold model   : NOT FOUND — hold detection disabled")
        print("  Train first  : python phase3_train.py")

    print(f"  Body weight  : {body_weight_kg} kg" if body_weight_kg
          else "  Body weight  : not set — force analysis disabled")
    print(f"  Target select: {TARGET_SELECTION}")

    # ── Open video source ─────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open source: {source}")

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_live = source in (0, "0")
    print(f"\n  Source  : {'Live camera' if is_live else source}")
    print(f"  Size    : {W}x{H} @ {fps:.1f}fps")

    # ── Output writer ─────────────────────────────────
    writer = None
    if output_path:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (W, H))
        print(f"  Saving  : {output_path}")

    # ── Detect holds once (runs only here) ────────────
    holds, grip_tracker = [], None
    if hold_model:
        print("\nDetecting holds (runs once then cached)...")
        holds       = detect_holds(hold_model, cap, W, H, device)
        grip_tracker = GripTracker(holds)

    # ── Instantiate modules ───────────────────────────
    path_tracker   = PathTracker()
    lock_detector  = LockArmDetector(fps)
    force_analyzer = ForceAnalyzer(body_weight_kg) if body_weight_kg else None
    scorers        = {}   # { person_id: ClimbScorer }
    last_colors    = {}   # { person_id: color } — remember color when lost
    last_stats     = {}   # { person_id: (arm_status, force_data, tri_status, path_len) }
    zoom_tracker   = EmergencyZoomTracker(fps)
    follow_tracker = FollowZoomTracker()
    lost_since     = defaultdict(int)  # { pid: consecutive frames not detected }

    # ── Main loop ─────────────────────────────────────
    print("\nStarting... (press Q to quit)\n")
    frame_num  = 0
    fps_cnt    = 0
    fps_timer  = time.time()
    disp_fps   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1

        # ── Phase 1: Full-frame detection (finds new climbers) ──────
        track_results = track_model.track(
            frame,
            classes=[0],
            persist=True,
            tracker="bytetrack_climbing.yaml",
            verbose=False,
            device=device,
        )

        wall_climbers  = []
        all_kps_holds  = {}
        fullframe_pids = set()   # PIDs found by full-frame this frame

        for result in track_results:
            if result.boxes is None or result.boxes.id is None:
                continue
            for i, box in enumerate(result.boxes):
                pid   = int(box.id[0])
                bxyxy = box.xyxy[0].cpu().numpy()

                if not on_wall(bxyxy, H):
                    x1,y1,x2,y2 = map(int, bxyxy)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(60,60,60),1)
                    continue

                fullframe_pids.add(pid)

                # Crop + zoom for better pose on small climbers
                x1,y1,x2,y2 = map(int, bxyxy)
                px1 = max(0, x1-int((x2-x1)*CROP_PAD))
                py1 = max(0, y1-int((y2-y1)*CROP_PAD))
                px2 = min(W, x2+int((x2-x1)*CROP_PAD))
                py2 = min(H, y2+int((y2-y1)*CROP_PAD))
                crop = frame[py1:py2, px1:px2]

                pose_results = pose_model(crop, verbose=False, device=device)

                best_kps, best_sum = None, -1
                for pr in pose_results:
                    if pr.keypoints is None: continue
                    for pi in range(len(pr.keypoints.xy)):
                        s = float(pr.keypoints.conf[pi].sum())
                        if s > best_sum:
                            best_sum = s
                            best_kps = extract_kps(
                                pr.keypoints.xy[pi],
                                pr.keypoints.conf[pi],
                                ox=px1, oy=py1)

                if best_kps is None:
                    best_kps = {}
                wall_climbers.append((pid, bxyxy, best_kps))

                lh = kp(best_kps, "left_hip")
                rh = kp(best_kps, "right_hip")
                hip      = (int((lh[0]+rh[0])/2), int((lh[1]+rh[1])/2)) if lh and rh else None
                box_size = (int(x2-x1), int(y2-y1))
                follow_tracker.update_seen(pid, hip, box_size=box_size)

        # ── Phase 2: Follow crop detection (keeps known climbers large) ──
        # For known climbers NOT found by full-frame, run detection
        # inside their follow crop so they stay large in the image
        for known_pid in list(scorers.keys()):
            if known_pid in fullframe_pids:
                continue   # already found by full-frame
            if not follow_tracker.is_following(known_pid):
                continue   # not enough history yet

            found = follow_tracker.detect_in_crop(
                known_pid, track_model, frame, W, H, device)

            if not found:
                continue

            best_box, _ = max(found, key=lambda x: x[1])
            bxyxy = np.array(best_box)

            if not on_wall(bxyxy, H):
                continue

            x1,y1,x2,y2 = map(int, bxyxy)
            px1 = max(0, x1-int((x2-x1)*CROP_PAD))
            py1 = max(0, y1-int((y2-y1)*CROP_PAD))
            px2 = min(W, x2+int((x2-x1)*CROP_PAD))
            py2 = min(H, y2+int((y2-y1)*CROP_PAD))
            crop = frame[py1:py2, px1:px2]

            pose_results = pose_model(crop, verbose=False, device=device)
            best_kps, best_sum = None, -1
            for pr in pose_results:
                if pr.keypoints is None: continue
                for pi in range(len(pr.keypoints.xy)):
                    s = float(pr.keypoints.conf[pi].sum())
                    if s > best_sum:
                        best_sum = s
                        best_kps = extract_kps(
                            pr.keypoints.xy[pi],
                            pr.keypoints.conf[pi],
                            ox=px1, oy=py1)

            if best_kps is None:
                best_kps = {}

            wall_climbers.append((known_pid, bxyxy, best_kps))

            lh = kp(best_kps, "left_hip")
            rh = kp(best_kps, "right_hip")
            hip      = (int((lh[0]+rh[0])/2), int((lh[1]+rh[1])/2)) if lh and rh else None
            box_size = (int(x2-x1), int(y2-y1))
            follow_tracker.update_seen(known_pid, hip, box_size=box_size)

            # Draw follow crop indicator
            crop_box = follow_tracker.get_crop_box(known_pid, W, H)
            if crop_box:
                col = path_color(known_pid)
                cv2.rectangle(frame,
                              (crop_box[0], crop_box[1]),
                              (crop_box[2], crop_box[3]),
                              col, 1, cv2.LINE_AA)
                cv2.putText(frame, "FOLLOW",
                            (crop_box[0]+4, crop_box[1]+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # ── ID remapping — always remap new IDs to nearest lost known climber ──
        # Only remap if the climber was lost recently — if gone too long
        # they left the wall and new detections nearby are different people.
        REMAP_DIST         = 150  # px
        MAX_LOST_FOR_REMAP = 45   # frames (~1.5s at 30fps, ~5s at 8fps)

        known_pids    = set(scorers.keys())
        detected_pids = {c[0] for c in wall_climbers}

        # Update lost_since counter
        for known_pid in known_pids:
            if known_pid in detected_pids:
                lost_since[known_pid] = 0
            else:
                lost_since[known_pid] += 1

        new_pids  = detected_pids - known_pids
        lost_pids = known_pids - detected_pids

        if new_pids and lost_pids:
            remapped = {}
            for new_pid in new_pids:
                new_box = next(c[1] for c in wall_climbers if c[0] == new_pid)
                new_cx  = (new_box[0] + new_box[2]) / 2
                new_cy  = (new_box[1] + new_box[3]) / 2

                best_match = None
                best_dist  = REMAP_DIST

                for lost_pid in lost_pids:
                    if lost_pid in remapped.values():
                        continue
                    # Skip if gone too long — they left the wall
                    if lost_since[lost_pid] > MAX_LOST_FOR_REMAP:
                        continue
                    last_pos = (path_tracker.history[lost_pid][-1]
                                if path_tracker.history[lost_pid] else None)
                    if last_pos is None:
                        continue
                    d = dist((new_cx, new_cy), last_pos)
                    if d < best_dist:
                        best_dist  = d
                        best_match = lost_pid

                if best_match is not None:
                    remapped[new_pid] = best_match

            if remapped:
                new_wall = []
                for pid, bxyxy, kps in wall_climbers:
                    if pid in remapped:
                        original_pid = remapped[pid]
                        path_tracker.history[pid] = []
                        new_wall.append((original_pid, bxyxy, kps))
                    else:
                        new_wall.append((pid, bxyxy, kps))
                wall_climbers = new_wall
                detected_pids = {c[0] for c in wall_climbers}

        # ── Update zoom tracker ───────────────────────
        # Track which known IDs were found this frame
        detected_pids = {c[0] for c in wall_climbers}
        for known_pid in list(scorers.keys()):
            if known_pid in detected_pids:
                zoom_tracker.update_seen(known_pid)
            else:
                zoom_tracker.update_lost(known_pid)

        # ── Emergency zoom recovery ───────────────────
        # For any known climber not found by normal detection,
        # try to recover them by zooming into their last known position.
        # Disabled if climbers are too close to avoid detecting wrong person.
        for known_pid in list(scorers.keys()):
            if known_pid in detected_pids:
                continue   # already found — no need for zoom
            if not zoom_tracker.should_zoom(known_pid):
                continue   # not lost long enough yet

            last_pos = (path_tracker.history[known_pid][-1]
                        if path_tracker.history[known_pid] else None)
            if last_pos is None:
                continue   # no path history yet

            # Skip zoom if any other REAL known climber (path > 300px)
            # is too close — would detect wrong person
            too_close_to_other = False
            for other_pid in scorers.keys():
                if other_pid == known_pid: continue
                if path_tracker.path_length(other_pid) < 300: continue
                other_pos = (path_tracker.history[other_pid][-1]
                             if path_tracker.history[other_pid] else None)
                if other_pos and dist(last_pos, other_pos) < 200:
                    too_close_to_other = True
                    break
            if too_close_to_other:
                continue

            # Try to find them in zoom crop
            recovered_box = zoom_tracker.try_recover(
                known_pid, track_model, frame, W, H,
                last_pos, device)

            if recovered_box is not None and on_wall(recovered_box, H):
                # Found! Run pose on the recovered box
                x1,y1,x2,y2 = map(int, recovered_box)
                px1 = max(0, x1-int((x2-x1)*CROP_PAD))
                py1 = max(0, y1-int((y2-y1)*CROP_PAD))
                px2 = min(W, x2+int((x2-x1)*CROP_PAD))
                py2 = min(H, y2+int((y2-y1)*CROP_PAD))
                crop = frame[py1:py2, px1:px2]

                pose_results = pose_model(crop, verbose=False, device=device)
                best_kps, best_sum = None, -1
                for pr in pose_results:
                    if pr.keypoints is None: continue
                    for pi in range(len(pr.keypoints.xy)):
                        s = float(pr.keypoints.conf[pi].sum())
                        if s > best_sum:
                            best_sum = s
                            best_kps = extract_kps(
                                pr.keypoints.xy[pi],
                                pr.keypoints.conf[pi],
                                ox=px1, oy=py1)

                if best_kps is None:
                    best_kps = {}

                wall_climbers.append((known_pid, recovered_box, best_kps))
                zoom_tracker.update_seen(known_pid)

                # Draw zoom indicator so we know it's active
                zx, zy   = int(last_pos[0]), int(last_pos[1])
                hw, hh   = zoom_tracker.zoom_crop_size(known_pid)
                cv2.rectangle(frame,
                              (max(0,zx-hw), max(0,zy-hh)),
                              (min(W,zx+hw), min(H,zy+hh)),
                              (0, 200, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "ZOOM RECOVERY",
                            (max(0,zx-hw)+4, max(0,zy-hh)+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 200, 255), 1)

        # ── Select target ─────────────────────────────
        target_id = select_target(wall_climbers)

        # ── Per-climber modules ───────────────────────
        active_wall    = [c for c in wall_climbers
                          if not target_id or c[0] == target_id]
        sorted_pids    = sorted([c[0] for c in active_wall])
        all_known_pids = sorted(set(list(scorers.keys()) + sorted_pids))

        for pid, bxyxy, kps in wall_climbers:
            if target_id and pid != target_id:
                continue

            color        = id_color(pid)
            badge_index  = all_known_pids.index(pid) if pid in all_known_pids else 0
            panel_index  = all_known_pids.index(pid) if pid in all_known_pids else 0
            all_kps_holds[pid] = kps
            last_colors[pid] = color   # remember color for when climber is lost

            if pid not in scorers:
                scorers[pid] = ClimbScorer(H)

            draw_skeleton(frame, kps, color)
            draw_person_box(frame, bxyxy, pid, color)

            path_tracker.update(pid, kps)
            path_tracker.draw(frame, pid, color=path_color(pid))
            path_len   = path_tracker.path_length(pid)
            hip_pos    = path_tracker.history[pid][-1] if path_tracker.history[pid] else None

            tri_status = draw_triangle(frame, kps, bxyxy=bxyxy)

            arm_status = lock_detector.update(pid, kps)
            lock_detector.draw(frame, pid, kps, arm_status)

            force_data = force_analyzer.analyze(kps) if force_analyzer else None

            # Update scorer
            scorers[pid].update(tri_status, arm_status, force_data, hip_pos, path_len)

            # Store last known stats for persistence when climber is lost
            last_stats[pid] = (arm_status, force_data, tri_status, path_len)

            draw_stats_panel(frame, pid, arm_status, force_data,
                             tri_status, path_len, color, force_analyzer,
                             panel_index=panel_index)

            scorers[pid].draw_live(frame, pid, color, badge_index)

        # ── Persist path + panels + badges for lost climbers ───
        for pid in all_known_pids:
            if pid not in sorted_pids:
                col = last_colors.get(pid, id_color(pid))
                i   = all_known_pids.index(pid)
                path_tracker.draw(frame, pid, color=path_color(pid))
                if pid in last_stats:
                    arm_s, force_d, tri_s, path_l = last_stats[pid]
                    draw_stats_panel(frame, pid, arm_s, force_d,
                                     tri_s, path_l, col, force_analyzer,
                                     panel_index=i)
                scorers[pid].draw_live(frame, pid, col, i)

        # ── Hold grip detection ───────────────────────
        if grip_tracker:
            grip_tracker.update(all_kps_holds)
            grip_tracker.draw(frame)
            draw_legend(frame)

        # ── FPS + info panel ──────────────────────────
        fps_cnt += 1
        if time.time() - fps_timer >= 1.0:
            disp_fps  = fps_cnt / (time.time() - fps_timer)
            fps_cnt   = 0
            fps_timer = time.time()

        active_ids = sorted_pids
        draw_info_panel(frame, frame_num, disp_fps, active_ids, device_name)

        if writer: writer.write(frame)

        cv2.imshow("Climbing AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user.")
            break

        if frame_num % 100 == 0 and not is_live:
            pct = frame_num/total*100 if total > 0 else 0
            print(f"  [{pct:5.1f}%] frame {frame_num:5d}  "
                  f"ids={len(active_ids)}  fps={disp_fps:.1f}  "
                  f"device={device_name}")

    # ── Final score overlay on last written frame ─────
    if writer and scorers:
        final_frame  = np.zeros((H, W, 3), dtype=np.uint8)
        # Filter out ghost tracks — only show climbers with meaningful path
        MIN_PATH_PX  = 300   # ignore any ID with path shorter than this
        sorted_pids  = sorted([
            pid for pid in scorers.keys()
            if path_tracker.path_length(pid) >= MIN_PATH_PX
        ])
        if not sorted_pids:   # fallback if all paths are short
            sorted_pids = sorted(scorers.keys())
        total_cards  = len(sorted_pids)
        # Header
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(final_frame, "CLIMB COMPLETE",
                    (W//2-140, H//2-210), font, 1.0, (200,200,200), 2)
        for i, pid in enumerate(sorted_pids):
            col = last_colors.get(pid, id_color(pid))
            scorers[pid].draw_final(final_frame, pid, col,
                                    card_index=i, total_cards=total_cards)
        for _ in range(int(fps * 3)):
            writer.write(final_frame)
        cv2.imshow("Climbing AI", final_frame)
        cv2.waitKey(2000)

    # ── Cleanup ───────────────────────────────────────
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    # ── Session summary ───────────────────────────────
    print("\n" + "="*55)
    print("  Session Summary")
    print("="*55)
    for pid in sorted(p for p in path_tracker.history.keys() if path_tracker.path_length(p) >= 300):
        print(f"\n  Climber ID {pid}:")
        print(f"    Path length : {path_tracker.path_length(pid):.0f} px")
        hp = path_tracker.highest_point(pid)
        if hp: print(f"    Highest Y   : {hp[1]} px from top")
        if pid in scorers:
            overall, bd = scorers[pid].compute()
            print(f"\n    ── Score ──────────────────────────")
            print(f"    OVERALL        : {overall} / 100")
            print(f"    Triangle       : {bd['triangle']}")
            print(f"    Arm technique  : {bd['arm']}")
    if grip_tracker:
        r = grip_tracker.session_report()
        print(f"\n  Hold report:")
        print(f"    Total on wall : {r['total_holds']}")
        print(f"    Hand holds    : {r['hand_holds']}")
        print(f"    Foot holds    : {r['foot_holds']}")
        print(f"    Both (H+F)    : {r['both_holds']}")
    if output_path:
        print(f"\n  Video saved → {output_path}")

# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Climbing AI — Real-Time")
    parser.add_argument("--source", "-s", default=DEFAULT_SOURCE,
                        help="Camera index (0) or video file path")
    parser.add_argument("--weight", "-w", type=float, default=BODY_WEIGHT_KG,
                        help="Climber body weight in kg")
    parser.add_argument("--output", "-o", default=None,
                        help="Save annotated output video")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    run(source, body_weight_kg=args.weight, output_path=args.output)