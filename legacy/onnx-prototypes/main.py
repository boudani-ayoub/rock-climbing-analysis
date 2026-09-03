"""
Climbing AI — Real-Time Analysis
=================================
Detects and tracks climbers, estimates pose, analyzes technique,
detects holds, and produces a scored summary report.

Usage:
    python main2.py --source "your_video.MOV" --weight 45
    python main2.py                               # live camera
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

TRACK_MODEL    = "yolo11m.onnx"
POSE_MODEL     = "yolo11m-pose.onnx"
HOLD_MODEL     = "runs/hold_detection/weights/best.onnx"

TRACK_MODEL_PT = "yolo11m.pt"
POSE_MODEL_PT  = "yolo11m-pose.pt"
HOLD_MODEL_PT  = "runs/hold_detection/weights/best.pt"

DEFAULT_SOURCE   = 0
TARGET_SELECTION = None
BODY_WEIGHT_KG   = None

GROUND_FILTER    = 0.80
MIN_PERSON_H     = 30

KP_CONF          = 0.25
CROP_PAD         = 0.20

HOLD_DETECT_FRAMES  = 5
HOLD_CONF           = 0.35
HOLD_NMS_DIST       = 30
GRIP_RADIUS         = 50
GRIP_CONFIRM_FRAMES = 8

TRI_GOOD_AREA    = 25000
TRI_WARN_AREA    = 8000

LOCK_ANGLE       = 110
LOCK_SECONDS     = 3.0

GRAVITY          = 9.81


# ═══════════════════════════════════════════════════════
#  DEVICE SETUP
# ═══════════════════════════════════════════════════════

def setup_device():
    print("\n  Device: CPU")
    return "cpu", "CPU"


def load_model(onnx_path, pt_path, task, device):
    if Path(onnx_path).exists():
        print(f"  Loading ONNX : {onnx_path}")
        return YOLO(onnx_path, task=task)
    elif Path(pt_path).exists():
        print(f"  Loading PT   : {pt_path}")
        return YOLO(pt_path)
    else:
        raise FileNotFoundError(
            f"Model not found.\n  ONNX: {onnx_path}\n  PT  : {pt_path}"
        )


# ═══════════════════════════════════════════════════════
#  COLORS (BGR)
# ═══════════════════════════════════════════════════════

ID_COLORS = [
    (100,220,255),(100,255,150),(255,150,100),
    (180,100,255),(100,180,255),(255,100,180),
]
def id_color(pid): return ID_COLORS[(pid-1) % len(ID_COLORS)]

COL_PATH      = (255,  60, 255)
COL_TRI_GOOD  = ( 50, 220,  50)
COL_TRI_OK    = ( 50, 200, 220)
COL_TRI_WARN  = ( 50,  50, 255)
COL_HOLD_IDLE = (160, 160, 160)
COL_HOLD_HAND = (  0, 165, 255)
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
    if len(pts) < 17: return
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

    def draw(self, frame, pid):
        pts = self.history[pid]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], COL_PATH, 3, cv2.LINE_AA)
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

def draw_triangle(frame, kps):
    lh = kp(kps,"left_wrist")  or kp(kps,"left_elbow")
    rh = kp(kps,"right_wrist") or kp(kps,"right_elbow")
    la = kp(kps,"left_ankle")  or kp(kps,"left_knee")
    ra = kp(kps,"right_ankle") or kp(kps,"right_knee")

    hands = [p for p in [lh,rh] if p]
    feet  = [p for p in [la,ra] if p]
    if len(hands) < 2 or not feet:
        return "unknown"

    p1, p2 = hands[0], hands[1]
    foot   = min(feet, key=lambda f: f[1])
    area   = tri_area(p1, p2, foot)

    status = ("good" if area >= TRI_GOOD_AREA
              else "ok" if area >= TRI_WARN_AREA
              else "warn")
    color  = {"good":COL_TRI_GOOD,"ok":COL_TRI_OK,
               "warn":COL_TRI_WARN}.get(status, (120,120,120))

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
        self.thresh = int(LOCK_SECONDS * fps)
        self.frames = defaultdict(lambda: {"left":0,"right":0})

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
            ex, ey    = int(el[0]), int(el[1])
            f         = self.frames[pid][side]
            if f > 0:
                pct       = min(f / self.thresh, 1.0)
                warning   = pct >= 1.0
                color     = (50, 50, 255) if warning else (50, 180, 255)
                radius    = 26 if warning else int(14 + pct * 6)
                thickness = 3 if warning else 2
                cv2.ellipse(frame, (ex, ey), (radius, radius), -90,
                            0, int(pct * 360), color, thickness, cv2.LINE_AA)
                if warning:
                    cv2.putText(frame, "Stretch arm!",
                                (ex + radius + 5, ey + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (50, 50, 255), 2, cv2.LINE_AA)
        if arm_status.get("warning"):
            H = frame.shape[0]
            cv2.rectangle(frame, (0, H-50), (frame.shape[1], H), (0, 0, 180), -1)
            cv2.putText(frame, "WARNING: Lock arm — straighten your arms!",
                        (15, H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


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
    """
    Live score (0-100) updated every frame.
    Technical (60%): triangle stability, arm technique, force balance.
    Performance (40%): path efficiency.
    """

    def __init__(self, frame_height):
        self.H           = frame_height
        self.tri_good    = 0
        self.tri_total   = 0
        self.lock_frames = 0
        self.arm_total   = 0
        self.force_diffs = []
        self.start_y     = None
        self.best_y      = None
        self.path_len    = 0.0
        self.prev_pos    = None

    def update(self, tri_status, arm_status, force_data, hip_pos, path_length):
        if tri_status != "unknown":
            self.tri_total += 1
            if tri_status in ("good", "ok"):
                self.tri_good += 1

        self.arm_total += 1
        if arm_status.get("warning"):
            self.lock_frames += 1

        if force_data:
            diff = abs(force_data["pct_left"] - force_data["pct_right"])
            self.force_diffs.append(diff)

        if hip_pos:
            cy = hip_pos[1]
            if self.start_y is None:
                self.start_y = cy
            if self.best_y is None or cy < self.best_y:
                self.best_y = cy

        self.path_len = path_length

    def compute(self):
        tri_score = (self.tri_good / self.tri_total * 100) if self.tri_total > 0 else 50.0

        if self.arm_total > 0:
            lock_ratio = self.lock_frames / self.arm_total
            arm_score  = max(0, 100 - lock_ratio * 300)
        else:
            arm_score = 100.0

        if self.force_diffs:
            avg_diff    = sum(self.force_diffs) / len(self.force_diffs)
            force_score = max(0, 100 - avg_diff * 2)
        else:
            force_score = 50.0

        technical = (tri_score * 0.25 + arm_score * 0.20 + force_score * 0.15) / 0.60

        if self.start_y and self.best_y and self.path_len > 0:
            efficiency_score = min(1.0, abs(self.start_y - self.best_y) / self.path_len) * 100
        else:
            efficiency_score = 50.0

        overall = round(technical * 0.60 + efficiency_score * 0.40)

        return overall, {
            "triangle":   round(tri_score),
            "arm":        round(arm_score),
            "force_bal":  round(force_score),
            "efficiency": round(efficiency_score),
            "technical":  round(technical),
            "performance":round(efficiency_score),
        }

    def draw_live(self, frame, pid, id_color, badge_index=0):
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
        cv2.putText(frame, "/100", (px+12, py+82), font, 0.5, (180,180,180), 1)
        cv2.putText(frame, f"ID {pid}", (px+12, py+100), font, 0.4, id_color, 1)

        items = [
            ("TRI", breakdown["triangle"],  COL_TRI_GOOD),
            ("ARM", breakdown["arm"],       (100,220,255)),
            ("BAL", breakdown["force_bal"], (255,180,100)),
            ("EFF", breakdown["efficiency"],(100,255,200)),
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
        overall, bd = self.compute()
        H, W = frame.shape[:2]

        card_w  = 420
        gap     = 20
        total_w = total_cards * card_w + (total_cards - 1) * gap
        start_x = (W - total_w) // 2
        cx      = start_x + card_index * (card_w + gap) + card_w // 2
        card_x1 = cx - card_w // 2
        card_x2 = cx + card_w // 2
        card_y1 = H // 2 - 190
        card_y2 = H // 2 + 210

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
            ("Technical",    bd["technical"],   True),
            ("  Triangle",   bd["triangle"],    False),
            ("  Arm tech",   bd["arm"],         False),
            ("  Balance",    bd["force_bal"],   False),
            ("Performance",  bd["performance"], True),
            ("  Efficiency", bd["efficiency"],  False),
        ]
        y     = card_y1 + 142
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
            cv2.putText(frame, str(val), (bar_x+bar_w+6, y), font, 0.36, col, 1)
            y += 20 if is_section else 17


# ═══════════════════════════════════════════════════════
#  STATS PANEL
# ═══════════════════════════════════════════════════════

def draw_stats_panel(frame, pid, arm_status, force_data,
                     tri_status, path_len, color, force_analyzer):
    W       = frame.shape[1]
    px, py  = W-200, 10
    ov      = frame.copy()
    cv2.rectangle(ov, (px,py), (px+190,py+185), (15,15,15), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
    f = cv2.FONT_HERSHEY_SIMPLEX
    x, y = px+8, py+18
    cv2.putText(frame, f"Climber ID {pid}", (x,y), f, 0.45, color, 1)
    y += 22
    tri_c = {"good":COL_TRI_GOOD,"ok":COL_TRI_OK,
              "warn":COL_TRI_WARN,"unknown":(120,120,120)}
    tri_l = {"good":"STABLE","ok":"OK","warn":"NARROW!","unknown":"---"}
    cv2.putText(frame, f"Triangle: {tri_l.get(tri_status,'---')}",
                (x,y), f, 0.4, tri_c.get(tri_status,(120,120,120)), 1)
    y += 20
    for side in ["left","right"]:
        s   = arm_status.get(side,"unknown")
        col = (50,50,255) if s=="bent" else (50,220,50) if s=="straight" else (120,120,120)
        cv2.putText(frame, f"{side.capitalize()} arm: {s}", (x,y), f, 0.38, col, 1)
        y += 18
    cv2.line(frame, (x,y), (px+182,y), (60,60,60), 1); y += 10
    cv2.putText(frame, "Hand load:", (x,y), f, 0.38, (180,180,180), 1); y += 14
    if force_analyzer:
        force_analyzer.draw_bar(frame, force_data, x, y)
    y += 28
    cv2.line(frame, (x,y), (px+182,y), (60,60,60), 1); y += 10
    cv2.putText(frame, f"Path: {path_len:.0f}px", (x,y), f, 0.38, (200,200,200), 1)

def draw_legend(frame):
    H     = frame.shape[0]
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
#  HOLD DETECTION + GRIP TRACKER
# ═══════════════════════════════════════════════════════

def detect_holds(hold_model, cap, W, H, device):
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
        self.status   = ["idle"] * len(holds)
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
                    if self.status[idx] in ("hand","foot"): continue
                    if dist(pos, hold["center"]) < GRIP_RADIUS:
                        active[idx].add(limb)

        for idx in range(len(self.holds)):
            if self.status[idx] in ("hand","foot"): continue
            limbs = active.get(idx, set())
            if limbs:
                for limb in limbs:
                    self.counter[idx][limb] += 1
                    if self.counter[idx][limb] >= GRIP_CONFIRM_FRAMES:
                        self.status[idx] = self.limb_map[limb]
            else:
                self.counter[idx].clear()

    def draw(self, frame):
        for idx, hold in enumerate(self.holds):
            cx, cy = int(hold["center"][0]), int(hold["center"][1])
            if hold["class"] == 1:
                x1,y1,x2,y2 = hold["box"]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(120,120,60),1)
                continue
            s = self.status[idx]
            if s == "hand":
                cv2.circle(frame,(cx,cy),14,COL_HOLD_HAND,-1,cv2.LINE_AA)
                cv2.circle(frame,(cx,cy),14,(255,255,255),1,cv2.LINE_AA)
            elif s == "foot":
                cv2.circle(frame,(cx,cy),14,COL_HOLD_FOOT,-1,cv2.LINE_AA)
                cv2.circle(frame,(cx,cy),14,(255,255,255),1,cv2.LINE_AA)
            else:
                cv2.circle(frame,(cx,cy),10,COL_HOLD_IDLE,1,cv2.LINE_AA)

    def session_report(self):
        return {
            "total_holds": len(self.holds),
            "hand_holds":  sum(1 for s in self.status if s=="hand"),
            "foot_holds":  sum(1 for s in self.status if s=="foot"),
        }


# ═══════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════

def run(source, body_weight_kg=None, output_path=None):
    print("\n" + "="*55)
    print("  Climbing AI — Real-Time Analysis")
    print("="*55)

    device, device_name = setup_device()

    print("\nLoading models...")
    track_model = load_model(TRACK_MODEL, TRACK_MODEL_PT, "detect", device)
    pose_model  = load_model(POSE_MODEL,  POSE_MODEL_PT,  "pose",   device)

    hold_model = None
    if Path(HOLD_MODEL).exists():
        print(f"  Loading ONNX : {HOLD_MODEL}")
        hold_model = YOLO(HOLD_MODEL, task="detect")
    elif Path(HOLD_MODEL_PT).exists():
        print(f"  Loading PT   : {HOLD_MODEL_PT}")
        hold_model = YOLO(HOLD_MODEL_PT)
    else:
        print("  Hold model   : NOT FOUND — hold detection disabled")

    print(f"  Body weight  : {body_weight_kg} kg" if body_weight_kg
          else "  Body weight  : not set — force analysis disabled")

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

    writer = None
    if output_path:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        print(f"  Saving  : {output_path}")

    holds, grip_tracker = [], None
    if hold_model:
        print("\nDetecting holds (runs once then cached)...")
        holds        = detect_holds(hold_model, cap, W, H, device)
        grip_tracker = GripTracker(holds)

    path_tracker   = PathTracker()
    lock_detector  = LockArmDetector(fps)
    force_analyzer = ForceAnalyzer(body_weight_kg) if body_weight_kg else None
    scorers        = {}
    last_colors    = {}

    print("\nStarting... (press Q to quit)\n")
    frame_num = 0
    fps_cnt   = 0
    fps_timer = time.time()
    disp_fps  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1

        track_results = track_model.track(
            frame,
            classes=[0],
            persist=True,
            tracker="bytetrack_climbing.yaml",
            verbose=False,
            device=device,
        )

        wall_climbers = []
        all_kps_holds = {}

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

        target_id   = select_target(wall_climbers)
        active_wall = [c for c in wall_climbers
                       if not target_id or c[0] == target_id]
        sorted_pids = sorted([c[0] for c in active_wall])

        for pid, bxyxy, kps in wall_climbers:
            if target_id and pid != target_id:
                continue

            color       = id_color(pid)
            badge_index = sorted_pids.index(pid)
            all_kps_holds[pid] = kps
            last_colors[pid]   = color

            if pid not in scorers:
                scorers[pid] = ClimbScorer(H)

            draw_skeleton(frame, kps, color)
            draw_person_box(frame, bxyxy, pid, color)

            path_tracker.update(pid, kps)
            path_tracker.draw(frame, pid)
            path_len = path_tracker.path_length(pid)
            hip_pos  = path_tracker.history[pid][-1] if path_tracker.history[pid] else None

            tri_status = draw_triangle(frame, kps)
            arm_status = lock_detector.update(pid, kps)
            lock_detector.draw(frame, pid, kps, arm_status)
            force_data = force_analyzer.analyze(kps) if force_analyzer else None

            scorers[pid].update(tri_status, arm_status, force_data, hip_pos, path_len)
            draw_stats_panel(frame, pid, arm_status, force_data,
                             tri_status, path_len, color, force_analyzer)
            scorers[pid].draw_live(frame, pid, color, badge_index)

        all_known_pids = sorted(scorers.keys())
        for i, pid in enumerate(all_known_pids):
            if pid not in sorted_pids:
                col = last_colors.get(pid, id_color(pid))
                scorers[pid].draw_live(frame, pid, col, i)

        if grip_tracker:
            grip_tracker.update(all_kps_holds)
            grip_tracker.draw(frame)
            draw_legend(frame)

        fps_cnt += 1
        if time.time() - fps_timer >= 1.0:
            disp_fps  = fps_cnt / (time.time() - fps_timer)
            fps_cnt   = 0
            fps_timer = time.time()

        draw_info_panel(frame, frame_num, disp_fps, sorted_pids, device_name)

        if writer: writer.write(frame)
        cv2.imshow("Climbing AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user.")
            break

        if frame_num % 100 == 0 and not is_live:
            pct = frame_num/total*100 if total > 0 else 0
            print(f"  [{pct:5.1f}%] frame {frame_num:5d}  "
                  f"ids={len(sorted_pids)}  fps={disp_fps:.1f}")

    if writer and scorers:
        final_frame = np.zeros((H, W, 3), dtype=np.uint8)
        sorted_pids = sorted(scorers.keys())
        total_cards = len(sorted_pids)
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

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    print("\n" + "="*55)
    print("  Session Summary")
    print("="*55)
    for pid in sorted(path_tracker.history.keys()):
        print(f"\n  Climber ID {pid}:")
        print(f"    Path length : {path_tracker.path_length(pid):.0f} px")
        hp = path_tracker.highest_point(pid)
        if hp: print(f"    Highest Y   : {hp[1]} px from top")
        if pid in scorers:
            overall, bd = scorers[pid].compute()
            print(f"\n    ── Score ──────────────────────────")
            print(f"    OVERALL          : {overall} / 100")
            print(f"    Technical (60%)  : {bd['technical']}")
            print(f"      Triangle       : {bd['triangle']}")
            print(f"      Arm technique  : {bd['arm']}")
            print(f"      Force balance  : {bd['force_bal']}")
            print(f"    Performance (40%): {bd['performance']}")
            print(f"      Path efficiency: {bd['efficiency']}")
    if grip_tracker:
        r = grip_tracker.session_report()
        print(f"\n  Hold report:")
        print(f"    Total on wall : {r['total_holds']}")
        print(f"    Hand holds    : {r['hand_holds']}")
        print(f"    Foot holds    : {r['foot_holds']}")
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