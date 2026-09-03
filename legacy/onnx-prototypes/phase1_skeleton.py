"""
Climbing AI - Phase 1 v3: Tracking + Persistent Person IDs
============================================================
What's new vs v2:
  - ByteTrack persistent IDs: each climber keeps the same ID across all frames
  - Wall filter: ignores people standing on the ground (bottom 20% of frame)
  - Tracks ALL climbers on the wall simultaneously with separate IDs
  - JSON now stores keypoints per person ID (ready for Phase 2 scoring)
  - Each tracked ID gets a unique color for easy visual distinction
  - Selection logic (which climber to score) left as TODO — pending client input

Usage:
    python phase1_skeleton.py --input your_video.MOV

Requirements:
    pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
import json
import argparse
import time
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

CONFIDENCE_THRESHOLD  = 0.3    # keypoint visibility threshold
CROP_PADDING          = 0.20   # expand person box by this % before pose estimation
MIN_PERSON_HEIGHT_PX  = 25     # ignore detections shorter than this (pixels)
GROUND_FILTER_RATIO   = 0.80   # ignore anyone whose center Y > this * frame height
                                # i.e. bottom 20% = ground/bystanders


# ─────────────────────────────────────────────
#  KEYPOINT MAP  (COCO 17-point)
# ─────────────────────────────────────────────

KEYPOINTS = {
    0:"nose",        1:"left_eye",       2:"right_eye",
    3:"left_ear",    4:"right_ear",      5:"left_shoulder",
    6:"right_shoulder", 7:"left_elbow",  8:"right_elbow",
    9:"left_wrist",  10:"right_wrist",   11:"left_hip",
    12:"right_hip",  13:"left_knee",     14:"right_knee",
    15:"left_ankle", 16:"right_ankle",
}

SKELETON_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

CRITICAL_KEYPOINTS = {9, 10, 15, 16}   # wrists + ankles

# Unique colors per tracked person ID (BGR) — cycles if more than 8 people
ID_COLORS = [
    (100, 220, 255),   # ID 1 — yellow
    (100, 255, 150),   # ID 2 — green
    (255, 150, 100),   # ID 3 — blue
    (180, 100, 255),   # ID 4 — purple
    (100, 180, 255),   # ID 5 — orange
    (255, 100, 180),   # ID 6 — pink
    (150, 255, 220),   # ID 7 — teal
    (200, 200, 100),   # ID 8 — cyan
]

def get_id_color(person_id):
    return ID_COLORS[(person_id - 1) % len(ID_COLORS)]


# ─────────────────────────────────────────────
#  WALL FILTER
# ─────────────────────────────────────────────

def is_on_wall(box_xyxy, frame_height):
    """
    Returns True if the person's center Y is above the ground threshold.
    People standing on the ground have their center in the bottom 20% of frame.

    TODO: Adjust GROUND_FILTER_RATIO if your camera angle is different.
    """
    x1, y1, x2, y2 = box_xyxy
    center_y = (y1 + y2) / 2
    return center_y < (frame_height * GROUND_FILTER_RATIO)


def is_large_enough(box_xyxy):
    """Filter out tiny false detections."""
    x1, y1, x2, y2 = box_xyxy
    return (y2 - y1) >= MIN_PERSON_HEIGHT_PX


# ─────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────

def draw_skeleton(frame, kpts_xy, kpts_conf, color, ox=0, oy=0, thickness=2):
    """
    Draw skeleton using the person's assigned ID color.
    ox/oy offset maps crop-space coordinates back to full frame.
    """
    kpts  = kpts_xy.cpu().numpy()
    confs = kpts_conf.cpu().numpy()

    for p1, p2 in SKELETON_CONNECTIONS:
        if confs[p1] >= CONFIDENCE_THRESHOLD and confs[p2] >= CONFIDENCE_THRESHOLD:
            pt1 = (int(kpts[p1][0]) + ox, int(kpts[p1][1]) + oy)
            pt2 = (int(kpts[p2][0]) + ox, int(kpts[p2][1]) + oy)
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    for i, (xy, conf) in enumerate(zip(kpts, confs)):
        if conf >= CONFIDENCE_THRESHOLD:
            cx, cy = int(xy[0]) + ox, int(xy[1]) + oy
            if i in CRITICAL_KEYPOINTS:
                cv2.circle(frame, (cx, cy), 7, color,        -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 7, (255,255,255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (cx, cy), 4, (255,255,255),-1, cv2.LINE_AA)


def draw_person_box(frame, box_xyxy, person_id, color):
    """Draw bounding box with ID label in the person's color."""
    x1, y1, x2, y2 = map(int, box_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    label = f"ID {person_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
    cv2.putText(frame, label, (x1+3, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20,20,20), 1)


def draw_hip_midpoint(frame, person_data, color):
    """Draw magenta dot at hip midpoint — preview for Phase 2 path tracking."""
    lh = person_data["keypoints"]["left_hip"]
    rh = person_data["keypoints"]["right_hip"]
    if lh["visible"] and rh["visible"]:
        mx = int((lh["x"] + rh["x"]) / 2)
        my = int((lh["y"] + rh["y"]) / 2)
        cv2.circle(frame, (mx, my), 6, (255, 0, 255), -1)
        cv2.circle(frame, (mx, my), 6, (255,255,255),  1)
        return (mx, my)
    return None


def draw_info_panel(frame, frame_num, fps, active_ids):
    """Top-left panel: frame, fps, list of active tracked IDs."""
    n_ids     = len(active_ids)
    panel_h   = 50 + n_ids * 18
    overlay   = frame.copy()
    cv2.rectangle(overlay, (10, 10), (230, 10 + panel_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Frame {frame_num}  |  {fps:.1f} FPS",
                (18, 30), font, 0.42, (200,200,200), 1)
    cv2.putText(frame, f"Climbers on wall: {n_ids}",
                (18, 48), font, 0.42, (100,255,150), 1)

    y = 66
    for pid in sorted(active_ids):
        col = get_id_color(pid)
        cv2.putText(frame, f"  ID {pid}", (18, y), font, 0.38, col, 1)
        y += 18


# ─────────────────────────────────────────────
#  KEYPOINT EXTRACTION
# ─────────────────────────────────────────────

def extract_keypoints_for_person(kpts_xy, kpts_conf, ox=0, oy=0):
    """
    Build the keypoint dict for one person.
    ox/oy shift crop-space coords back to full frame space.
    """
    kpts_np = kpts_xy.cpu().numpy()
    conf_np = kpts_conf.cpu().numpy()
    kp_dict = {}
    for kp_idx, name in KEYPOINTS.items():
        cv = float(conf_np[kp_idx])
        kp_dict[name] = {
            "x":          float(kpts_np[kp_idx][0]) + ox,
            "y":          float(kpts_np[kp_idx][1]) + oy,
            "confidence": cv,
            "visible":    cv >= CONFIDENCE_THRESHOLD,
        }
    return kp_dict


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def process_video(input_path, output_path, model_size="s"):
    """
    Per-frame pipeline:
      1. Run YOLO track() on full frame → get boxes + persistent IDs
      2. Filter: keep only people ON the wall (not standing on ground)
      3. For each wall person: crop + zoom → run pose model on crop
      4. Store keypoints per person ID in JSON
      5. Draw everything on original frame
    """
    print(f"\n[1/4] Loading models (YOLO11-{model_size})...")
    # Tracking model: detection + ByteTrack IDs
    track_model = YOLO(f"yolo11{model_size}.pt")
    # Pose model: keypoint estimation on cropped region
    pose_model  = YOLO(f"yolo11{model_size}-pose.pt")

    print(f"\n[2/4] Opening: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"    {W}x{H} @ {fps:.1f}fps | {total} frames")
    print(f"    Wall filter: ignoring anyone with center Y > {GROUND_FILTER_RATIO*100:.0f}% of frame")

    out       = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    json_path = str(Path(output_path).with_suffix("")) + "_keypoints.json"

    # JSON structure: { "frame_N": { "person_ID": { keypoints... } } }
    # This is the format Phase 2 expects
    all_data  = []

    fps_cnt   = 0
    fps_timer = time.time()
    disp_fps  = 0.0
    frame_num = 0

    # Confidence tracking per ID for final report
    id_conf_history = {}   # { id: { joint_name: [conf, conf, ...] } }

    print(f"\n[3/4] Processing {total} frames with ByteTrack...")
    print(f"      Each person gets a persistent ID across all frames\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        frame_data = {
            "frame":   frame_num,
            "climbers": {}         # keyed by person ID (string)
        }
        active_ids = []

        # ── Step 1: Track all people in full frame ──────
        # model.track() uses ByteTrack to assign persistent IDs
        track_results = track_model.track(
            frame,
            classes=[0],        # person class only
            persist=True,       # maintain track state across frames
            tracker="bytetrack_climbing.yaml",
            verbose=False,
        )

        # ── Step 2: Filter + process each tracked person ─
        for result in track_results:
            if result.boxes is None or result.boxes.id is None:
                continue

            for i, box in enumerate(result.boxes):
                # Get persistent track ID
                person_id  = int(box.id[0])
                box_coords = box.xyxy[0].cpu().numpy()   # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box_coords)

                # ── Wall filter ────────────────────────
                if not is_large_enough(box_coords):
                    continue
                if not is_on_wall(box_coords, H):
                    # Draw a faint gray box for filtered-out people
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (80,80,80), 1)
                    cv2.putText(frame, "ground", (x1, y1-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,80), 1)
                    continue

                # ── Crop + zoom for pose estimation ───
                pad_x = int((x2-x1) * CROP_PADDING)
                pad_y = int((y2-y1) * CROP_PADDING)
                cx1   = max(0, x1-pad_x)
                cy1   = max(0, y1-pad_y)
                cx2   = min(W, x2+pad_x)
                cy2   = min(H, y2+pad_y)
                crop  = frame[cy1:cy2, cx1:cx2]

                pose_results = pose_model(crop, verbose=False)

                # ── Extract best person from crop ──────
                best_kpts_xy   = None
                best_kpts_conf = None
                best_conf_sum  = -1

                for pr in pose_results:
                    if pr.keypoints is None:
                        continue
                    for p_idx in range(len(pr.keypoints.xy)):
                        kpts_conf = pr.keypoints.conf[p_idx].cpu().numpy()
                        conf_sum  = float(kpts_conf.sum())
                        if conf_sum > best_conf_sum:
                            best_conf_sum  = conf_sum
                            best_kpts_xy   = pr.keypoints.xy[p_idx]
                            best_kpts_conf = pr.keypoints.conf[p_idx]

                if best_kpts_xy is None:
                    continue

                # ── Extract keypoints (map back to full frame) ──
                person_color  = get_id_color(person_id)
                person_kp     = extract_keypoints_for_person(
                                    best_kpts_xy, best_kpts_conf,
                                    ox=cx1, oy=cy1)

                # ── Draw skeleton + box ────────────────
                draw_skeleton(frame, best_kpts_xy, best_kpts_conf,
                              person_color, ox=cx1, oy=cy1)
                draw_person_box(frame, (x1,y1,x2,y2), person_id, person_color)

                # Draw hip midpoint (magenta) — Phase 2 path tracking preview
                draw_hip_midpoint(frame, {"keypoints": person_kp}, person_color)

                # ── Store in frame data ────────────────
                frame_data["climbers"][str(person_id)] = {
                    "track_id":  person_id,
                    "box":       [x1, y1, x2, y2],
                    "keypoints": person_kp,
                }
                active_ids.append(person_id)

                # Track confidence history for final report
                if person_id not in id_conf_history:
                    id_conf_history[person_id] = {name: [] for name in KEYPOINTS.values()}
                for name, kp in person_kp.items():
                    id_conf_history[person_id][name].append(kp["confidence"])

        # ── Info panel ─────────────────────────────────
        fps_cnt += 1
        if time.time() - fps_timer >= 1.0:
            disp_fps  = fps_cnt / (time.time() - fps_timer)
            fps_cnt   = 0
            fps_timer = time.time()

        draw_info_panel(frame, frame_num, disp_fps, active_ids)
        out.write(frame)
        all_data.append(frame_data)

        if frame_num % 100 == 0 or frame_num == 1:
            pct      = frame_num / total * 100 if total > 0 else 0
            ids_str  = str(sorted(active_ids)) if active_ids else "none"
            print(f"  [{pct:5.1f}%] frame {frame_num:5d}  "
                  f"wall_ids={ids_str}")

    # ── Cleanup ────────────────────────────────────────
    cap.release()
    out.release()

    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)

    # ── Summary ───────────────────────────────────────
    print(f"\n[4/4] Done!")
    print(f"    Annotated video → {output_path}")
    print(f"    Keypoint JSON   → {json_path}")
    print(f"    Total frames    : {frame_num}")
    print(f"    Unique IDs seen : {sorted(id_conf_history.keys())}")

    print("\n── Per-ID Keypoint Confidence Report ───────────────────────")
    crit = {
        "left_wrist": 9, "right_wrist": 10,
        "left_ankle": 15, "right_ankle": 16
    }
    for pid in sorted(id_conf_history.keys()):
        print(f"\n  Climber ID {pid}:")
        for name in crit:
            confs  = id_conf_history[pid].get(name, [])
            if not confs:
                continue
            vis    = sum(1 for c in confs if c >= CONFIDENCE_THRESHOLD)
            pct    = vis / len(confs) * 100
            status = "✓ GOOD" if pct > 60 else ("⚠ LOW" if pct > 30 else "✗ POOR")
            print(f"    {name:<15}: {pct:5.1f}%  {status}")

    print("\n── JSON Structure ───────────────────────────────────────────")
    print("  Each frame: { 'frame': N, 'climbers': { 'ID': { keypoints } } }")
    print("  Phase 2 reads this JSON to build path/triangle/score per climber")

    # ── Selection TODO note ───────────────────────────
    print("\n── TODO (pending client input) ──────────────────────────────")
    print("  Which climber ID to score? Options:")
    print("  A) Highest on wall (smallest Y of hip midpoint)")
    print("  B) Largest in frame (biggest bounding box area)")
    print("  C) User selects ID before session starts")
    print("  → Add 'TARGET_ID = X' to config once client decides")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Climbing AI — Phase 1 v3 (Tracking)")
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--model",  "-m", default="s",
                        choices=["n","s","m","l","x"])
    args = parser.parse_args()

    p           = Path(args.input)
    output_path = args.output or str(p.parent / f"{p.stem}_annotated.mp4")

    process_video(args.input, output_path, model_size=args.model)