import os
import cv2
import math
import itertools
import subprocess
from ultralytics import YOLO
from collections import deque
import mediapipe as mp

# =========================
# YOLO
# =========================
model = YOLO("yolov8n.pt")

# 👉 METTI QUI gli ID quando hai modello armi
WEAPON_CLASS_IDS = []

# =========================
# MEDIAPIPE (NUOVA API)
# =========================
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE
)

pose_detector = PoseLandmarker.create_from_options(options)

def detect_aggressive_pose(frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = pose_detector.detect(mp_image)

    if not result.pose_landmarks:
        return False

    landmarks = result.pose_landmarks[0]

    # wrist sopra testa
    nose_y = landmarks[0].y
    left_wrist_y = landmarks[15].y
    right_wrist_y = landmarks[16].y

    if left_wrist_y < nose_y or right_wrist_y < nose_y:
        return True

    return False

# =========================
# VIDEO INPUT
# =========================
video_path = "/home/joel/Tesi/Dataset/Test/Weaponized/t_w003_converted.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video non aperto")
    exit()

# =========================
# VIDEO OUTPUT
# =========================
fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = cap.get(cv2.CAP_PROP_FPS) or 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "/home/joel/Tesi/Videos/Outputs/final_output.avi"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# =========================
# PARAMETRI
# =========================
WINDOW_SIZE = 20
MAX_HISTORY = 10

track_history = {}
pair_contact_counter = {}
event_buffer = deque(maxlen=WINDOW_SIZE)

# =========================
# UTILS
# =========================
def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def compute_speed(track):
    if len(track) < 2:
        return 0
    return math.dist(track[-1], track[-2])

def compute_acc(track):
    if len(track) < 3:
        return 0
    v1 = math.dist(track[-2], track[-3])
    v2 = math.dist(track[-1], track[-2])
    return abs(v2 - v1)

# =========================
# CLASSIFICATORE
# =========================
def classify_event(events):
    if not events:
        return "normal"

    if any(e["weapon"] for e in events):
        return "weaponized"

    if any(e["pose"] and e["contact"] > 3 for e in events):
        return "fight"

    if any(e["contact"] > 5 for e in events):
        return "aggression"

    return "suspicious"

# =========================
# LLaVA
# =========================
def run_llava(prompt):
    result = subprocess.run(
        f'ollama run llava "{prompt}"',
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

# =========================
# LOOP
# =========================
frame_count = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    pose_flag = detect_aggressive_pose(frame)

    results = model.track(frame, persist=True, device=0)
    r = results[0]
    boxes = r.boxes

    ids, coords = [], []
    frame_events = []

    # WEAPON DETECTION
    weapon_flag = False
    if boxes is not None and boxes.cls is not None:
        for cls in boxes.cls.int().cpu().tolist():
            if cls in WEAPON_CLASS_IDS:
                weapon_flag = True

    # TRACKING
    if boxes is not None and boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
        coords = boxes.xyxy.cpu().tolist()

        for id_, box in zip(ids, coords):
            c = center(box)
            track_history.setdefault(id_, deque(maxlen=MAX_HISTORY)).append(c)

    # EVENT ANALYSIS
    if len(coords) >= 2:

        for (id1, b1), (id2, b2) in itertools.combinations(zip(ids, coords), 2):

            c1, c2 = center(b1), center(b2)
            dist = math.dist(c1, c2)

            h = ((b1[3]-b1[1]) + (b2[3]-b2[1])) / 2
            if h == 0:
                continue

            norm_dist = dist / h

            t1 = track_history.get(id1, [])
            t2 = track_history.get(id2, [])

            speed1, speed2 = compute_speed(t1), compute_speed(t2)
            acc1, acc2 = compute_acc(t1), compute_acc(t2)

            pair = tuple(sorted((id1, id2)))

            if norm_dist < 1.2:
                pair_contact_counter[pair] = pair_contact_counter.get(pair, 0) + 1
            else:
                pair_contact_counter[pair] = 0

            contact = pair_contact_counter[pair]

            if norm_dist < 2.5 and (
                speed1 > 1.5 or speed2 > 1.5 or
                acc1 > 1.0 or acc2 > 1.0 or
                contact > 5 or
                weapon_flag or
                pose_flag
            ):
                frame_events.append({
                    "norm_dist": norm_dist,
                    "speed1": speed1,
                    "speed2": speed2,
                    "acc1": acc1,
                    "acc2": acc2,
                    "contact": contact,
                    "weapon": weapon_flag,
                    "pose": pose_flag
                })

    event_buffer.append(frame_events)

    annotated = r.plot()

    # =========================
    # WINDOW ANALYSIS
    # =========================
    if frame_count % WINDOW_SIZE == 0:

        flat = [e for sub in event_buffer for e in sub]
        event_type = classify_event(flat)

        print("\nEVENT:", event_type)

        if event_type != "normal":

            prompt = f"""
Surveillance scene.

Event: {event_type}
Weapon: {any(e['weapon'] for e in flat)}
Aggressive pose: {any(e['pose'] for e in flat)}

Explain briefly.
"""
            print("🧠 LLaVA:")
            print(run_llava(prompt))

    cv2.putText(annotated, f"Frame: {frame_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    out.write(annotated)

cap.release()
out.release()

print("✔ Done")