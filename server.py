import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import math
import statistics
import json
import shutil
import os
import csv

app = FastAPI()

model = YOLO("models/ball_net_detection.pt")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

BALL_CLASS = 1
NET_CLASS = 3

FPS = 60
MIN_UPWARD_TRAVEL = int((30.0 / FPS) * 60)
TEXT_PERSIST_FRAMES = int(7 * FPS)


def calculate_angle(a, b, c):
    try:
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle if not np.isnan(angle) else 0
    except:
        return 0


def extract_detections(frame):
    detectionResults = model(frame, device="mps")
    detectionResult = detectionResults[0]
    bboxes = np.array(detectionResult.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(detectionResult.boxes.cls.cpu(), dtype="int")
    confs = np.array(detectionResult.boxes.conf.cpu(), dtype="float")
    return bboxes, classes, confs


def extract_pose(frame):
    poseResults = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return poseResults.pose_landmarks


def extract_ball_and_net(bboxes, classes, confs):
    ball_center = None
    net_bbox = None

    ball_indices = np.where(classes == BALL_CLASS)[0]
    net_indices = np.where(classes == NET_CLASS)[0]

    if len(ball_indices) > 0:
        best_ball_idx = ball_indices[np.argmax(confs[ball_indices])]
        x, y, x2, y2 = bboxes[best_ball_idx]
        ball_center = (int((x + x2) // 2), int((y + y2) // 2))

    if len(net_indices) > 0:
        best_net_idx = net_indices[np.argmax(confs[net_indices])]
        net_bbox = bboxes[best_net_idx]

    return ball_center, net_bbox


def calculate_shots(ball_centers, wrist_to_ball_dists, net_bboxes, pose_landmarks_all, width, height, handedness):
    shots = []
    i = 1
    while i < len(wrist_to_ball_dists):
        if wrist_to_ball_dists[i-1] and wrist_to_ball_dists[i]:
            if wrist_to_ball_dists[i-1] < 500 and wrist_to_ball_dists[i] > 500:
                if ball_centers[i] and ball_centers[i-1]:
                    upward_movement = ball_centers[i-1][1] - ball_centers[i][1]
                    if upward_movement > MIN_UPWARD_TRAVEL:
                        landmarks = pose_landmarks_all[i]
                        elbow_angle_follow = compute_follow_angle(landmarks, width, height, handedness)

                        shot = {
                            'follow_frame': i,
                            'set_frame': None,
                            'end': None,
                            'elbow_follow': elbow_angle_follow,
                            'elbow_set': None,
                            'knee_set': None,
                            'success': False,
                            'release_angle': None
                        }

                        detect_set_stage(shot, i, pose_landmarks_all, width, height, handedness)
                        detect_shot_end(shot, i, ball_centers, net_bboxes)

                        if shot['end']:
                            shot["success"] = detect_shot_success_linear(ball_centers, net_bboxes, shot['follow_frame'], shot['end'])
                            shot["release_angle"] = calculate_release_angle(ball_centers, shot['follow_frame'])

                        shots.append(shot)
                        i = shot['end']
        i += 1
    return shots


def detect_shot_success_linear(ball_centers, net_bboxes, follow_idx, end_idx, debug_draw=None):
    above_point = None
    below_point = None

    for i in range(follow_idx, end_idx):
        if ball_centers[i] is not None and net_bboxes[i] is not None:
            _, y = ball_centers[i]
            net_x1, net_y1, net_x2, net_y2 = net_bboxes[i]

            # last point before entering top of net
            if y < net_y1:
                above_point = ball_centers[i]

            # first point after exiting bottom of net
            elif y > net_y2 and below_point is None:
                below_point = ball_centers[i]
                break

    if above_point and below_point:
        x1, y1 = above_point
        x2, y2 = below_point

        if x2 - x1 == 0:
            return False  # vertical line, undefined slope

        # line equation: y = m * x + b  → invert to get x = (y - b) / m
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # find the midpoint y between above and below
        mid_y = (y1 + y2) / 2
        mid_x = (mid_y - b) / m

        # Get padded net bounds
        rim_x1 = min(net_x1, net_x2)
        rim_x2 = max(net_x1, net_x2)
        padding = (rim_x2 - rim_x1) * 0.1  # 10% padding inside each side

        padded_x1 = rim_x1 + padding
        padded_x2 = rim_x2 - padding

        if debug_draw is not None:
            # draw main line
            cv2.line(debug_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # draw padded rim zone
            cv2.rectangle(debug_draw, (int(padded_x1), net_y1), (int(padded_x2), net_y2), (0, 100, 255), 2)

            # draw midpoint
            cv2.circle(debug_draw, (int(mid_x), int(mid_y)), 6, (255, 255, 0), -1)

        # success if midpoint X is inside padded net
        if padded_x1 < mid_x < padded_x2:
            return True

    return False


def calculate_release_angle(ball_centers, follow_idx, num_points = int(FPS * 0.1666)):
    """Calculate release angle based on first N ball positions."""
    points = []
    for i in range(follow_idx, follow_idx + num_points):
        if i < len(ball_centers) and ball_centers[i]:
            points.append(ball_centers[i])

    if len(points) >= 2:
        dx = points[-1][0] - points[0][0]
        dy = points[0][1] - points[-1][1]  # y axis is inverted in image coordinates
        angle = np.degrees(np.arctan2(dy, dx))
        return round(angle, 2)
    return None


def compute_follow_angle(landmarks, width, height, handedness):
    if landmarks:
        if handedness == "right":
            sh = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            el = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wr = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        else:
            sh = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            el = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            wr = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        return calculate_angle(
            (int(sh.x * width), int(sh.y * height)),
            (int(el.x * width), int(el.y * height)),
            (int(wr.x * width), int(wr.y * height))
        )
    return 0


def detect_set_stage(shot, follow_idx, pose_landmarks_all, width, height, handedness):
    for j in range(follow_idx - 1, -1, -1):
        landmarks = pose_landmarks_all[j]
        if landmarks:
            if handedness == "right":
                sh = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                el = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                wr = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                hi = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                kn = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                an = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            else:
                sh = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                el = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                wr = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                hi = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                kn = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                an = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

            shoulder_pt = (int(sh.x * width), int(sh.y * height))
            elbow_pt = (int(el.x * width), int(el.y * height))
            wrist_pt = (int(wr.x * width), int(wr.y * height))
            hip_pt = (int(hi.x * width), int(hi.y * height))
            angle = calculate_angle(hip_pt, shoulder_pt, elbow_pt)
            if 80 <= angle <= 110:
                shot['set_frame'] = j
                knee_angle = calculate_angle(hip_pt,
                                        (int(kn.x * width), int(kn.y * height)),
                                        (int(an.x * width), int(an.y * height)))
                elbow_angle = calculate_angle(shoulder_pt, elbow_pt, wrist_pt)
                shot['elbow_set'] = elbow_angle
                shot['knee_set'] = knee_angle
                break


def detect_shot_end(shot, follow_idx, ball_centers, net_bboxes, delay = int(FPS * 0.3333) ):
    """
    Delay shot end by additional frames beyond the initial net cross detection.
    """
    found = False
    for k in range(follow_idx + 1, len(ball_centers)):
        if ball_centers[k] and net_bboxes[k] is not None:
            net_y = net_bboxes[k][3]  # bottom of the net bbox
            if ball_centers[k][1] > net_y:
                shot['end'] = min(k + delay, len(ball_centers) - 1)
                found = True
                break
    if not found:
        shot['end'] = min(len(ball_centers) - 1, follow_idx + delay)  # fallback


def process_imu_data(shots, imu_objects, frame_timestamps):
    for shot in shots:
        # Match Follow Through Frame
        follow_ts = frame_timestamps[shot['follow_frame']]
        closest_follow = min(imu_objects, key=lambda imu: abs(imu.timestamp - follow_ts))
        idx_follow = imu_objects.index(closest_follow)
        follow_window = imu_objects[max(0, idx_follow - 4):idx_follow + 1]

        # Compute BNO accel magnitude average over window of 5 entries
        magnitudes = []
        for imu in follow_window:
            x, y, z = imu.bno_accel
            mag = math.sqrt(x ** 2 + y ** 2 + z ** 2)
            magnitudes.append(mag)
        shot['follow_accel'] = sum(magnitudes) / len(magnitudes) if magnitudes else None

        # Match set frame
        set_ts = frame_timestamps[shot['set_frame']]
        closest_set = min(imu_objects, key=lambda imu: abs(imu.timestamp - set_ts))
        idx_set = imu_objects.index(closest_set)
        set_window = imu_objects[max(0, idx_set - 4):idx_set + 1]

        # Compute shoulder deviation angle average over 5 entries
        angles = []
        for imu in set_window:
            bmi_matrix = np.array(imu.bmi_matrix).reshape(3, 3)
            x_axis = bmi_matrix[:, 0]
            angle_rad = math.atan2(x_axis[2], x_axis[0])
            angles.append(math.degrees(angle_rad))

        shot['shoulder_set'] = sum(angles) / len(angles) if angles else None

def calculate_consistency(shots, use_imu_values = True):
    metrics = []

    if use_imu_values:
        metrics = ["elbow_follow", "elbow_set", "knee_set", "release_angle", "follow_accel", "shoulder_set"]
    else:
        metrics = ["elbow_follow", "elbow_set", "knee_set", "release_angle"]

    cv_values = []

    for metric in metrics:
        values = [shot[metric] for shot in shots if shot[metric] is not None]

        if len(values) >= 2:
            # Standard Deviation and Mean
            sd = round(statistics.stdev(values), 2)
            mean_val = statistics.mean(values)

            # Coefficient of Variation as percentage
            cv = round((sd / mean_val) * 100, 2) if mean_val != 0 else 0

            # Add CV to list for overall consistency
            cv_values.append(cv)

    # Calculate average of cv's for all metrics
    overall_cv = round(sum(cv_values) / len(cv_values), 2) if cv_values else None

    # Calculate Consistency Score (100 - cv)
    consistency_score = round(100 - overall_cv, 2) if overall_cv is not None else None

    return consistency_score


def process_video(video_path, imu_objects, video_start_time, handedness):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ball_centers, wrist_to_ball_dists, net_bboxes, pose_landmarks_all = [], [], [], []
    frame_times = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute frame's absolute unix timestamp
        frame_unix_time = video_start_time + (frame_idx / fps)
        frame_times.append(frame_unix_time)
        frame_idx += 1

        bboxes, classes, confs = extract_detections(frame)
        landmarks = extract_pose(frame)
        pose_landmarks_all.append(landmarks)

        ball_center, net_bbox = extract_ball_and_net(bboxes, classes, confs)

        if ball_center:
            wrist_landmark = None
            if landmarks:
                wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                wrist_landmark = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
            if wrist_landmark:
                dist = np.linalg.norm(np.array(ball_center) - np.array(wrist_landmark))
                wrist_to_ball_dists.append(dist)
            else:
                wrist_to_ball_dists.append(None)
        else:
            wrist_to_ball_dists.append(None)

        ball_centers.append(ball_center)
        net_bboxes.append(net_bbox)

    cap.release()

    shots = calculate_shots(ball_centers, wrist_to_ball_dists, net_bboxes, pose_landmarks_all, width, height, handedness)

    process_imu_data(shots, imu_objects, frame_times)

    return {
        "shots": shots,
        "ball_centers": ball_centers,
        "net_bboxes": net_bboxes,
        "pose_landmarks_all": pose_landmarks_all
    }


def process_video(video_path, handedness):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ball_centers, wrist_to_ball_dists, net_bboxes, pose_landmarks_all = [], [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, classes, confs = extract_detections(frame)
        landmarks = extract_pose(frame)
        pose_landmarks_all.append(landmarks)

        ball_center, net_bbox = extract_ball_and_net(bboxes, classes, confs)

        if ball_center:
            wrist_landmark = None
            if landmarks:
                wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                wrist_landmark = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
            if wrist_landmark:
                dist = np.linalg.norm(np.array(ball_center) - np.array(wrist_landmark))
                wrist_to_ball_dists.append(dist)
            else:
                wrist_to_ball_dists.append(None)
        else:
            wrist_to_ball_dists.append(None)

        ball_centers.append(ball_center)
        net_bboxes.append(net_bbox)

    cap.release()

    shots = calculate_shots(ball_centers, wrist_to_ball_dists, net_bboxes, pose_landmarks_all, width, height, handedness)

    consistency_score = calculate_consistency(shots, False)

    return {
        "shots": shots,
        "consistency_score": consistency_score,
        "ball_centers": ball_centers,
        "net_bboxes": net_bboxes,
        "pose_landmarks_all": pose_landmarks_all
    }


def generate_overlay_video(video_path, processed_data, output_path="output_overlay.mp4"):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ball_centers = processed_data["ball_centers"]
    net_bboxes = processed_data["net_bboxes"]
    pose_landmarks_all = processed_data["pose_landmarks_all"]
    shots = processed_data["shots"]

    frame_idx, shot_idx, trail = 0, 0, []
    overlay_timers = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(pose_landmarks_all):
            pose_landmarks = pose_landmarks_all[frame_idx]
            if pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

        if frame_idx < len(ball_centers) and ball_centers[frame_idx]:
            cv2.circle(frame, ball_centers[frame_idx], 8, (255, 0, 0), -1)
        if frame_idx < len(net_bboxes) and net_bboxes[frame_idx] is not None:
            net_bbox = net_bboxes[frame_idx]
            cv2.rectangle(frame, (net_bbox[0], net_bbox[1]), (net_bbox[2], net_bbox[3]), (255, 0, 255), 3)

        if shot_idx < len(shots):
            shot = shots[shot_idx]
            if frame_idx == shot['follow_frame']:
                overlay_timers["start"] = frame_idx

            if shot['follow_frame'] <= frame_idx <= shot['end']:
                if ball_centers[frame_idx]:
                    trail.append(ball_centers[frame_idx])
                for pt in trail:
                    cv2.circle(frame, pt, 8, (0, 0, 255), -1)

            if frame_idx > shot['end']:
                overlay_timers = {}
                trail = []
                shot_idx += 1

        if "start" in overlay_timers and frame_idx - overlay_timers["start"] <= TEXT_PERSIST_FRAMES:
            shot = shots[shot_idx]

            detect_shot_success_linear(
                ball_centers,
                net_bboxes,
                shot['follow_frame'],
                shot['end'],
                debug_draw=frame
            )

            # Bigger text settings
            font_scale_title = 2.5
            font_scale_main = 2.0
            font_scale_small = 1.8
            thickness = 5

            y = 100
            line_spacing = 70
            extra_space = 30  # space between sections

            # Shot number and "Set Stage"
            cv2.putText(frame, f"Shot {shot_idx + 1}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 255, 255), thickness)
            y += line_spacing + 20

            cv2.putText(frame, "Set Stage", (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (255, 255, 255), thickness)
            y += line_spacing

            elbow_set_text = f"Elbow: {int(shot['elbow_set'])} deg" if shot['elbow_set'] is not None else "Elbow: None"
            cv2.putText(frame, elbow_set_text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness - 2)
            y += line_spacing

            # Knee Set
            knee_set_text = f"Knee: {int(shot['knee_set'])} deg" if shot['knee_set'] is not None else "Knee: None"
            cv2.putText(frame, knee_set_text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness - 2)

            # Follow Through Elbow
            y += line_spacing + extra_space
            cv2.putText(frame, "Follow Through", (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (255, 255, 255), thickness)
            y += line_spacing

            elbow_follow_text = f"Elbow: {int(shot['elbow_follow'])} deg" if shot['elbow_follow'] is not None else "Elbow: None"
            cv2.putText(frame, elbow_follow_text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness - 2)

            # Release Angle
            y += line_spacing + extra_space
            release_str = f"{shot['release_angle']:.1f} deg" if shot['release_angle'] is not None else "None"
            cv2.putText(frame, f"Release Angle: {release_str}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), thickness - 2)

            # EXTRA SPACE between "Release Angle" and "Shot Success"
            y += line_spacing + extra_space
            success_str = "YES" if shot['success'] else "NO"
            cv2.putText(frame, f"Shot Success: {success_str}", (50, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_main, (0, 255, 0) if shot['success'] else (0, 0, 255), thickness)


        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


class IMUData(BaseModel):
    timestamp: float           # unix timestamp
    bno_matrix: List[float]  # 9 floats
    bmi_matrix: List[float]  # 9 floats
    bno_gyro: List[float]      # 3 floats
    bno_accel: List[float]     # 3 floats
    bmi_gyro: List[float]      # 3 floats


@app.post("/upload/")
async def upload_video(
    file: UploadFile = File(...),
    handedness: str = Form(...),  # 'left' or 'right'
    imu_data: str = Form(...), # JSON stringified IMU data list
    video_start_time: str = Form(...),
    height: str = Form(...)     
):
    file_path = f"temp_videos/{file.filename}"
    os.makedirs("temp_videos", exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_start_time_float = float(video_start_time)

    # # Parse IMU data into JSON
    # imu_data_list = json.loads(imu_data)
    # imu_objects = [IMUData(**entry) for entry in imu_data_list]

    imu_data_list = json.loads(imu_data)

    for entry in imu_data_list:
        entry["timestamp"] = float(entry["timestamp"])
        entry["bno_matrix"] = [float(v) for v in entry["bno_matrix"]]
        entry["bmi_matrix"] = [float(v) for v in entry["bmi_matrix"]]
        entry["bno_gyro"] = [float(v) for v in entry["bno_gyro"]]
        entry["bno_accel"] = [float(v) for v in entry["bno_accel"]]
        entry["bmi_gyro"] = [float(v) for v in entry["bmi_gyro"]]

    imu_objects = [IMUData(**entry) for entry in imu_data_list]

    # --- Save IMU data to CSV ---
    # imu_csv_path = f"imu_logs/{file.filename}_imu_data.csv"
    # os.makedirs("imu_logs", exist_ok=True)
    # with open(imu_csv_path, "w", newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # CSV Header
    #     writer.writerow([
    #         "timestamp",
    #         *["bno_matrix_" + str(i) for i in range(9)],
    #         *["bmi_matrix_" + str(i) for i in range(9)],
    #         *["bno_gyro_" + str(i) for i in range(3)],
    #         *["bno_accel_" + str(i) for i in range(3)],
    #         *["bmi_gyro_" + str(i) for i in range(3)]
    #     ])
    #     # CSV Rows
    #     for imu in imu_objects:
    #         row = [
    #             imu.timestamp,
    #             *imu.bno_matrix,
    #             *imu.bmi_matrix,
    #             *imu.bno_gyro,
    #             *imu.bno_accel,
    #             *imu.bmi_gyro
    #         ]
    #         writer.writerow(row)
    # print(f"IMU data saved to {imu_csv_path}")
    # # -------------------------------------

    results = process_video(file_path, imu_objects, video_start_time_float, handedness)
    shots = results["shots"]
    consistency_score = calculate_consistency(shots)

    print(f"Hand: {handedness}")
    print(f"Video Start Time: {video_start_time_float}")
    print(f"First IMU Entry: {imu_data_list[0]}")

    # if len(shots) > 0:
    #     shot = shots[0]
    #     filtered_shot = {k: v for k, v in shot.items() if k not in ['follow_frame', 'set_frame', 'end']}
    #     print(filtered_shot)
    #     return filtered_shot
    # else:
    #     print("No shots detected")
    #     return {}
    
    if len(shots) > 0:
        filtered_shots = [
            {k: v for k, v in shot.items() if k not in ['set_frame']}
            for shot in shots
        ]
        print(filtered_shots)
        return {
            "shots": filtered_shots,
            "consistency_score": consistency_score
        }
    else:
        print("No shots detected")
        return {}


if __name__ == "__main__":
    test_video_path = "KapiShooting60.mp4"
    processed = process_video(test_video_path, "right")
    generate_overlay_video(test_video_path, processed)
    print("Results: ", processed["shots"])
    print("Consistency Score: ", processed["consistency_score"])
