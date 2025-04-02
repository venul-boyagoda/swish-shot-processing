import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
from scipy import interpolate
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


def compute_angular_velocity(joints, fps):
    """
    Use Central Difference theorem to estimate rotational velocity (rad/s), store in pandas DataFrame

    """
    dt = 1 / fps
    angular_velocity_list = []

    # Compute central difference: (θ_(i+1) - θ_(i-1)) / (2 * dt)
    # Compute central difference for frames 1 to n-2
    for i in range(1, len(joints) - 1):
        frame_angles = {}
        for joint in joints[0].keys():
            prev_angle = joints[i - 1][joint]
            next_angle = joints[i + 1][joint]

            # Handle NaN or missing data gracefully
            if np.isnan(prev_angle) or np.isnan(next_angle):
                frame_angles[joint] = np.nan
            else:
                # Compute central difference
                frame_angles[joint] = np.radians((next_angle - prev_angle) / (2 * dt))

        angular_velocity_list.append(frame_angles)

    # Set first frame equal to the second frame’s velocity
    angular_velocity_list.insert(0, angular_velocity_list[0])

    # Set last frame equal to the second-to-last frame’s velocity
    angular_velocity_list.append(angular_velocity_list[-1])

    return angular_velocity_list


def upscale_to_100hz(angular_velocity_list, fps):
    target_fps = 100  ### DESIRED FREQ BASED OFF IMU ###
    original_time = np.arange(len(angular_velocity_list)) / fps
    target_time = np.arange(0, original_time[-1], 1 / target_fps)

    # Prepare list to store upscaled angular velocities
    upscaled_list = [{"Time": t} for t in target_time]

    # Get list of joints to iterate through
    all_joints = list(angular_velocity_list[0].keys())

    for joint in all_joints:
        # Extract joint data as a NumPy array
        joint_data = np.array([frame[joint] for frame in angular_velocity_list])

        # Cubic interpolation
        interp_func = interpolate.interp1d(
            original_time, joint_data, kind="cubic", fill_value="extrapolate"
        )
        upscaled_joint_data = interp_func(target_time)

        # Add interpolated joint data to upscaled list
        for i, t in enumerate(target_time):
            upscaled_list[i][joint] = upscaled_joint_data[i]

    return upscaled_list


def perp(r):
    """Returns the 2D perpendicular vector."""
    return np.array([-r[1], r[0]])


def rotation_matrix(theta):
    """2D rotation matrix."""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


def compute_linear_velocities(angular_velocity_upscaled, joint_angles_upscaled, limb_lengths):
    """
    Computes the linear velocities of the elbow, shoulder, and hip based on
    angular velocities and limb lengths.
    """

    # --- Segment lengths ---
    r_knee_to_hip_local = np.array([limb_lengths[('left_knee', 'left_hip')], 0.0])
    r_hip_to_shoulder_local = np.array([limb_lengths[('left_hip', 'left_shoulder')], 0.0])
    r_shoulder_to_elbow_local = np.array([-limb_lengths[('left_shoulder', 'left_elbow')], 0.0])

    # --- Prepare arrays ---
    omega_knee = np.array([frame["Knee"] for frame in angular_velocity_upscaled])
    omega_hip = np.array([frame["Hip"] for frame in angular_velocity_upscaled])
    omega_shoulder = np.array([frame["Shoulder"] for frame in angular_velocity_upscaled])

    theta_knee = np.array([frame["Knee"] for frame in joint_angles_upscaled])
    theta_hip = np.array([frame["Hip"] for frame in joint_angles_upscaled])
    theta_shoulder = np.array([frame["Shoulder"] for frame in joint_angles_upscaled])

    # --- Propagate velocities for all frames ---
    v_elbow_list, v_shoulder_list, v_hip_list = [], [], []

    for i in range(len(theta_knee)):
        # Rotation matrices for each frame
        R_knee = rotation_matrix(theta_knee[i])
        R_hip = rotation_matrix(theta_hip[i])
        R_shoulder = rotation_matrix(theta_shoulder[i])

        # Rotate segment vectors into global frame
        r_knee_to_hip_global = R_knee @ r_knee_to_hip_local
        r_hip_to_shoulder_global = R_hip @ r_hip_to_shoulder_local
        r_shoulder_to_elbow_global = R_shoulder @ r_shoulder_to_elbow_local

        # Base velocity (knee assumed grounded)
        v_knee = np.array([0.0, 0.0])
        v_hip = v_knee + omega_knee[i] * perp(r_knee_to_hip_global)
        v_shoulder = v_hip + omega_hip[i] * perp(r_hip_to_shoulder_global)
        v_elbow = v_shoulder + omega_shoulder[i] * perp(r_shoulder_to_elbow_global)

        # Append velocities to lists
        v_elbow_list.append(v_elbow)
        v_shoulder_list.append(v_shoulder)
        v_hip_list.append(v_hip)

    # --- Convert lists to NumPy arrays ---
    v_elbow_array = np.stack(v_elbow_list)  # shape (N, 2)
    v_shoulder_array = np.stack(v_shoulder_list)
    v_hip_array = np.stack(v_hip_list)

    return v_elbow_array, v_shoulder_array, v_hip_array


def calculate_power(limb_lengths, R_wrist_IMU_to_global, omega_wrist_global, v_elbow_array, accel_wrist_global):
    r_elbow_to_wrist_local = np.array([-limb_lengths[('left_elbow', 'left_wrist')], 0.0, 0.0])  # 3D vector

    # --- Forearm IMU propagation (already flipped to global wrist frame) ---

    # Rotation matrix: IMU to global, after X-flip correction
    R_wrist = R_wrist_IMU_to_global  # shape (3, 3)

    # Rotate r vector into global frame
    r_elbow_to_wrist_global = R_wrist @ r_elbow_to_wrist_local  # shape (3,)

    # Angular velocity in global frame (already flipped)
    omega_wrist = omega_wrist_global  # shape (3,)

    # Convert a single row of v_elbow_array into a 3D array
    v_elbow_global = np.array([v_elbow_array[0], v_elbow_array[1], 0.0])

    # Takes v_elbow_array from pose estimation propagation
    # Tangential velocity due to forearm rotation: v = ω x r
    v_wrist_from_forearm = v_elbow_global + np.cross(omega_wrist, r_elbow_to_wrist_global)  # shape (3,)

    # --- Power calculation ---
    # Linear acceleration in global frame (already flipped, gravity-compensated)
    a_wrist = accel_wrist_global  # shape (3,)

    # Power = dot(a, v)
    power = np.dot(a_wrist, v_wrist_from_forearm)
    
    return power


def calculate_elbow_flare_angle(R_upper_arm, R_forearm):
    """
    Calculate the elbow flare angle based on the rotation matrices of the upper arm and forearm IMUs.

    Parameters:
    R_upper_arm (numpy.ndarray): Shape (3, 3) rotation matrices from the upper arm IMU.
    R_forearm (numpy.ndarray): Shape (3, 3) rotation matrices from the forearm IMU.

    Returns:
    float: Shape elbow flare angles in degrees.


    1. Calculates the relative orientation of the forearm in the upper arm's frame
    2. Projects the forearm's Z-axis onto the Y-Z plane of the upper arm (removing flexion component)
    3. Measures the angle between this projected Z-axis and where it would be in a neutral position
    4. Determines the sign of the angle to distinguish between lateral (outward) and medial (inward) flare
    """

    # Calculate the relative rotation for each sample in the batch
    R_forearm_to_upper_arm = np.dot(R_upper_arm.T, R_forearm)

    # Extract the Z-axis of the forearm in the upper arm's reference frame
    z_forearm_in_upper_arm = R_forearm_to_upper_arm[:, 2]

    # Project onto Y-Z plane (removing X component)
    z_forearm_yz_plane = np.array([0, z_forearm_in_upper_arm[1], z_forearm_in_upper_arm[2]])

    # Normalize
    norm = np.linalg.norm(z_forearm_yz_plane)
    if norm < 1e-10:  # Avoid division by zero
        return None
    z_forearm_yz_plane = z_forearm_yz_plane / norm

    # Calculate angle with neutral position
    neutral_z = np.array([0, 0, 1])
    dot_product = np.dot(z_forearm_yz_plane, neutral_z)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    flare_angle_rad = np.arccos(dot_product)

    # Determine sign (positive for lateral, negative for medial)
    if z_forearm_yz_plane[1] < 0:
        flare_angle_rad = -flare_angle_rad

    # Convert to degrees
    flare_angle = np.degrees(flare_angle_rad)

    return flare_angle


def apply_axis_transformations(imu_rotation_matrices_forearm, imu_rotation_matrices_upperarm, imu_accel_forearm, imu_gyro_forearm):
    """
    Uses a reflection matrix to swap the direction of the x-axis to match other data sets,
    where positive is pointing upwards to shoulder and negative is pointing downwards'
    """

    flip_x_matrix = np.array([
        [-1, 0, 0],
        [0,  1, 0],
        [0,  0, 1]
    ])

    # Apply axis transformation to Rotation Matrix
    R_wrist_IMU_to_global = flip_x_matrix @ imu_rotation_matrices_forearm @ flip_x_matrix
    R_elbow_IMU_to_global = flip_x_matrix @ imu_rotation_matrices_upperarm @ flip_x_matrix

    # Apply axis transformation to linear accel values
    accel_wrist_global = np.dot(flip_x_matrix, imu_accel_forearm.T).T

    # Apply axis transformation to gyro values
    omega_wrist_global = np.dot(flip_x_matrix, imu_gyro_forearm.T).T

    return R_wrist_IMU_to_global, R_elbow_IMU_to_global, accel_wrist_global, omega_wrist_global



def process_imu_data(shots, imu_objects, frame_timestamps, height, pose_landmarks):
    # Calculate estimate lengths based on height (Reference: https://www.openlab.psu.edu/tools/)
    limb_lengths = {
        ("left_ankle", "left_knee"): 0.255*height,
        ("left_knee", "left_hip"): 0.232*height,
        ("left_hip", "left_shoulder"): 0.3*height,
        ("left_shoulder", "left_elbow"): 0.186*height,
        ("left_elbow", "left_wrist"): 0.146*height,
    }

    # Define joint sets for angle calculation
    joint_sets = {
        "Right Elbow": [12, 14, 16],
        "Right Shoulder": [14, 12, 24],
        "Right Wrist": [20, 16, 14],
        "Right Hip": [12, 24, 26],
        "Right Knee": [24, 26, 28]
    }

    joint_angles = []

    for landmark in pose_landmarks:
        frame_angles = {}
        if landmark is not None:
            # Extract angles for each joint set
            for joint_name, (j1, j2, j3) in joint_sets.items():
                landmarks = landmark.landmark

                p1 = (landmarks[j1].x, landmarks[j1].y)
                p2 = (landmarks[j2].x, landmarks[j2].y)
                p3 = (landmarks[j3].x, landmarks[j3].y)

                angle = calculate_angle(p1, p2, p3)
                frame_angles[joint_name] = angle

            joint_angles.append(frame_angles)
        else:
            joint_angles.append({joint: np.nan for joint in joint_sets.keys()})
    
    angular_velocity_list = compute_angular_velocity(joint_angles, FPS)
    angular_velocity_upscaled = upscale_to_100hz(angular_velocity_list, FPS)
    joint_angles_upscaled = upscale_to_100hz(joint_angles, FPS)

    # Compute Linear Velocities
    v_elbow_array, v_shoulder_array, v_hip_array = compute_linear_velocities(
        angular_velocity_upscaled, joint_angles_upscaled, limb_lengths
    )

    for shot in shots:
        # Match Follow Through Frame
        follow_ts = frame_timestamps[shot['follow_frame']]
        closest_follow = min(imu_objects, key=lambda imu: abs(imu.timestamp - follow_ts))
        idx_follow = imu_objects.index(closest_follow)
        start_idx_follow = max(0, idx_follow - 15)
        power_window = imu_objects[start_idx_follow:idx_follow + 1]

        # Compute max power over window of 15 entries
        powers = []
        for i, imu in enumerate(power_window):
            R_wrist_IMU_to_global, R_elbow_IMU_to_global, accel_wrist_global, omega_wrist_global = apply_axis_transformations(imu.bno_matrix, imu.bmi_matrix, imu.bno_accel, imu.bno_gyro)
            upscaled_idx = (start_idx_follow+i)*100/FPS
            power = calculate_power(limb_lengths, R_wrist_IMU_to_global, omega_wrist_global, v_elbow_array[upscaled_idx], accel_wrist_global)
            powers.append(power)

        shot['power'] = max(powers) if powers else None

        # Match set frame
        set_ts = frame_timestamps[shot['set_frame']]
        closest_set = min(imu_objects, key=lambda imu: abs(imu.timestamp - set_ts))
        idx_set = imu_objects.index(closest_set)
        start_idx_set = max(0, idx_set - 3)
        set_window = imu_objects[start_idx_set:idx_set + 3]

        # Compute elbow flare angle average over 6 entries
        flare_angles = []
        for i, imu in enumerate(set_window):
            R_wrist_IMU_to_global, R_elbow_IMU_to_global, accel_wrist_global, omega_wrist_global = apply_axis_transformations(imu.bno_matrix, imu.bmi_matrix, imu.bno_accel, imu.bno_gyro)
            flare_angle = calculate_elbow_flare_angle(R_elbow_IMU_to_global, R_wrist_IMU_to_global)
            flare_angles.append(flare_angle)

        shot['elbow_flare'] = sum(flare_angles) / len(flare_angles) if flare_angles else None


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


def process_video(video_path, imu_objects, video_start_time, handedness, height):
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

    process_imu_data(shots, imu_objects, frame_times, height, pose_landmarks_all)

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
    height_float = float(height)/100.0

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

    results = process_video(file_path, imu_objects, video_start_time_float, handedness, height_float)
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
    test_video_path = "test_videos/KapiShooting60.mp4"
    processed = process_video(test_video_path, "right")
    generate_overlay_video(test_video_path, processed)
    print("Results: ", processed["shots"])
    print("Consistency Score: ", processed["consistency_score"])
