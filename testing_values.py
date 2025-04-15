from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from pydantic import BaseModel
import os
import shutil
import csv
import json
import numpy as np
import cv2
import mediapipe as mp

app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

class IMUData(BaseModel):
    timestamp: float
    bno_matrix: List[float]  # 9 floats
    bmi_matrix: List[float]  # 9 floats
    bno_gyro: List[float]    # 3 floats
    bno_accel: List[float]   # 3 floats
    bmi_gyro: List[float]    # 3 floats

def apply_axis_transformations(imu_rotation_matrices_forearm, imu_rotation_matrices_upperarm):
    flip_x_matrix = np.array([
        [-1, 0, 0],
        [0,  1, 0],
        [0,  0, 1]
    ])
    R_wrist_raw = np.array(imu_rotation_matrices_forearm).reshape(3, 3)
    R_elbow_raw = np.array(imu_rotation_matrices_upperarm).reshape(3, 3)
    R_wrist_IMU_to_global = flip_x_matrix @ R_wrist_raw @ flip_x_matrix
    R_elbow_IMU_to_global = flip_x_matrix @ R_elbow_raw @ flip_x_matrix
    return R_wrist_IMU_to_global, R_elbow_IMU_to_global

def calculate_elbow_flare_angle(R_upper_arm, R_forearm):
    R_forearm_to_upper_arm = np.dot(R_upper_arm.T, R_forearm)
    z_forearm_in_upper_arm = R_forearm_to_upper_arm[:, 2]
    z_forearm_yz_plane = np.array([0, z_forearm_in_upper_arm[1], z_forearm_in_upper_arm[2]])
    norm = np.linalg.norm(z_forearm_yz_plane)
    if norm < 1e-10:
        return None
    z_forearm_yz_plane = z_forearm_yz_plane / norm
    neutral_z = np.array([0, 0, 1])
    dot_product = np.dot(z_forearm_yz_plane, neutral_z)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    flare_angle_rad = np.arccos(dot_product)
    if z_forearm_yz_plane[1] < 0:
        flare_angle_rad = -flare_angle_rad
    flare_angle = np.degrees(flare_angle_rad)
    return flare_angle

def calculate_elbow_bend_angle(landmarks, width, height, handedness):
    if handedness == "right":
        shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    else:
        shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

    a = np.array([shoulder.x * width, shoulder.y * height])
    b = np.array([elbow.x * width, elbow.y * height])
    c = np.array([wrist.x * width, wrist.y * height])

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

@app.post("/upload/")
async def extract_elbow_angles(
    file: UploadFile = File(...),
    handedness: str = Form(...),
    imu_data: str = Form(...),
    video_start_time: str = Form(...),
    height: str = Form(...)
):
    os.makedirs("temp_videos", exist_ok=True)
    os.makedirs("imu_logs", exist_ok=True)

    file_path = f"temp_videos/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    imu_data_list = json.loads(imu_data)
    imu_objects = [IMUData(**entry) for entry in imu_data_list]

    # Save original IMU data (JSON)
    raw_imu_json_path = f"imu_logs/{file.filename}_raw_imu.json"
    with open(raw_imu_json_path, "w") as f:
        json.dump(imu_data_list, f, indent=2)

    # Save original IMU data (CSV)
    raw_imu_csv_path = f"imu_logs/{file.filename}_raw_imu.csv"
    with open(raw_imu_csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp",
            *[f"bno_matrix_{i}" for i in range(9)],
            *[f"bmi_matrix_{i}" for i in range(9)],
            *[f"bno_gyro_{i}" for i in range(3)],
            *[f"bno_accel_{i}" for i in range(3)],
            *[f"bmi_gyro_{i}" for i in range(3)]
        ])
        for imu in imu_objects:
            writer.writerow([
                imu.timestamp,
                *imu.bno_matrix,
                *imu.bmi_matrix,
                *imu.bno_gyro,
                *imu.bno_accel,
                *imu.bmi_gyro
            ])

    # Elbow Bend per Frame
    output_csv_path = f"imu_logs/{file.filename}_elbow_bend.csv"
    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with open(output_csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_idx", "elbow_bend_angle_deg"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                bend = calculate_elbow_bend_angle(results.pose_landmarks, width, height_px, handedness)
                writer.writerow([frame_idx, bend])
            else:
                writer.writerow([frame_idx, None])
            frame_idx += 1

    cap.release()

    # Elbow Flare per IMU entry
    flare_csv_path = f"imu_logs/{file.filename}_elbow_flare.csv"
    with open(flare_csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "elbow_flare_angle_deg"])

        for imu in imu_objects:
            R_wrist, R_elbow = apply_axis_transformations(imu.bno_matrix, imu.bmi_matrix)
            flare = calculate_elbow_flare_angle(R_elbow, R_wrist)
            writer.writerow([imu.timestamp, flare])

    return {
        "message": "Elbow bend and flare angles extracted and saved.",
        "video_path": file_path,
        "imu_json_path": raw_imu_json_path,
        "imu_csv_path": raw_imu_csv_path,
        "elbow_bend_csv": output_csv_path,
        "elbow_flare_csv": flare_csv_path
    }
