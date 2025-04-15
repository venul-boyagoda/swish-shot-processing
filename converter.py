import csv
import json

def convert_imu_csv_to_txt(csv_path, txt_path):
    imu_data = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            imu_entry = {
                "timestamp": float(row["timestamp"]),
                "bno_matrix": [float(row[f"bno_matrix_{i}"]) for i in range(9)],
                "bmi_matrix": [float(row[f"bmi_matrix_{i}"]) for i in range(9)],
                "bno_gyro": [float(row[f"bno_gyro_{i}"]) for i in range(3)],
                "bno_accel": [float(row[f"bno_accel_{i}"]) for i in range(3)],
                "bmi_gyro": [float(row[f"bmi_gyro_{i}"]) for i in range(3)]
            }
            imu_data.append(imu_entry)

    # Write the JSON string to a .txt file
    with open(txt_path, 'w') as txt_file:
        json.dump(imu_data, txt_file)
    
    print(f"✅ IMU JSON string saved to {txt_path}")

# Example usage:
convert_imu_csv_to_txt("imu_logs/KapiShootingMain.mp4_imu_data.csv", "imu_data.txt")