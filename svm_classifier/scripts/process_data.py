import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

stride = 100 # 1s
window_size = 200 # 2s 

# Load data and preprocess
def extract_features(window):
    # Extract features from each sensor data within the window
    linear_features = extract_features_from_sensor_data(window.iloc[:, 2:5])
    gravity_features = extract_features_from_sensor_data(window.iloc[:, 7:10])
    gyro_features = extract_features_from_sensor_data(window.iloc[:, 12:15])
    # Combine features from all sensors
    return np.concatenate((linear_features, gravity_features, gyro_features))

def extract_features_from_sensor_data(sensor_data):
    # Calculate mean and variance for the sensor data
    mean = np.mean(sensor_data, axis=0)
    var = np.var(sensor_data, axis=0)

    feature_vector = np.concatenate([mean, var])

    return feature_vector

def process_data(folder_path):
    linear_data = pd.read_csv(folder_path + "linear.csv")
    gravity_data = pd.read_csv(folder_path + "gravity.csv")
    gyro_data = pd.read_csv(folder_path + "gyro.csv") 

    data = pd.concat([linear_data, gravity_data, gyro_data], axis=1)
    
    # Extract features with stride and window size
    features = []
    labels = []
    for i in range(0, len(data) - window_size, stride):
        window = data.iloc[i:i+window_size, :]
        features.append(extract_features(window))
        labels.append(data.iloc[i, 0]) 

        # Feature scaling (-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(features)
        features_scaled = scaler.transform(features)

    return features_scaled, labels

# Main processing loop
data_dir = "data/raw"
all_features = []
all_labels = []

for folder in range(7):
    folder_path = f"{data_dir}/{folder}/"
    features, labels = process_data(folder_path)
    all_features.extend(features)
    all_labels.extend(labels)

# Convert to LibSVM format
with open("data/data.txt", "w") as f:
    for label, features in zip(all_labels, all_features):
        feature_str = " ".join([f"{i+1}:{val}" for i, val in enumerate(features)])
        f.write(f"{label} {feature_str}\n")