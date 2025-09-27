# Multi-Sensor Data Classification System

A machine learning project that classifies security threat levels using data from multiple sensor modalities including audio, thermal imaging, electromagnetic, GPS, proximity, and IMU sensors. The system uses Random Forest classification with Leave-One-Out cross-validation to predict threat categories: "safe", "threat", and "failsafe triggered".

## Overview

This project processes and analyzes multi-modal sensor data to create a comprehensive threat detection system. It combines various sensor inputs to provide accurate classification of security scenarios in real-time environments.

## Features

- **Multi-Modal Sensor Integration**: Combines 6 different sensor types
- **Advanced Data Preprocessing**: Handles missing values, categorical encoding, and feature engineering
- **Machine Learning Classification**: Uses Random Forest with robust cross-validation
- **Comprehensive Evaluation**: Leave-One-Out cross-validation for small dataset scenarios
- **Automated Feature Engineering**: Dynamic thermal matrix flattening and categorical encoding

## Sensor Data Types

### 1. **Audio Sensor (Microphone)**
- **Features**: Peak decibel levels, audio patterns, temporal data
- **Use Case**: Detecting unusual sounds, conversations, alarms
- **Preprocessing**: Removes description and temporal metadata

### 2. **Thermal Sensor**
- **Features**: 2D thermal matrix data, night vision capability
- **Use Case**: Heat signature detection, person/object identification
- **Preprocessing**: Flattens thermal matrix into individual temperature readings

### 3. **Electromagnetic (EM) Sensor**
- **Features**: EM field measurements, signal interference detection
- **Use Case**: Electronic device detection, communication monitoring
- **Preprocessing**: Standard numerical feature handling

### 4. **GPS/Environment Sensor**
- **Features**: Location coordinates, time of day, movement status
- **Use Case**: Geofencing, movement pattern analysis
- **Preprocessing**: Categorical encoding (day/night, movement status)

### 5. **Proximity Sensor**
- **Features**: Distance measurements from multiple directions (front, left, right, rear)
- **Use Case**: Object detection, perimeter monitoring
- **Preprocessing**: Handles missing directional data

### 6. **IMU (Inertial Measurement Unit)**
- **Features**: Accelerometer and gyroscope data (X, Y, Z axes)
- **Use Case**: Motion detection, orientation tracking, vibration analysis
- **Preprocessing**: Standard sensor data handling

## Installation

### Prerequisites
```bash
pip install pandas scikit-learn numpy jupyter
```

### Dataset Requirements
Ensure you have the following CSV files in your project directory:
- `microphone_audio_sensor.csv`
- `thermal_sensor_data.csv`
- `em_sensor_data.csv`
- `gps_environment_data.csv`
- `proximity_sensor_data.csv`
- `imu_sensor_data.csv`

## Usage

### 1. Data Preprocessing and Merging

```python
import pandas as pd

# Load all sensor datasets
audio_df = pd.read_csv("microphone_audio_sensor.csv")
thermal_df = pd.read_csv("thermal_sensor_data.csv")
em_df = pd.read_csv("em_sensor_data.csv")
gps_df = pd.read_csv("gps_environment_data.csv")
proximity_df = pd.read_csv("proximity_sensor_data.csv")
imu_df = pd.read_csv("imu_sensor_data.csv")

# Preprocessing steps
audio_df.drop(columns=["description", "start_time", "end_time", "audio_pattern"], inplace=True)
thermal_df.drop(columns=["description", "night_vision"], inplace=True)
gps_df.drop(columns=["description", "timestamp", "location_type"], inplace=True)

# Encode categorical variables
gps_df["time_of_day"] = gps_df["time_of_day"].map({"day": 0, "night": 1})
gps_df["movement_status"] = gps_df["movement_status"].map({"moving": 0, "standing": 1, "stationary": 2})

# Process thermal matrix
thermal_df["thermal_matrix"] = thermal_df["thermal_matrix"].apply(lambda x: list(map(float, str(x).split(','))))
thermal_expanded = thermal_df["thermal_matrix"].apply(pd.Series)
thermal_expanded.columns = [f"thermal_{i}" for i in thermal_expanded.columns]
```

### 2. Model Training and Evaluation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report

# Prepare features and labels
X = pd.concat([imu, proximity, em, gps, thermal, X_mic], axis=1)
X_encoded = pd.get_dummies(X)
y_encoded = label_encoder.fit_transform(y)

# Initialize model and cross-validation
model = RandomForestClassifier(random_state=42)
loo = LeaveOneOut()

# Perform Leave-One-Out cross-validation
for train_idx, test_idx in loo.split(X_encoded):
    X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
```

### 3. Running the Complete Pipeline

```bash
jupyter notebook classification_model.ipynb
```

## Data Schema

### Expected CSV Structure

#### Audio Sensor CSV
```csv
peak_db,description,start_time,end_time,audio_pattern,label
41.0,ambient_noise,09:00,09:01,low_frequency,safe
92.0,alarm_sound,14:30,14:31,high_frequency,threat
```

#### Thermal Sensor CSV
```csv
thermal_matrix,description,night_vision,label
"22.0,22.0,23.0,22.0,22.0",normal_temp,enabled,safe
"38.0,42.0,40.0,45.0,41.0",heat_signature,enabled,threat
```

#### GPS Environment CSV
```csv
latitude,longitude,altitude,time_of_day,movement_status,description,timestamp,location_type,label
40.7128,-74.0060,10.0,day,moving,outdoor,2023-01-01T12:00:00,public,safe
40.7129,-74.0061,10.1,night,stationary,indoor,2023-01-01T20:00:00,restricted,threat
```

## Classification Categories

The system classifies scenarios into three categories:

1. **Safe**: Normal operational conditions
2. **Threat**: Potential security risk detected
3. **Failsafe Triggered**: Critical security breach requiring immediate response

## Model Performance

### Evaluation Metrics
- **Cross-Validation**: Leave-One-Out for robust evaluation with small datasets
- **Accuracy Score**: Overall prediction accuracy
- **Classification Report**: Precision, recall, and F1-score for each class
- **Support**: Number of samples per class

### Current Results
```
Accuracy: Variable (depends on dataset quality and size)
Classes: ["failsafe triggered", "safe", "threat"]
Validation: Leave-One-Out Cross-Validation
```

## Technical Implementation

### Data Preprocessing Pipeline
1. **Missing Value Handling**: NaN values preserved where appropriate
2. **Categorical Encoding**: One-hot encoding for categorical features
3. **Feature Engineering**: Thermal matrix flattening, timestamp processing
4. **Data Alignment**: Ensures all sensor data has matching sample counts

### Machine Learning Pipeline
1. **Algorithm**: Random Forest Classifier
2. **Feature Selection**: All available sensor features
3. **Cross-Validation**: Leave-One-Out (LOO-CV)
4. **Label Encoding**: String labels converted to numerical format

## File Structure

```
sensor-classification/
├── classification_model.ipynb          # Main analysis notebook
├── microphone_audio_sensor.csv        # Audio sensor data
├── thermal_sensor_data.csv            # Thermal imaging data
├── em_sensor_data.csv                 # Electromagnetic sensor data
├── gps_environment_data.csv           # GPS and environment data
├── proximity_sensor_data.csv          # Proximity measurements
├── imu_sensor_data.csv                # IMU accelerometer/gyroscope data
├── final_merged_sensor_data.csv       # Processed combined dataset
└── README.md                          # This file
```

## Troubleshooting

### Common Issues

1. **Dataset Size Mismatch**
   ```python
   min_rows = min(len(imu), len(proximity), len(em), len(gps), len(thermal), len(X_mic))
   dfs = [df.reset_index(drop=True).iloc[:min_rows] for df in dfs]
   ```

2. **Missing CSV Files**
   - Ensure all 6 CSV files are in the project directory
   - Check file names match exactly

3. **Data Format Issues**
   - Verify thermal matrix format (comma-separated values)
   - Check categorical value mappings

4. **Low Accuracy**
   - Increase dataset size
   - Review feature engineering
   - Consider different algorithms

## Optimization Strategies

### For Better Performance
1. **Feature Engineering**
   - Add statistical features (mean, std, min, max) for sensor groups
   - Create temporal features from timestamp data
   - Engineer interaction features between sensors

2. **Model Improvements**
   - Hyperparameter tuning with GridSearchCV
   - Ensemble methods (XGBoost, Gradient Boosting)
   - Deep learning approaches for temporal patterns

3. **Data Augmentation**
   - Synthetic data generation
   - Time-series augmentation techniques
   - Noise injection for robustness

## Future Enhancements

- **Real-time Processing**: Stream processing capabilities
- **Advanced ML Models**: LSTM/GRU for temporal patterns
- **Anomaly Detection**: Unsupervised learning for unknown threats
- **Sensor Fusion**: Advanced fusion algorithms
- **Deployment**: Edge computing and IoT integration
- **Visualization**: Real-time dashboard and monitoring

## Contributing

1. Fork the repository
2. Create feature branches
3. Add comprehensive tests
4. Submit pull requests with detailed descriptions

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
- Check data format requirements
- Verify all dependencies are installed
- Review preprocessing steps for data alignment
- Ensure sufficient data samples for meaningful classification
