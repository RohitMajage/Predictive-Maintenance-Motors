import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------
# 1. Load Dataset
# ------------------------------
# Replace with your dataset (CSV with columns: vibration, temperature, label)
# label = {0: Healthy, 1: Warning, 2: Faulty}

data = pd.read_csv("motor_sensor_data.csv")
print("Sample Data:")
print(data.head())

# ------------------------------
# 2. Feature Extraction
# ------------------------------
def extract_features(df):
    features = pd.DataFrame()

    # Basic statistics
    features['vibration_mean'] = [np.mean(df['vibration'])]
    features['vibration_std'] = [np.std(df['vibration'])]
    features['vibration_kurtosis'] = [kurtosis(df['vibration'])]
    features['vibration_skew'] = [skew(df['vibration'])]

    # Frequency domain (FFT peak)
    fft_vals = np.abs(fft(df['vibration'].values))
    features['fft_peak'] = [np.max(fft_vals)]

    # Temperature stats
    features['temp_mean'] = [np.mean(df['temperature'])]
    features['temp_std'] = [np.std(df['temperature'])]
    features['temp_max'] = [np.max(df['temperature'])]

    return features

# Extract features for the whole dataset (window-based can also be applied)
feature_list = []
labels = []

# Assuming each row = one time sample with vibration & temperature
# If dataset is continuous time-series, split into windows
for i in range(0, len(data), 50):  # window size = 50 samples
    window = data.iloc[i:i+50]
    if len(window) == 50:
        feat = extract_features(window)
        feature_list.append(feat)
        labels.append(window['label'].mode()[0])

features = pd.concat(feature_list, ignore_index=True)
labels = np.array(labels)

print("Extracted Features Shape:", features.shape)

# ------------------------------
# 3. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# ------------------------------
# 4. Train Model
# ------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# 5. Evaluation
# ------------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Warning", "Faulty"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# ------------------------------
# 6. Visualization
# ------------------------------
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy","Warning","Faulty"], yticklabels=["Healthy","Warning","Faulty"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Motor Health Prediction")
plt.show()

# ------------------------------
# 7. Predict on New Sensor Data
# ------------------------------
new_data = pd.DataFrame({
    'vibration': np.random.normal(0.5, 0.1, 50),
    'temperature': np.random.normal(60, 2, 50)
})

new_features = extract_features(new_data)
new_prediction = model.predict(new_features)

state_map = {0: "Healthy", 1: "Warning", 2: "Faulty"}
print("New Motor Prediction:", state_map[new_prediction[0]])
