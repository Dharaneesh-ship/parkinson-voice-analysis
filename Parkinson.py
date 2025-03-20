import os
import glob
import librosa
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Balances dataset

# Define dataset path
data_path = r"D:\parkinson"
MIN_DURATION = 2.0  # Minimum audio duration in seconds

# Function to get all audio files
def get_audio_files(root_dir):
    return glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)

# Function to extract features safely
def extract_features(file_path, n_mfcc=40):
    try:
        # Force librosa to use audioread by not specifying res_type
        y, sr = librosa.load(file_path, sr=None)

        # Skip empty or silent files
        if y is None or len(y) == 0 or np.all(y == 0):
            print(f"Skipping {file_path}: Empty or silent file")
            return None

        # Handle short audio: Pad if necessary
        min_samples = int(MIN_DURATION * sr)
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)), mode="constant")

        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, fmin=100.0)
        zcr = librosa.feature.zero_crossing_rate(y)

        # Compute deltas
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(mfccs_delta, axis=1),
            np.mean(mfccs_delta2, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1),
            np.mean(zcr, axis=1)
        ])

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Get all valid audio files
audio_files = get_audio_files(data_path)
X, y = [], []

for file in audio_files:
    features = extract_features(file)
    if features is not None:
        X.append(features)
        y.append(1 if "Dys" in file else 0)  # Parkinson’s = 1, Healthy = 0

X, y = np.array(X), np.array(y)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Evaluate Model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Single Audio File Test
test_file = "D:/parkinson/test_audio.wav"  # Change to your test file path

# Extract features for the test file
test_features = extract_features(test_file)
if test_features is not None:
    test_features = imputer.transform([test_features])  # Handle missing values
    test_features = scaler.transform(test_features)  # Apply scaling

    prediction = xgb_model.predict(test_features)[0]  # Make prediction

    # Display result
    result = "Parkinson’s Detected" if prediction == 1 else "Healthy"
    print(f"Test File: {test_file}")
    print(f"Prediction: {result}")
else:
    print(f"Could not process {test_file} (invalid or too short)")
