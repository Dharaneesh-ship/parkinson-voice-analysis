# Parkinson's Disease Detection from Speech Patterns

## Project Overview
This project aims to detect early signs of Parkinson’s disease by analyzing a patient's speech patterns using machine learning. The model extracts various speech features from audio recordings and classifies whether the patient is likely to have Parkinson’s disease.

## Dataset
- The dataset consists of `.wav` audio files stored in the `D:\parkinson` directory.
- Files containing "Dys" in their names are considered Parkinson’s patients (label `1`), while others are considered healthy (label `0`).

## Features Extracted
- **MFCC (Mel-frequency cepstral coefficients)**: Captures timbral properties of speech.
- **Chroma Features**: Represents tonal content.
- **Spectral Contrast**: Measures the difference between peaks and valleys in a spectrum.
- **Zero Crossing Rate (ZCR)**: Counts the rate at which the signal changes sign.
- **Deltas & Delta-Delta Features**: Measures changes in MFCCs over time.

## Model & Techniques Used
- **Feature Extraction**: `Librosa`
- **Data Imputation**: Missing values handled with `SimpleImputer`.
- **Class Imbalance Handling**: `SMOTE` (Synthetic Minority Over-sampling Technique).
- **Feature Scaling**: `StandardScaler`
- **Classifier**: `XGBoost` (eXtreme Gradient Boosting).

## Setup & Installation

### Prerequisites
Make sure you have the following libraries installed:

```bash
pip install numpy librosa xgboost scikit-learn imbalanced-learn
```

### Run the Model
1. Place all `.wav` files in the `D:\parkinson` directory.
2. Run the script:

   ```bash
   python parkinson_detection.py
   ```

3. The model will train and display accuracy along with a classification report.

## Testing on a New Audio File
- Place the test audio file (`.wav`) in the dataset directory.
- Modify the `test_file` path in the script.
- Run the script to get the prediction (`Parkinson’s Detected` or `Healthy`).

## Results & Evaluation
- The model is evaluated using accuracy and classification metrics.
- Handles class imbalance using SMOTE.
- Provides an automated pipeline for feature extraction, classification, and prediction.

## Future Improvements
- Use additional datasets for better generalization.
- Experiment with deep learning-based models like CNNs or LSTMs.
- Optimize hyperparameters for improved accuracy.

