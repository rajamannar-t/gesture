# Gesture Recognition with Cultural Context

A machine learning project that recognizes hand gestures from real-time video input using computer vision and provides cultural context information about the gestures.

## Project Overview

This gesture recognition system combines:
- **Real-time hand detection** using MediaPipe for capturing hand landmarks
- **Machine learning classification** using Support Vector Machines (SVM) for gesture recognition
- **Cultural awareness** by providing meanings, regional usage, and offensive contexts of gestures
- **Live prediction** with webcam input for interactive gesture recognition

##  Features

- **Data Collection**: Capture hand gesture data frame-by-frame with custom gesture labels
- **Model Training**: Train an SVM classifier with feature scaling for accurate gesture recognition
- **Live Detection**: Real-time gesture recognition with cultural context display
- **Cultural Database**: JSON-based database storing gesture meanings across different regions
- **Gesture Landmarks**: 21-point hand landmark detection for precise gesture capture

##  Project Structure

```
gesture_project/
├── collect_data.py          # Script to collect hand gesture training data
├── train_model.py           # Script to train the SVM classification model
├── predict_live.py          # Real-time gesture recognition from webcam
├── gesture_data.csv         # Training dataset with hand landmarks and labels
├── cultural_data.json       # Cultural context for recognized gestures
├── gesture_model.pkl        # Trained model and scaler (auto-generated)
└── README.md               # This file
```

## 🛠️ Requirements

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- scikit-learn
- pandas
- joblib
- numpy

##  Installation

1. Clone or download this project
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - **Windows**: `.venv\Scripts\Activate.ps1`
   - **Linux/Mac**: `source .venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install opencv-python mediapipe scikit-learn pandas joblib
   ```

##  Usage

### Step 1: Collect Training Data

Run the data collection script to capture gestures:

```bash
python collect_data.py
```

**Instructions:**
- Enter the gesture name when prompted (e.g., "thumbs_up", "peace", "namaste")
- Position your hand in front of the webcam
- Press **'s'** to save sample frames
- Press **'q'** to quit and finish collecting for this gesture
- Repeat for different gestures to build your training dataset

**Output**: Adds hand landmark data to `gesture_data.csv`

### Step 2: Train the Model

Train the SVM classifier on collected gesture data:

```bash
python train_model.py
```

**What it does:**
- Loads gesture data from CSV
- Applies feature scaling using StandardScaler
- Splits data into training (80%) and testing (20%) sets
- Trains an SVM model with RBF kernel
- Reports accuracy on test set
- Saves trained model and scaler to `gesture_model.pkl`

### Step 3: Live Gesture Recognition

Run real-time gesture prediction:

```bash
python predict_live.py
```

**Features:**
- Displays live webcam feed with hand landmark visualization
- Predicts gesture in real-time
- Shows cultural context (meaning, region, emotion, offensive contexts)
- Press **'q'** to exit

##  Data Format

### gesture_data.csv
```
x0, y0, x1, y1, ..., x20, y20, label
0.5, 0.3, 0.52, 0.25, ..., 0.48, 0.42, thumbs_up
...
```
- **x0-x20, y0-y20**: 21 hand landmark coordinates (normalized 0-1)
- **label**: Gesture name/class

### cultural_data.json
```json
{
  "gesture_name": "Namaste",
  "region": "India",
  "meaning": "Respectful greeting",
  "emotion": "Respect, humility",
  "usage_context": "Formal and informal greetings",
  "offensive_in": [],
  "cultural_notes": "Traditional Indian greeting with folded hands."
}
```

##  How It Works

1. **Hand Detection**: MediaPipe detects hand landmarks (21 keypoints per hand)
2. **Feature Extraction**: Extracts x, y coordinates for each landmark
3. **Feature Scaling**: Normalizes features using StandardScaler for better SVM performance
4. **Classification**: SVM model predicts gesture class
5. **Cultural Lookup**: Retrieves and displays cultural context from JSON database

##  Model Details

- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Features**: 42 features (21 landmarks × 2 coordinates each)
- **Training**: Binary/Multi-class classification based on gesture variety
- **Optimization**: Feature scaling applied before training and prediction

##  Important Notes

- Ensure good lighting for accurate hand detection
- Collect diverse samples of each gesture (different angles, distances, hand sizes)
- More training samples = better model accuracy
- Model and scaler must be retrained after adding new gesture classes
- The system works best with one hand at a time

##  Contributing

Feel free to:
- Add more gesture types to the cultural database
- Collect more training data for better accuracy
- Implement additional models (Random Forest, Neural Networks)
- Add support for two-handed gestures
- Create a GUI interface

##  License

This project is open for educational and research purposes.

##  References

- [MediaPipe Hand Detection](https://mediapipe.dev/)
- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)
- [OpenCV Documentation](https://docs.opencv.org/)
