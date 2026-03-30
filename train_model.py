import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv('gesture_data.csv')

print("Columns found:", data.columns)

if 'label' not in data.columns:
    print("ERROR: 'label' column missing in CSV")
    exit()

X = data.drop('label', axis=1)
y = data['label']

# ✅ Feature scaling (IMPORTANT)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = SVC(kernel='rbf')   # better than linear
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
print("Real Accuracy:", accuracy_score(y_test, y_pred))

# Save BOTH model + scaler
joblib.dump((model, scaler), 'gesture_model.pkl')

print("Model saved successfully")