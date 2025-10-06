# train_and_save.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Load your processed dataset
# -----------------------------
DATA_PATH = "outputs/processed_dataset.csv"
assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"

df = pd.read_csv(DATA_PATH)

# Target column
target = "prognosis"   # adjust if named differently

# Features + target
X = df.drop(columns=[target])
y = df[target]

# Encode labels (diseases)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# 2. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------------
# 3. Train model (RandomForest for demo)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# -----------------------------
# 4. Save model + features
# -----------------------------
os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/weather_disease_model.joblib"
FEATURES_PATH = "models/feature_names.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

joblib.dump(model, MODEL_PATH)
joblib.dump(X.columns.tolist(), FEATURES_PATH)
joblib.dump(le, ENCODER_PATH)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Features saved to {FEATURES_PATH}")
print(f"✅ Label encoder saved to {ENCODER_PATH}")
