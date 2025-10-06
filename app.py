# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weather & Disease Prediction", layout="wide")
st.title("🧑‍⚕️ Weather & Disease Prediction (Demo)")

# --------------------------
# Load model + features + encoder
# --------------------------
MODEL_PATH = "models/weather_disease_model.joblib"
FEATURE_PATH = "models/feature_names.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
assert os.path.exists(FEATURE_PATH), f"Feature names not found at {FEATURE_PATH}"
assert os.path.exists(ENCODER_PATH), f"Label encoder not found at {ENCODER_PATH}"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)
label_encoder = joblib.load(ENCODER_PATH)

SYMPTOM_LIST = [
    "nausea", "joint_pain", "abdominal_pain", "high_fever", "chills", "fatigue",
    "runny_nose", "pain_behind_the_eyes", "dizziness", "headache", "chest_pain",
    "vomiting", "cough", "shivering", "asthma_history", "high_cholesterol",
    "diabetes", "obesity", "hiv_aids", "nasal_polyps", "asthma",
    "high_blood_pressure", "severe_headache", "weakness", "trouble_seeing",
    "fever", "body_aches", "sore_throat", "sneezing", "diarrhea",
    "rapid_breathing", "rapid_heart_rate", "swollen_glands", "rashes",
    "sinus_headache", "facial_pain"
]

# --------------------------
# UI inputs
# --------------------------
st.subheader("🌦️ Weather & Demographics")
col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input("🌡️ Temperature (°C)", value=30.0)
with col2:
    humidity = st.number_input("💧 Humidity (%)", value=50.0)
with col3:
    wind_speed = st.number_input("🌬️ Wind Speed (km/h)", value=10.0)

col4, col5 = st.columns(2)
with col4:
    age = st.number_input("🎂 Age", min_value=0, max_value=120, value=25)
with col5:
    gender = st.radio("⚧️ Gender", ["Male", "Female"])
    gender_encoded = 0 if gender == "Male" else 1

st.subheader("🤒 Symptoms")
selected_symptoms = st.multiselect("Type or select symptoms:", SYMPTOM_LIST)

symptom_sum = len(selected_symptoms)
temp_x_fever = temperature if "fever" in selected_symptoms else 0

# --------------------------
# Build feature vector
# --------------------------
user_features = {
    "Temperature (C)": temperature,
    "Humidity": humidity,
    "Wind Speed (km/h)": wind_speed,
    "Age": age,
    "Gender": gender_encoded,
    "symptom_sum": symptom_sum,
    "temp_x_fever": temp_x_fever
}

for symptom in SYMPTOM_LIST:
    user_features[symptom] = 1 if symptom in selected_symptoms else 0

X_input = pd.DataFrame([[user_features.get(f, 0) for f in feature_names]],
                       columns=feature_names)

# --------------------------
# Prediction
# --------------------------
if st.button("🔮 Predict Disease"):
    pred_idx = model.predict(X_input)[0]
    prediction = label_encoder.inverse_transform([pred_idx])[0]
    proba = model.predict_proba(X_input)[0]
    classes = label_encoder.inverse_transform(np.arange(len(proba)))

    st.success(f"### ✅ Predicted Disease: **{prediction}**")

    # --- Top-5 Probability Chart ---
    st.write("### 📊 Probability Distribution (Top 5)")
    top_idx = np.argsort(proba)[::-1][:5]
    top_diseases = [classes[i] for i in top_idx]
    top_probs = [proba[i] for i in top_idx]

    fig, ax = plt.subplots()
    ax.barh(top_diseases[::-1], np.array(top_probs[::-1]))
    ax.set_xlabel("Probability")
    ax.set_title("Top 5 Most Likely Diseases")
    st.pyplot(fig)
