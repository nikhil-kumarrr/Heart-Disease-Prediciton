import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('heart_model.joblib')
scaler = joblib.load('scaler.joblib')

# --- Page Config ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
/* Hospital-style background */
.stApp {
    background: linear-gradient(135deg, #e6f7f8, #f5ffff); /* Light teal-white medical look */
    font-family: 'Segoe UI', sans-serif;
}

/* Main Card Styling */
.card {
    background-color: #FFFFFF;
    padding: 25px;
    margin: 15px;
    border-radius: 20px;
    box-shadow: 0px 3px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: scale(1.02);
    box-shadow: 0px 8px 30px rgba(0, 150, 136, 0.3);
}

/* Section Titles */
.card h3 {
    color: #006064;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 20px;
}

/* Predict button - big rectangular glow */
div.stButton > button:first-child {
    background-color: #00796B;
    color: white;
    font-size: 26px;
    padding: 25px;
    width: 100%;
    border-radius: 16px;
    border: none;
    cursor: pointer;
    font-weight: bold;
    box-shadow: 0px 5px 25px rgba(0,0,0,0.2);
    transition: transform 0.2s, box-shadow 0.3s, background-color 0.3s;
}
div.stButton > button:first-child:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 30px #00BCD4, 0px 0px 60px #00BCD4 inset;
    background-color: #004D40;
}

/* Progress bar customization */
.stProgress > div > div {
    background-color: #00796B;
}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='text-align:center; color:#004D40; font-size:50px; font-weight:bold;'>🩺 Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Your virtual healthcare assistant for heart risk analysis</p>", unsafe_allow_html=True)

# --- Input Sections ---

# Personal Info
with st.container():
    st.markdown("<div class='card'><h3>🧍 Personal Info</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 25)
        cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
        sex = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        trestbps = st.number_input("Resting BP", 50, 250, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["Yes","No"])
    st.markdown("</div>", unsafe_allow_html=True)

# Vitals & Test Results
with st.container():
    st.markdown("<div class='card'><h3>🏃‍♂️ Vitals & Test Results</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
        thalach = st.number_input("Max Heart Rate", 50, 250, 150)
    with col2:
        exang = st.selectbox("Exercise Induced Angina", ["Yes","No"])
        oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0, step=0.1)
    with col3:
        slope = st.selectbox("Slope of ST segment (0-2)", [0,1,2])
        ca = st.selectbox("Number of major vessels (0-3)", [0,1,2,3])

    # Thalassemia centered neatly
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        thal = st.selectbox("Thalassemia (1=normal,2=fixed,3=reversible)", [1,2,3])
    st.markdown("</div>", unsafe_allow_html=True)

# --- Convert categorical ---
sex = 1 if sex=="Male" else 0
fbs = 1 if fbs=="Yes" else 0
exang = 1 if exang=="Yes" else 0

# --- Prediction & Risk Visualization ---
if st.button("Predict Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction_prob = model.predict_proba(input_scaled)[0][1]

    risk_percent = int(prediction_prob*100)
    st.markdown(f"<h3 style='text-align:center;'>Estimated Risk: {risk_percent}%</h3>", unsafe_allow_html=True)
    
    st.progress(risk_percent / 100)

    if prediction == 1:
        st.markdown("⚠️ **High Risk of Heart Disease!**", unsafe_allow_html=True)
        st.info("💡 Recommendations: Eat a heart-healthy diet, exercise regularly, avoid smoking, and monitor blood pressure.")
    else:
        st.markdown("✅ **Low Risk of Heart Disease**", unsafe_allow_html=True)
        st.info("💡 Tips: Maintain a balanced diet, exercise regularly, and go for annual check-ups.")
