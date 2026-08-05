import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("heart_model.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient's details below.")

# -----------------------------
# User Inputs
# -----------------------------

age = st.number_input("Age", min_value=1, max_value=120, value=30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)
gender = 1 if gender == "Male" else 0

cp = st.selectbox(
    "Chest Pain Type (cp)",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure (trestbps)",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol (chol)",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl (fbs)",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG (restecg)",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate (thalach)",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina (exang)",
    [0, 1]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels (ca)",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict"):

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    # Arrange columns exactly as model expects
    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    # Optional: Display input values
    with st.expander("View Input Data"):
        st.dataframe(input_data)