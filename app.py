import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("❤️ Heart Disease Prediction System")

st.write("Enter patient details:")

# Input fields
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [1,0])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1,0])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Number of vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (1-3)", [1,2,3])

# Prediction button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")