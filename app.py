import streamlit as st
import pandas as pd
import joblib

# Load saved model & encoders
rf_model = joblib.load("rf_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")
category_modes = joblib.load("category_modes.pkl")

st.set_page_config(page_title="Stress Detection App", layout="centered")

st.title("🧠 Stress Detection System")
st.write("Enter your details to predict stress level")

# -------- INPUT FIELDS --------
Age = st.number_input("Age", 10, 100, 21)
Gender = st.selectbox("Gender", ["Male", "Female"])
Occupation = st.selectbox("Occupation", ["Student", "Employee", "Self-Employed"])
Marital_Status = st.selectbox("Marital Status", ["Single", "Married"])

Sleep_Duration = st.slider("Sleep Duration (hours)", 0, 12, 6)
Sleep_Quality = st.slider("Sleep Quality (1–5)", 1, 5, 3)
Wake_Up_Time = st.number_input("Wake Up Time", 0, 24, 7)
Bed_Time = st.number_input("Bed Time", 0, 24, 23)

Physical_Activity = st.slider("Physical Activity Level", 0, 5, 2)
Screen_Time = st.slider("Screen Time (hours)", 0, 12, 5)
Caffeine_Intake = st.slider("Caffeine Intake", 0, 5, 1)
Alcohol_Intake = st.slider("Alcohol Intake", 0, 5, 0)
Smoking_Habit = st.selectbox("Smoking Habit", [0, 1])

Work_Hours = st.slider("Work Hours", 0, 16, 8)
Travel_Time = st.slider("Travel Time (hours)", 0.0, 5.0, 1.0)
Social_Interactions = st.slider("Social Interactions", 0, 10, 5)
Meditation_Practice = st.selectbox("Meditation Practice", [0, 1])

Exercise_Type = st.selectbox("Exercise Type", ["Cardio", "Strength", "None"])

Blood_Pressure = st.number_input("Blood Pressure", 80, 180, 110)
Cholesterol_Level = st.number_input("Cholesterol Level", 100, 300, 180)
Blood_Sugar_Level = st.number_input("Blood Sugar Level", 60, 200, 90)

# -------- CREATE DATAFRAME --------
input_data = {
    "Age": Age,
    "Gender": Gender,
    "Occupation": Occupation,
    "Marital_Status": Marital_Status,
    "Sleep_Duration": Sleep_Duration,
    "Sleep_Quality": Sleep_Quality,
    "Wake_Up_Time": Wake_Up_Time,
    "Bed_Time": Bed_Time,
    "Physical_Activity": Physical_Activity,
    "Screen_Time": Screen_Time,
    "Caffeine_Intake": Caffeine_Intake,
    "Alcohol_Intake": Alcohol_Intake,
    "Smoking_Habit": Smoking_Habit,
    "Work_Hours": Work_Hours,
    "Travel_Time": Travel_Time,
    "Social_Interactions": Social_Interactions,
    "Meditation_Practice": Meditation_Practice,
    "Exercise_Type": Exercise_Type,
    "Blood_Pressure": Blood_Pressure,
    "Cholesterol_Level": Cholesterol_Level,
    "Blood_Sugar_Level": Blood_Sugar_Level
}

df = pd.DataFrame([input_data])

# -------- ENCODE CATEGORICAL --------
for col in label_encoders:
    encoder = label_encoders[col]
    if df[col][0] in encoder.classes_:
        df[col] = encoder.transform(df[col])
    else:
        df[col] = category_modes[col]

# -------- PREDICTION --------
if st.button("Predict Stress Level"):
    prediction = rf_model.predict(df)
    stress_label = target_encoder.inverse_transform(prediction)[0]

    st.success(f"🧩 Predicted Stress Level: **{stress_label}**")
