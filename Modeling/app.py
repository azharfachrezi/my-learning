import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_data():
    # Load the model and the feature list we saved earlier
    model = joblib.load('hospital_readmission_model_v1.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

try:
    model, features = load_data()
except FileNotFoundError:
    st.error("Error Model files not found. Make sure .pkl files are in the same folder!")
    st.stop()

# --- THE APP TITLE ---
st.title("Hospital Readmission Predictor")
st.write("Enter patient details below to estimate the risk of readmission witihin 30 days.")

# --- SIDEBAR (USER INPUTS) ---
st.sidebar.header("Patient Data")

# We only ask for the top 5 most important features (to keep it simple)
# But we will feed the model ZEROS for everything else.

# Input 1: Number of Inpatient Visits (The #1 Predictor)
inpatient_visits = st.sidebar.number_input(
    "Number of Past Inpatient Visits",
    min_value=0, max_value=20, value=0
)

# Input 2: Discharge Location (The Secret Weapon)
# We map simple names to the IDs the model understands
discharge_map = {
    "Home (Routine)" : 1,
    "Skilled Nursing Facility": 3,
    "Home Health Service": 6,
    "Left AMA (Against Advice)": 7,
    "Hospice": 11,
    "Rehab Facility": 22
}
discharge_choice = st.sidebar.selectbox("Discharge Destination", list(discharge_map.keys()))
discharge_id = discharge_map[discharge_choice]

# input 3: Lab Procedures
num_labs = st.sidebar.slider("Number of Lab Procedures", 1, 130, 40)

# Input 4: Lab Medications
num_meds = st.sidebar.slider("Number of Medications", 1, 80, 15)

# Input 5: Time in Hospital
time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 3)

# --- PREPARE DATA FOR MODEL ---
# The model expects ~50 columns (one-shot encoded). We can't just send 5 numbers.
# TRICK: Create a "Zero Row" with all the correct column names.
input_data = pd.DataFrame(0, index=[0], columns=features)

# Now fill in the user's values into the correct columns
# NOTE: WE have to be careful to match the column names exactly as they were trained!

# Simple numeric columns
if 'number_inpatient' in input_data.columns:
    input_data['number_inpatient'] = inpatient_visits
if 'num_lab_procedures' in input_data.columns:
    input_data['num_lab_procedures'] = num_labs
if 'num_medications' in input_data.columns:
    input_data['num_medications'] = num_meds
if 'time_in_hospital' in input_data.columns:
    input_data['time_in_hospital'] = time_in_hospital

# For Discharge ID, we need to handle it carefully
# If your model used it as a NUMBER (integer), we just put the number in.
if 'discharge_disposition_id' in input_data.columns:
    input_data['discharge_disposition_id'] = discharge_id

# If you One-Hot Encoded it manually (e.g. discharge_disposition_id_22), we would need extra logic.
# Based on our "Manual Scikit-Learn" run, we likely treated it as a number or handled encoding inside the pipeline.
# For this prototype, we'll assume it handles the raw number (LightGBM can handle it).

# --- THE PREDICTION BUTTON ---
if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of "Yes"

    st.write("---")
    st.subheader("Prediction Result:")

    if prediction == 1:
        st.error(f"HIGH RISK of Readmission ({probability:.1f} Probability)")
        st.write("Recommendation: Assign Case Manager for follow-up.")

    else:
        st.success(f"LOW RISK of Readmission ({probability:.1f} Probability)")
        st.write("Recommendation: Standard discharge protocol.")