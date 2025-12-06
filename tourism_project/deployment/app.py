import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="jitesh1351/tourism-package-prediction", filename="tourism_package_Prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package prediction System")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
Please enter the app details below to get a Package prediction.
""")

# User input
type_of_contact = st.selectbox("Type Of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Small Business", "Salaried", "Free Lancer", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])

age = st.number_input("Age", min_value=18.0, max_value=80.0, value=50.0, step=1.0)
city_tier = st.number_input("City Tier", min_value=1.0, max_value=3.0, value=1.0, step=1.0)
duration_of_pitch = st.number_input("Duration Of Pitch", min_value=5.0, max_value=150.0, value=50.0, step=1.0)
number_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=1.0, max_value=5.0, value=1.0, step=1.0)
number_of_follow_ups = st.number_input("Number Of Follow Ups", min_value=1.0, max_value=6.0, value=3.0, step=1.0)
preferred_property_star = st.number_input("Preferred Property Star", min_value=3.0, max_value=5.0, value=3.0, step=1.0)
number_of_trips = st.number_input("Number Of Trips", min_value=1.0, max_value=25.0, value=5.0, step=1.0)
passport = st.number_input("Passport", min_value=0, max_value=1.0, value=1.0)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
own_car = st.number_input("Own Car", min_value=0, max_value=1.0, value=1.0)
number_of_children_visiting	= st.number_input("Number Of Children Visiting", min_value=0, max_value=5.0, value=2.0)
monthly_income = st.number_input("Monthly Income", min_value=1000.0,max_value=100000.0, value=20000.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'type_of_contact': type_of_contact,
    'occupation': occupation,
    'gender': gender,
    'product_pitched': product_pitched,
    'marital_status': marital_status,
    'designation': designation,
    'age_in_year': age,
    'city_tier': city_tier,
    'duration_of_pitch': duration_of_pitch,
    'number_of_person_visiting': number_of_person_visiting,
    'number_of_follow_ups': number_of_follow_ups,
    'preferred_property_star': preferred_property_star,
    'number_of_trips': number_of_trips,
    'passport': passport,
    'pitch_satisfaction_score': pitch_satisfaction_score,
    'own_car':own_car,
    'number_of_children_visiting': number_of_children_visiting,
    'monthly_income': monthly_income
}])

# Predict button
if st.button("Predict Package"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Package purchased by customer is: {'Yes' if prediction == 0 else 'No'} ")
