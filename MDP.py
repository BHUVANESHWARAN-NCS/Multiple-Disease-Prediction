import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu  # Ensure this is installed

# Function to load models safely
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}")
        return None
    except pickle.UnpicklingError:
        st.error(f"⚠️ Error unpickling model: {model_path}")
        return None
    except Exception as e:
        st.error(f"⚠️ Unexpected error loading model: {e}")
        return None

# Load models with corrected filenames
Kidney_model = load_model("kidney_model6.pkl")
liver_model = load_model("india_liver_model1.pkl")
parkinson_model = load_model("parkinsons_model1.pkl")

# Check if models are loaded properly
if not Kidney_model:
    st.warning("⚠️ Kidney model failed to load.")
if not liver_model:
    st.warning("⚠️ Liver model failed to load.")
if not parkinson_model:
    st.warning("⚠️ Parkinson's model failed to load.")

# Sidebar Navigation
with st.sidebar:
    selected_model = option_menu(
        "Multiple Disease Prediction System 🏥",
        ["Home", "Kidney Prediction", "Liver Prediction", "Parkinsons Prediction"],
        icons=['house', 'activity', 'heart', 'person'],
        menu_icon="hospital-fill",
        default_index=0
    )

# Home Page
if selected_model == "Home":
    st.title("Welcome to the Multiple Disease Prediction System 🏥")
    st.write("""
        This tool helps you predict the risk of various diseases using AI models. 🚀  
        
        - **KIDNEY DISEASE** 💧  
        - **LIVER DISEASE** 🍂  
        - **PARKINSON'S DISEASE** 🧠  
        
        🛠️ **HOW TO USE**:  
        Select a model from the **SIDEBAR MENU** 📂 and input your data.  
        
        💡 Stay proactive about your health! 🌿  
    """)

# Function to display results
def display_result(condition_name, prediction, advice):
    if prediction == 1:
        st.error(f"⚠️ High risk of {condition_name}. 😔")
        st.write(f"💡 **Advice**: {advice}")
    else:
        st.success(f"✅ No significant risk of {condition_name}! 🎉")
        st.write(f"💡 **Advice**: {advice}")

# 🚑 **LIVER DISEASE PREDICTION**
if selected_model == "Liver Prediction":
    st.title("Liver Disease Prediction 🍂")

    # Input fields
    Age = st.number_input('Age', min_value=1)
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Gender = 1 if Gender == 'Male' else 0  # Encode gender
    Total_Bilirubin = st.number_input('Total Bilirubin', min_value=0.0)
    Direct_Bilirubin = st.number_input('Direct Bilirubin', min_value=0.0)
    Alkaline_Phosphotase = st.number_input('Alkaline Phosphotase', min_value=0)
    Alamine_Aminotransferase = st.number_input('Alamine Aminotransferase', min_value=0)
    Aspartate_Aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0)
    Total_Proteins = st.number_input('Total Proteins', min_value=0.0)
    Albumin = st.number_input('Albumin', min_value=0.0)
    Albumin_and_Globulin_Ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0)

    if st.button('Predict'):
        input_data = pd.DataFrame([{
            "Age": Age, "Gender": Gender, "Total_Bilirubin": Total_Bilirubin,
            "Direct_Bilirubin": Direct_Bilirubin, "Alkaline_Phosphotase": Alkaline_Phosphotase,
            "Alamine_Aminotransferase": Alamine_Aminotransferase,
            "Aspartate_Aminotransferase": Aspartate_Aminotransferase, "Total_Proteins": Total_Proteins,
            "Albumin": Albumin, "Albumin_and_Globulin_Ratio": Albumin_and_Globulin_Ratio
        }])

        if liver_model:
            expected_features = liver_model.feature_names_in_
            input_data = input_data.reindex(columns=expected_features, fill_value=0)
            prediction = liver_model.predict(input_data)[0]
            advice = "Maintain a healthy diet with minimal alcohol and fatty foods."
            display_result("Liver Disease", prediction, advice)
        else:
            st.error("🚨 Liver model is not available.")

# 🚰 **KIDNEY DISEASE PREDICTION**
elif selected_model == "Kidney Prediction":
    st.title("Kidney Disease Prediction 💧")

    features = {
        "age": st.number_input('Age', min_value=1, max_value=120),
        "bp": st.number_input('Blood Pressure', min_value=0.0),
        "sg": st.number_input('Specific Gravity', min_value=1.000, max_value=1.050, step=0.001, format="%.3f"),
        "al": st.number_input('Albumin', min_value=0),
        "su": st.number_input('Sugar', min_value=0),
        "bgr": st.number_input('Blood Glucose Random', min_value=0.0),
        "bu": st.number_input('Blood Urea', min_value=0.0),
        "sc": st.number_input('Serum Creatinine', min_value=0.0),
    }

    if st.button("Predict"):
        input_data = pd.DataFrame([features])

        if Kidney_model:
            expected_features = Kidney_model.feature_names_in_
            input_data = input_data.reindex(columns=expected_features, fill_value=0)
            prediction = Kidney_model.predict(input_data)[0]
            advice = "Drink plenty of water and reduce sodium intake."
            display_result("Kidney Disease", prediction, advice)
        else:
            st.error("🚨 Kidney model is not available.")

# 🧠 **PARKINSON'S DISEASE PREDICTION**
elif selected_model == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction 🧠")

    features = {
        "MDVP:Fo(Hz)": st.number_input('MDVP:Fo(Hz)', min_value=0.0),
        "MDVP:Fhi(Hz)": st.number_input('MDVP:Fhi(Hz)', min_value=0.0),
        "MDVP:Flo(Hz)": st.number_input('MDVP:Flo(Hz)', min_value=0.0),
        "MDVP:Jitter(%)": st.number_input('MDVP:Jitter(%)', min_value=0.0),
    }

    if st.button('Predict'):
        input_data = pd.DataFrame([features])

        if parkinson_model:
            expected_features = parkinson_model.feature_names_in_
            input_data = input_data.reindex(columns=expected_features, fill_value=0)
            prediction = parkinson_model.predict(input_data)[0]
            advice = "Exercise regularly and eat a balanced diet."
            display_result("Parkinson’s Disease", prediction, advice)
        else:
            st.error("🚨 Parkinson's model is not available.")
