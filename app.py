import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import os

st.set_page_config(
    page_title="Medical Diagnosis using AI",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="auto"
)


def set_background():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(
                rgba(0, 0, 0, 0.6), 
                rgba(0, 0, 0, 0.6)
            ),
            url("https://isnmedical.com/wp-content/uploads/2024/05/shutterstock_2063938817.webp");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        .main-title {
            color: #ffffff;
            text-align: center;
            padding: 20px;
            text-shadow: 2px 2px 8px #000000;
            font-size: 3em;
        }

        .css-1d391kg, .css-1v0mbdj {
            background-color: rgba(255, 255, 255, 0.85) !important;
            border-radius: 10px;
            padding: 1rem;
        }
        
        .app-description {
            color: #f0f0f0;
            text-align: center;
            max-width: 800px;
            margin: 0 auto 30px auto;
            font-size: 1.1em;
            line-height: 1.6;
            background-color: rgba(0, 0, 0, 0.5);
            padding: 15px 25px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        }
        
        </style>
    """, unsafe_allow_html=True)


def show_title():
    st.markdown('<h1 class="main-title">Medical Diagnosis Using AI</h1>', unsafe_allow_html=True)


def show_description():
    st.markdown("""
    <div class="app-description">
        <p>
            This AI-powered web application assists in the early detection of common medical conditions 
            including <strong>Diabetes</strong>, <strong>Heart Disease</strong>, <strong>Parkinson's Disease</strong>, 
            <strong>Lung Cancer</strong> and <strong>Thyroid Disorders</strong>. <br><br>
            By analyzing user-provided health parameters, the system uses trained machine learning models 
            to predict the likelihood of a condition — all within a simple, user-friendly interface.
        </p>
    </div>
    """, unsafe_allow_html=True)



# Get model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'Models')

# Load model safely
def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Model file not found: {filename}")
        return None

# Load models
diabetes_model = load_model('diabetes_model.sav')
heart_disease_model = load_model('heart_disease_model.sav')
parkinsons_model = load_model("parkinsons_model.sav")
lungs_disease_model = load_model("lungs_disease_model.sav")
thyroid_model = load_model("Thyroid_model.sav")


# Sidebar Logo & Header
with st.sidebar:
    st.markdown("### 🧬 Medical Diagnosis using AI")


# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes', 'Heart Disease', "Parkinson's", "Lung Cancer", 'Thyroid'],
        icons=['activity', 'heart', 'person', 'lungs', 'thermometer'],
        default_index=0
    )


# Dark/Light Theme
# with st.sidebar:
#     theme = st.radio("⚫⚪ Select Theme", ["Dark", "Light"])


# Pages
def diabetes_prediction():
    st.title("Diabetes Prediction")
    Pregnancies = st.number_input('Pregnancies', min_value=0)
    Glucose = st.number_input('Glucose', min_value=0)
    BloodPressure = st.number_input('Blood Pressure', min_value=0)
    SkinThickness = st.number_input('Skin Thickness', min_value=0)
    Insulin = st.number_input('Insulin', min_value=0)
    BMI = st.number_input('BMI', min_value=0.0)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0)
    Age = st.number_input('Age', min_value=0)

    if st.button("Predict"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result = diabetes_model.predict(input_data)[0]
        st.success("Diabetic" if result else "Not Diabetic")

def heart_disease_prediction():
    st.title("Heart Disease Prediction")
    inputs = [
        st.number_input('Age', min_value=0),
        st.number_input('Sex (1 = Male, 0 = Female)', min_value=0, max_value=1),
        st.number_input('Chest Pain Type (0–3)', min_value=0, max_value=3),
        st.number_input('Resting Blood Pressure'),
        st.number_input('Serum Cholesterol'),
        st.number_input('Fasting Blood Sugar > 120 (1 = True, 0 = False)', min_value=0, max_value=1),
        st.number_input('Rest ECG Results (0–2)', min_value=0, max_value=2),
        st.number_input('Max Heart Rate Achieved'),
        st.number_input('Exercise Induced Angina (1 = Yes, 0 = No)', min_value=0, max_value=1),
        st.number_input('Oldpeak (ST depression)', format="%.2f"),
        st.number_input('Slope (0–2)', min_value=0, max_value=2),
        st.number_input('CA (0–4)', min_value=0, max_value=4),
        st.number_input('Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)', min_value=0, max_value=2)
    ]

    if st.button("Predict"):
        result = heart_disease_model.predict([inputs])[0]
        st.success("Has Heart Disease" if result else "No Heart Disease")

def parkinsons_prediction():
    st.title("Parkinson's Disease Detection")
    features = [st.number_input(label) for label in [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]]

    if st.button("Predict"):
        result = parkinsons_model.predict([features])[0]
        st.success("Parkinson's Detected" if result else "No Parkinson's")

def lungs_disease_prediction():
    st.title("Lung Cancer Prediction")
    features = [st.number_input(label) for label in [
        'Age', 'Smokes', 'AreaQ', 'Alkhol', 'Dust Allergy', 'Genetic Risk',
        'Chronic Lung Disease', 'Fatigue', 'Coughing of Blood', 'Shortness of Breath',
        'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
        'Frequent Cold', 'Dry Cough', 'Chest Pain'
    ]]

    if st.button("Predict"):
        result = lungs_disease_model.predict([features])[0]
        st.success("Lung Cancer Detected" if result else "No Lung Cancer")

def thyroid_prediction():
    st.title("Thyroid Prediction")
    features = [st.number_input(label) for label in [
        'age', 'sex (1=Male, 0=Female)', 'on_thyroxine', 'query_on_thyroxine',
        'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
        'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
        'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'
    ]]

    if st.button("Predict"):
        result = thyroid_model.predict([features])[0]
        st.success("Thyroid Issue Detected" if result else "No Thyroid Issue")

set_background()
show_title()
show_description()

# Main logic
if selected == 'Diabetes':
    diabetes_prediction()
elif selected == 'Heart Disease':
    heart_disease_prediction()
elif selected == "Parkinson's":
    parkinsons_prediction()
elif selected == "Lung Cancer":
    lungs_disease_prediction()
elif selected == 'Thyroid':
    thyroid_prediction()


# Copyright label
st.markdown("""
<hr style="border-top: 1px solid #bbb; margin-top: 50px;">

<div style='text-align: center; color: #ffffff; font-size: 1.1em; font-weight: 500;
            background-color: rgba(0, 0, 0, 0.6); padding: 15px; border-radius: 8px;
            max-width: 500px; margin: 20px auto; box-shadow: 0 0 10px rgba(0,0,0,0.3);'>
    © 2025 | Created by <strong>Sheik Arshad Ibrahim</strong>
</div>
""", unsafe_allow_html=True)