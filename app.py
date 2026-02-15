"""
Streamlit Web Application for Heart Disease Prediction
Supports 6 different ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .positive {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .negative {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models and scaler"""
    models = {}
    model_names = [
        'logistic_regression',
        'decision_tree',
        'knn',
        'naive_bayes',
        'random_forest_(ensemble)',
        'xgboost_(ensemble)'
    ]
    
    models_dir = Path('models')
    if not models_dir.exists():
        st.error("Models directory not found! Please train the models first by running train_models.py")
        return None, None
    
    for model_name in model_names:
        model_path = models_dir / f'{model_name}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    # Load scaler
    scaler_path = models_dir / 'scaler.pkl'
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    return models, scaler

def get_user_input():
    """Get user input from the sidebar"""
    st.sidebar.header("Patient Information")
    st.sidebar.write("Enter the patient's details below:")
    
    # Create input fields for all features
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", 
                               ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                               index=0)
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.sidebar.selectbox("Resting ECG", 
                                    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", 
                                  ["Upsloping", "Flat", "Downsloping"])
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", 
                                 ["Normal", "Fixed Defect", "Reversible Defect"])
    
    # Convert categorical variables to numerical
    sex_num = 1 if sex == "Male" else 0
    cp_num = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    fbs_num = 1 if fbs == "Yes" else 0
    restecg_num = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    exang_num = 1 if exang == "Yes" else 0
    slope_num = ["Upsloping", "Flat", "Downsloping"].index(slope)
    thal_num = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1
    
    # Create feature array
    features = np.array([[age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, 
                         thalach, exang_num, oldpeak, slope_num, ca, thal_num]])
    
    return features

def main():
    """Main application function"""
    
    # Title
    st.title("❤️ Heart Disease Prediction System")
    st.markdown("### Machine Learning-Based Cardiovascular Risk Assessment")
    st.write("This application uses 6 different machine learning models to predict the likelihood of heart disease.")
    
    # Load models
    models, scaler = load_models()
    
    if models is None or scaler is None:
        st.error("⚠️ Models not found! Please train the models first.")
        st.info("Run the following command in your terminal:\n```\npython train_models.py\n```")
        return
    
    # Get user input
    features = get_user_input()
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Model selection
    st.sidebar.markdown("---")
    st.sidebar.header("Model Selection")
    
    model_display_names = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'knn': 'k-Nearest Neighbors (kNN)',
        'naive_bayes': 'Naive Bayes',
        'random_forest_(ensemble)': 'Random Forest (Ensemble)',
        'xgboost_(ensemble)': 'XGBoost (Ensemble)'
    }
    
    selected_model = st.sidebar.selectbox(
        "Choose a model",
        list(models.keys()),
        format_func=lambda x: model_display_names[x]
    )
    
    # Prediction button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔍 Predict", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Patient Data Summary")
        
        # Display input features in a nice format
        feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 
                        'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 
                        'Exercise Angina', 'ST Depression', 'Slope', 'Major Vessels', 'Thalassemia']
        
        df_display = pd.DataFrame(features, columns=feature_names)
        st.dataframe(df_display, use_container_width=True)
    
    with col2:
        st.subheader("🤖 Selected Model")
        st.info(f"**{model_display_names[selected_model]}**")
        st.write(f"Total models available: **{len(models)}**")
    
    # Make prediction when button is clicked
    if predict_button:
        model = models[selected_model]
        prediction = model.predict(features_scaled)[0]
        
        # Get probability if available
        try:
            probability = model.predict_proba(features_scaled)[0]
            prob_positive = probability[1] * 100
            prob_negative = probability[0] * 100
        except:
            prob_positive = None
            prob_negative = None
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-box positive">
                    <h2 style="color: #d32f2f;">⚠️ High Risk of Heart Disease</h2>
                    <p style="font-size: 18px;">The model predicts a positive indication for heart disease.</p>
                    {f'<p style="font-size: 16px;"><strong>Confidence:</strong> {prob_positive:.2f}%</p>' if prob_positive else ''}
                    <p style="font-size: 14px; margin-top: 1rem;"><em>⚕️ Please consult with a healthcare professional for proper diagnosis and treatment.</em></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-box negative">
                    <h2 style="color: #388e3c;">✅ Low Risk of Heart Disease</h2>
                    <p style="font-size: 18px;">The model predicts a negative indication for heart disease.</p>
                    {f'<p style="font-size: 16px;"><strong>Confidence:</strong> {prob_negative:.2f}%</p>' if prob_negative else ''}
                    <p style="font-size: 14px; margin-top: 1rem;"><em>💚 Continue maintaining a healthy lifestyle and regular check-ups.</em></p>
                </div>
            """, unsafe_allow_html=True)
        
        # Additional information
        st.subheader("📈 Compare with All Models")
        
        predictions_all = {}
        for model_name, model_obj in models.items():
            pred = model_obj.predict(features_scaled)[0]
            predictions_all[model_display_names[model_name]] = "High Risk" if pred == 1 else "Low Risk"
        
        df_predictions = pd.DataFrame(predictions_all.items(), columns=['Model', 'Prediction'])
        st.dataframe(df_predictions, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>⚕️ This is a machine learning model for educational purposes only.</p>
            <p>Always consult with qualified healthcare professionals for medical advice.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
