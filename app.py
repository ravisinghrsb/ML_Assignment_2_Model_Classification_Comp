"""
Streamlit Web Application for Wine Quality Prediction
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
    page_title="Wine Quality Prediction",
    page_icon="🍷",
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
        background-color: #8B0000;
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
    .good {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .bad {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
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
    st.sidebar.header("🍷 Wine Properties")
    st.sidebar.write("Enter the wine's chemical properties:")
    
    # Create input fields for all 12 features
    fixed_acidity = st.sidebar.slider("Fixed Acidity (g/dm³)", 3.0, 16.0, 7.0, 0.1)
    volatile_acidity = st.sidebar.slider("Volatile Acidity (g/dm³)", 0.0, 2.0, 0.5, 0.01)
    citric_acid = st.sidebar.slider("Citric Acid (g/dm³)", 0.0, 1.5, 0.3, 0.01)
    residual_sugar = st.sidebar.slider("Residual Sugar (g/dm³)", 0.0, 20.0, 2.0, 0.1)
    chlorides = st.sidebar.slider("Chlorides (g/dm³)", 0.0, 0.7, 0.08, 0.001)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide (mg/dm³)", 0.0, 100.0, 15.0, 1.0)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide (mg/dm³)", 0.0, 300.0, 50.0, 1.0)
    density = st.sidebar.slider("Density (g/cm³)", 0.98, 1.01, 0.996, 0.0001)
    pH = st.sidebar.slider("pH", 2.5, 4.5, 3.3, 0.01)
    sulphates = st.sidebar.slider("Sulphates (g/dm³)", 0.0, 2.0, 0.6, 0.01)
    alcohol = st.sidebar.slider("Alcohol (%)", 8.0, 15.0, 10.0, 0.1)
    wine_type = st.sidebar.selectbox("Wine Type", ["Red Wine", "White Wine"])
    
    # Convert wine type (matches training data: Red=1, White=0)
    wine_type_num = 1 if wine_type == "Red Wine" else 0
    
    # Create feature array (order matches training data)
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                         pH, sulphates, alcohol, wine_type_num]])
    
    return features

def main():
    """Main application function"""
    
    # Title
    st.title("🍷 Wine Quality Prediction System")
    st.markdown("### AI-Powered Wine Quality Assessment")
    st.write("This application uses 6 different machine learning models to predict wine quality (Good vs Bad).")
    
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
        st.subheader("📊 Wine Properties Summary")
        
        # Display input features in a nice format
        feature_names = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 
                        'Chlorides', 'Free SO₂', 'Total SO₂', 'Density', 
                        'pH', 'Sulphates', 'Alcohol', 'Wine Type']
        
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
            prob_good = probability[1] * 100
            prob_bad = probability[0] * 100
        except:
            prob_good = None
            prob_bad = None
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-box good">
                    <h2 style="color: #388e3c;">✨ Good Quality Wine</h2>
                    <p style="font-size: 18px;">The model predicts this wine has <strong>GOOD QUALITY</strong> (Rating ≥ 6).</p>
                    {f'<p style="font-size: 16px;"><strong>Confidence:</strong> {prob_good:.2f}%</p>' if prob_good else ''}
                    <p style="font-size: 14px; margin-top: 1rem;"><em>🍷 This wine meets high quality standards based on its chemical properties.</em></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-box bad">
                    <h2 style="color: #f57c00;">⚠️ Below Average Quality</h2>
                    <p style="font-size: 18px;">The model predicts this wine has <strong>BELOW AVERAGE QUALITY</strong> (Rating < 6).</p>
                    {f'<p style="font-size: 16px;"><strong>Confidence:</strong> {prob_bad:.2f}%</p>' if prob_bad else ''}
                    <p style="font-size: 14px; margin-top: 1rem;"><em>🔬 Consider adjusting chemical properties for better quality.</em></p>
                </div>
            """, unsafe_allow_html=True)
        
        # Additional information
        st.subheader("📈 Compare with All Models")
        
        predictions_all = {}
        for model_name, model_obj in models.items():
            pred = model_obj.predict(features_scaled)[0]
            predictions_all[model_display_names[model_name]] = "Good Quality" if pred == 1 else "Below Average"
        
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
