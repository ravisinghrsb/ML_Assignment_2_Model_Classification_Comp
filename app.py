"""
Enhanced Wine Quality Prediction - Model Comparison System
Complete ML pipeline with comprehensive model comparison and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Wine Quality ML Comparison",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8eaf6;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3f51b5;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .prediction-good {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-bad {
        background: linear-gradient(135deg, #ff7043 0%, #f4511e 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    h1 { color: #3f51b5; }
    h2 { color: #5c6bc0; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    """Load all trained models and scaler"""
    models_dir = Path('models')
    models = {}
    
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'k-Nearest Neighbors': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest_(ensemble).pkl',
        'XGBoost': 'xgboost_(ensemble).pkl'
    }
    
    for name, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
    
    # Load scaler
    scaler_path = models_dir / 'scaler.pkl'
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    return models, scaler

@st.cache_data
def load_metrics():
    """Load model performance metrics"""
    metrics_path = Path('models/model_metrics.csv')
    if metrics_path.exists():
        df = pd.read_csv(metrics_path, index_col=0)
        # Rename index for display
        model_rename = {
            'Logistic Regression': 'Logistic Regression',
            'Decision Tree': 'Decision Tree',
            'kNN': 'k-Nearest Neighbors',
            'Naive Bayes': 'Naive Bayes',
            'Random Forest (Ensemble)': 'Random Forest',
            'XGBoost (Ensemble)': 'XGBoost'
        }
        df.rename(index=model_rename, inplace=True)
        return df
    return None

@st.cache_data
def load_dataset():
    """Load the wine quality dataset"""
    dataset_path = Path('data/wine_quality.csv')
    if dataset_path.exists():
        return pd.read_csv(dataset_path)
    return None

@st.cache_data
def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def get_user_input():
    """Get wine properties from sidebar"""
    st.sidebar.header("🍷 Wine Properties Input")
    
    with st.sidebar.expander("📊 Chemical Properties", expanded=True):
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            fixed_acidity = st.slider("Fixed Acidity", 3.0, 16.0, 7.0, 0.1, help="Tartaric acid content (g/dm³)")
            citric_acid = st.slider("Citric Acid", 0.0, 1.5, 0.3, 0.01, help="Freshness and flavor (g/dm³)")
            chlorides = st.slider("Chlorides", 0.0, 0.7, 0.08, 0.001, help="Salt content (g/dm³)")
            total_sulfur_dioxide = st.slider("Total SO₂", 0.0, 300.0, 50.0, 1.0, help="Total sulfur dioxide (mg/dm³)")
            pH = st.slider("pH", 2.5, 4.5, 3.3, 0.01, help="Acidity level")
            alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.0, 0.1, help="Alcohol content")
        
        with col2:
            volatile_acidity = st.slider("Volatile Acidity", 0.0, 2.0, 0.5, 0.01, help="Acetic acid content (g/dm³)")
            residual_sugar = st.slider("Residual Sugar", 0.0, 20.0, 2.0, 0.1, help="Sweetness (g/dm³)")
            free_sulfur_dioxide = st.slider("Free SO₂", 0.0, 100.0, 15.0, 1.0, help="Free sulfur dioxide (mg/dm³)")
            density = st.slider("Density", 0.98, 1.01, 0.996, 0.0001, help="Density (g/cm³)")
            sulphates = st.slider("Sulphates", 0.0, 2.0, 0.6, 0.01, help="Wine additive (g/dm³)")
    
    st.sidebar.markdown("")
    wine_type = st.sidebar.radio("🍇 Wine Type", ["Red Wine", "White Wine"], horizontal=True)
    wine_type_num = 1 if wine_type == "Red Wine" else 0
    
    # Download Dataset Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Download Dataset")
    dataset = load_dataset()
    if dataset is not None:
        csv_data = convert_df_to_csv(dataset)
        st.sidebar.download_button(
            label="📊 Download Wine Quality Dataset",
            data=csv_data,
            file_name="wine_quality_dataset.csv",
            mime="text/csv",
            help="Download the complete wine quality dataset (6,497 samples)"
        )
        st.sidebar.info(f"📈 Dataset: {len(dataset)} samples, {len(dataset.columns)} features")
    
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                         pH, sulphates, alcohol, wine_type_num]])
    
    feature_dict = {
        'Fixed Acidity': fixed_acidity,
        'Volatile Acidity': volatile_acidity,
        'Citric Acid': citric_acid,
        'Residual Sugar': residual_sugar,
        'Chlorides': chlorides,
        'Free SO₂': free_sulfur_dioxide,
        'Total SO₂': total_sulfur_dioxide,
        'Density': density,
        'pH': pH,
        'Sulphates': sulphates,
        'Alcohol': alcohol,
        'Wine Type': wine_type
    }
    
    return features, feature_dict

def create_metrics_comparison_chart(metrics_df):
    """Create interactive comparison chart for all metrics"""
    fig = go.Figure()
    
    metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = ['#3f51b5', '#2196f3', '#00bcd4', '#009688', '#4caf50', '#8bc34a']
    
    for i, metric in enumerate(metrics_to_plot):
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df.index,
            y=metrics_df[metric],
            marker_color=colors[i],
            text=metrics_df[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Model Performance Comparison - All Metrics",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_radar_chart(metrics_df):
    """Create radar chart for model comparison"""
    fig = go.Figure()
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']
    colors = ['#3f51b5', '#2196f3', '#00bcd4', '#009688', '#4caf50', '#ff9800']
    
    for idx, model in enumerate(metrics_df.index):
        values = [metrics_df.loc[model, metric] for metric in metrics]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model,
            line_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Radar Chart",
        height=600
    )
    
    return fig

def display_prediction_comparison(models, features_scaled, feature_dict):
    """Display predictions from all models"""
    st.subheader("🎯 All Models Predictions")
    
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        pred = model.predict(features_scaled)[0]
        predictions[name] = pred
        
        try:
            proba = model.predict_proba(features_scaled)[0]
            probabilities[name] = proba[1] * 100  # Probability of good quality
        except:
            probabilities[name] = None
    
    # Create comparison dataframe
    comparison_data = []
    for name in models.keys():
        comparison_data.append({
            'Model': name,
            'Prediction': 'Good Quality ✅' if predictions[name] == 1 else 'Below Average ⚠️',
            'Confidence': f"{probabilities[name]:.1f}%" if probabilities[name] else "N/A"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display with color coding
    def color_prediction(val):
        if '✅' in val:
            return 'background-color: #c8e6c9'
        elif '⚠️' in val:
            return 'background-color: #ffccbc'
        return ''
    
    st.dataframe(
        df_comparison.style.map(color_prediction, subset=['Prediction']),
        hide_index=True,
        width='stretch'
    )
    
    # Consensus
    good_count = sum(1 for p in predictions.values() if p == 1)
    total_count = len(predictions)
    
    st.markdown("---")
    consensus_col1, consensus_col2, consensus_col3 = st.columns(3)
    
    with consensus_col1:
        st.metric("Models Predicting Good Quality", f"{good_count}/{total_count}")
    with consensus_col2:
        st.metric("Models Predicting Below Average", f"{total_count - good_count}/{total_count}")
    with consensus_col3:
        consensus = "Good Quality" if good_count > total_count/2 else "Below Average"
        st.metric("Consensus Prediction", consensus)

def process_batch_predictions(uploaded_file, models, scaler):
    """Process batch predictions from uploaded CSV file"""
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Expected columns
        expected_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                        'pH', 'sulphates', 'alcohol', 'wine_type']
        
        # Validate columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            return None, f"Missing columns: {', '.join(missing_cols)}"
        
        # Extract features
        X = df[expected_cols].copy()
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions with all models
        results = df.copy()
        
        for model_name, model in models.items():
            predictions = model.predict(X_scaled)
            results[f'{model_name}_Prediction'] = ['Good Quality' if p == 1 else 'Below Average' for p in predictions]
            
            try:
                probabilities = model.predict_proba(X_scaled)[:, 1]
                results[f'{model_name}_Confidence'] = [f"{p*100:.1f}%" for p in probabilities]
            except:
                results[f'{model_name}_Confidence'] = "N/A"
        
        # Add consensus prediction
        prediction_cols = [col for col in results.columns if col.endswith('_Prediction')]
        results['Consensus'] = results[prediction_cols].apply(
            lambda row: 'Good Quality' if list(row).count('Good Quality') > len(prediction_cols)/2 else 'Below Average', 
            axis=1
        )
        
        return results, None
        
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def main():
    """Main application"""
    
    # Header
    st.title("🍷 Wine Quality Classification - ML Model Comparison")
    st.markdown("### Advanced Machine Learning Pipeline with 6 Classification Algorithms")
    
    # Load models and metrics
    models, scaler = load_all_models()
    metrics_df = load_metrics()
    
    if models is None or scaler is None:
        st.error("⚠️ Models not found! Please train models first by running: `python train_models.py`")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Control Panel")
    
    # Get user input
    features, feature_dict = get_user_input()
    features_scaled = scaler.transform(features)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Performance", 
        "🎯 Predictions", 
        "📈 Visualizations",
        "🧪 Test Your Data",
        "ℹ️ About"
    ])
    
    # Tab 1: Model Performance
    with tab1:
        st.header("Model Performance Metrics")
        
        if metrics_df is not None:
            # Download metrics button at the top
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                metrics_csv = convert_df_to_csv(metrics_df)
                st.download_button(
                    label="📥 Download Metrics",
                    data=metrics_csv,
                    file_name="model_metrics.csv",
                    mime="text/csv",
                    help="Download model performance metrics as CSV"
                )
            
            st.markdown("")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📋 Complete Metrics Table")
                # Style the dataframe
                styled_df = metrics_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'AUC', 'F1']) \
                                            .format("{:.4f}")
                st.dataframe(styled_df, width='stretch')
            
            with col2:
                st.subheader("🏆 Best Performers")
                best_accuracy = metrics_df['Accuracy'].idxmax()
                best_auc = metrics_df['AUC'].idxmax()
                best_f1 = metrics_df['F1'].idxmax()
                
                st.metric("Best Accuracy", best_accuracy, f"{metrics_df.loc[best_accuracy, 'Accuracy']:.2%}")
                st.metric("Best AUC", best_auc, f"{metrics_df.loc[best_auc, 'AUC']:.2%}")
                st.metric("Best F1", best_f1, f"{metrics_df.loc[best_f1, 'F1']:.2%}")
            
            # Comparison charts
            st.markdown("---")
            st.plotly_chart(create_metrics_comparison_chart(metrics_df), use_container_width=True, key='metrics_chart')
        else:
            st.warning("Metrics file not found. Please train models first.")
    
    # Tab 2: Predictions
    with tab2:
        st.header("Wine Quality Predictions")
        
        # Display input features
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Input Features")
            # Create a copy and format properly
            feature_display = {k: (f"{v}" if isinstance(v, str) else f"{v:.4f}" if isinstance(v, float) else str(v)) 
                             for k, v in feature_dict.items()}
            feature_df = pd.DataFrame([feature_display]).T
            feature_df.columns = ['Value']
            st.table(feature_df)  # Use st.table instead of st.dataframe for mixed types
        
        with col2:
            st.subheader("🔍 Quick Stats")
            st.info(f"**Wine Type:** {feature_dict['Wine Type']}")
            st.info(f"**Alcohol Content:** {feature_dict['Alcohol']}%")
            st.info(f"**pH Level:** {feature_dict['pH']}")
            st.info(f"**Total Features:** 12")
        
        st.markdown("---")
        
        # Individual model predictions
        display_prediction_comparison(models, features_scaled, feature_dict)
        
        # Detailed predictions
        st.markdown("---")
        st.subheader("🔬 Detailed Model Analysis")
        
        selected_model = st.selectbox("Select a model for detailed view:", list(models.keys()))
        
        if selected_model:
            model = models[selected_model]
            prediction = model.predict(features_scaled)[0]
            
            try:
                proba = model.predict_proba(features_scaled)[0]
                prob_good = proba[1] * 100
                prob_bad = proba[0] * 100
                
                if prediction == 1:
                    st.markdown(f"""
                        <div class="prediction-good">
                            <h2>✨ Good Quality Wine</h2>
                            <p style="font-size: 20px;">Confidence: {prob_good:.1f}%</p>
                            <p>This wine is predicted to have <strong>GOOD QUALITY</strong> (Rating ≥ 6)</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="prediction-bad">
                            <h2>⚠️ Below Average Quality</h2>
                            <p style="font-size: 20px;">Confidence: {prob_bad:.1f}%</p>
                            <p>This wine is predicted to have <strong>BELOW AVERAGE QUALITY</strong> (Rating < 6)</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_good,
                    title = {'text': "Quality Probability"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            except Exception as e:
                st.write(f"Prediction: {'Good Quality' if prediction == 1 else 'Below Average'}")
    
    # Tab 3: Visualizations
    with tab3:
        st.header("Performance Visualizations")
        
        if metrics_df is not None:
            # Radar chart
            st.plotly_chart(create_radar_chart(metrics_df), use_container_width=True, key='radar_chart')
            
            # Individual metric comparisons
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig_acc = px.bar(
                    metrics_df.reset_index(),
                    x='index',
                    y='Accuracy',
                    title="Accuracy Comparison",
                    labels={'index': 'Model', 'Accuracy': 'Accuracy Score'},
                    color='Accuracy',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                # F1 Score comparison
                fig_f1 = px.bar(
                    metrics_df.reset_index(),
                    x='index',
                    y='F1',
                    title="F1 Score Comparison",
                    labels={'index': 'Model', 'F1': 'F1 Score'},
                    color='F1',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_f1, use_container_width=True)
    
    # Tab 4: Test Your Data
    with tab4:
        st.header("🧪 Test Your Own Wine Data")
        st.markdown("Upload a CSV file with wine properties to get batch predictions from all models.")
        
        # Instructions
        with st.expander("📋 Instructions & File Format", expanded=True):
            st.markdown("""
            ### Required CSV Format
            Your CSV file must contain the following 12 columns (in any order):
            
            | Column Name | Description | Example |
            |------------|-------------|---------|
            | `fixed_acidity` | Fixed acidity (g/dm³) | 7.0 |
            | `volatile_acidity` | Volatile acidity (g/dm³) | 0.5 |
            | `citric_acid` | Citric acid (g/dm³) | 0.3 |
            | `residual_sugar` | Residual sugar (g/dm³) | 2.0 |
            | `chlorides` | Chlorides (g/dm³) | 0.08 |
            | `free_sulfur_dioxide` | Free SO₂ (mg/dm³) | 15.0 |
            | `total_sulfur_dioxide` | Total SO₂ (mg/dm³) | 50.0 |
            | `density` | Density (g/cm³) | 0.996 |
            | `pH` | pH level | 3.3 |
            | `sulphates` | Sulphates (g/dm³) | 0.6 |
            | `alcohol` | Alcohol (%) | 10.0 |
            | `wine_type` | Wine type (0=White, 1=Red) | 0 or 1 |
            
            ### Sample CSV Template
            ```csv
            fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,wine_type
            7.0,0.5,0.3,2.0,0.08,15.0,50.0,0.996,3.3,0.6,10.0,0
            8.0,0.4,0.4,2.5,0.09,20.0,60.0,0.997,3.2,0.5,11.0,1
            ```
            """)
            
            # Download sample template
            sample_data = pd.DataFrame({
                'fixed_acidity': [7.0, 8.0, 6.5],
                'volatile_acidity': [0.5, 0.4, 0.6],
                'citric_acid': [0.3, 0.4, 0.2],
                'residual_sugar': [2.0, 2.5, 1.8],
                'chlorides': [0.08, 0.09, 0.07],
                'free_sulfur_dioxide': [15.0, 20.0, 12.0],
                'total_sulfur_dioxide': [50.0, 60.0, 45.0],
                'density': [0.996, 0.997, 0.995],
                'pH': [3.3, 3.2, 3.4],
                'sulphates': [0.6, 0.5, 0.7],
                'alcohol': [10.0, 11.0, 9.5],
                'wine_type': [0, 1, 0]
            })
            
            csv_template = convert_df_to_csv(sample_data)
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=csv_template,
                file_name="wine_data_template.csv",
                mime="text/csv",
                help="Download a sample CSV file with the correct format"
            )
        
        st.markdown("---")
        
        # File uploader
        col_upload1, col_upload2 = st.columns([2, 1])
        
        with col_upload1:
            uploaded_file = st.file_uploader(
                "📤 Upload your wine data CSV file",
                type=['csv'],
                help="Upload a CSV file containing wine properties"
            )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Process predictions
            with st.spinner("🔄 Processing predictions..."):
                results, error = process_batch_predictions(uploaded_file, models, scaler)
            
            if error:
                st.error(f"❌ {error}")
            elif results is not None:
                st.success(f"✅ Successfully processed {len(results)} wine samples!")
                
                # Summary statistics
                st.markdown("### 📊 Prediction Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Samples", len(results))
                with col2:
                    good_count = (results['Consensus'] == 'Good Quality').sum()
                    st.metric("Predicted Good Quality", good_count)
                with col3:
                    below_avg_count = (results['Consensus'] == 'Below Average').sum()
                    st.metric("Predicted Below Average", below_avg_count)
                with col4:
                    good_percentage = (good_count / len(results)) * 100
                    st.metric("Good Quality %", f"{good_percentage:.1f}%")
                
                st.markdown("---")
                
                # Display results
                st.markdown("### 📋 Detailed Predictions")
                
                # Option to show/hide original features
                show_features = st.checkbox("Show original wine properties", value=False)
                
                if show_features:
                    display_cols = results.columns.tolist()
                else:
                    # Show only prediction columns
                    prediction_cols = [col for col in results.columns if 'Prediction' in col or 'Confidence' in col or col == 'Consensus']
                    display_cols = prediction_cols
                
                # Color coding function
                def highlight_predictions(val):
                    if val == 'Good Quality':
                        return 'background-color: #c8e6c9; color: black'
                    elif val == 'Below Average':
                        return 'background-color: #ffccbc; color: black'
                    return ''
                
                # Apply styling
                styled_results = results[display_cols].style.map(highlight_predictions)
                
                st.dataframe(styled_results, width='stretch', height=400)
                
                # Download results
                st.markdown("---")
                st.markdown("### 💾 Download Results")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    results_csv = convert_df_to_csv(results)
                    st.download_button(
                        label="📥 Download Complete Results",
                        data=results_csv,
                        file_name="wine_quality_predictions.csv",
                        mime="text/csv",
                        help="Download all predictions with original data"
                    )
                
                with col_dl2:
                    summary_cols = ['Consensus'] + [col for col in results.columns if 'Prediction' in col]
                    summary_df = results[summary_cols]
                    summary_csv = convert_df_to_csv(summary_df)
                    st.download_button(
                        label="📥 Download Summary Only",
                        data=summary_csv,
                        file_name="predictions_summary.csv",
                        mime="text/csv",
                        help="Download only the prediction results"
                    )
                
                # Model agreement analysis
                st.markdown("---")
                st.markdown("### 🤝 Model Agreement Analysis")
                
                prediction_cols = [col for col in results.columns if col.endswith('_Prediction')]
                
                # Calculate agreement for each sample
                def calculate_agreement(row):
                    predictions = [row[col] for col in prediction_cols]
                    good_count = predictions.count('Good Quality')
                    return f"{good_count}/{len(predictions)}"
                
                results['Model_Agreement'] = results.apply(calculate_agreement, axis=1)
                
                # Show agreement distribution
                agreement_counts = results['Model_Agreement'].value_counts().sort_index()
                
                fig_agreement = px.bar(
                    x=agreement_counts.index,
                    y=agreement_counts.values,
                    labels={'x': 'Models Predicting Good Quality', 'y': 'Number of Samples'},
                    title='Distribution of Model Agreement',
                    color=agreement_counts.values,
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_agreement, use_container_width=True)
        
        else:
            st.info("👆 Upload a CSV file to get started with batch predictions!")
    
    # Tab 5: About
    with tab5:
        st.header("About This Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Dataset Information")
            st.markdown("""
            - **Name:** Wine Quality Dataset
            - **Source:** UCI Machine Learning Repository
            - **Samples:** 6,497 wines (1,599 red + 4,898 white)
            - **Features:** 12 chemical properties
            - **Target:** Binary classification (Good/Below Average)
            """)
            
            # Dataset Download Button
            st.markdown("")
            dataset = load_dataset()
            if dataset is not None:
                csv_data = convert_df_to_csv(dataset)
                st.download_button(
                    label="📥 Download Complete Dataset",
                    data=csv_data,
                    file_name="wine_quality_dataset.csv",
                    mime="text/csv",
                    help="Download the full wine quality dataset in CSV format"
                )
                
                # Show dataset preview
                with st.expander("👁️ Preview Dataset", expanded=False):
                    st.dataframe(dataset.head(10), width='stretch')
                    st.caption(f"Showing first 10 rows of {len(dataset)} total samples")
            
            st.subheader("🤖 Machine Learning Models")
            st.markdown("""
            1. **Logistic Regression** - Linear classifier
            2. **Decision Tree** - Rule-based classifier
            3. **k-Nearest Neighbors** - Instance-based learning
            4. **Naive Bayes** - Probabilistic classifier
            5. **Random Forest** - Ensemble of decision trees
            6. **XGBoost** - Gradient boosting ensemble
            """)
        
        with col2:
            st.subheader("📊 Evaluation Metrics")
            st.markdown("""
            - **Accuracy:** Overall correctness
            - **AUC:** Area Under ROC Curve
            - **Precision:** True positive rate
            - **Recall:** Sensitivity
            - **F1 Score:** Harmonic mean of precision and recall
            - **MCC:** Matthews Correlation Coefficient
            """)
            
            st.subheader("🎯 Features")
            st.markdown("""
            - **Real-time Predictions** from 6 models
            - **Interactive Visualizations** for comparison
            - **Comprehensive Metrics** display
            - **Consensus Prediction** system
            - **Model Performance Analysis**
            """)
        
        st.markdown("---")
        st.info("💡 **Tip:** Adjust the wine properties in the sidebar to see how different models predict wine quality!")

if __name__ == "__main__":
    main()
