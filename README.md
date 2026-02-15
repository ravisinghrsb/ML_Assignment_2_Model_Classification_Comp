# Machine Learning Assignment - Heart Disease Classification

## Project Title
**Heart Disease Prediction using Multiple Machine Learning Algorithms**

## Student Information
- **Assignment**: ML Assignment 2
- **Project Type**: Binary Classification (MLOps Pipeline)
- **Dataset**: Heart Disease Dataset (UCI Repository)
- **Date**: February 2026

---

## 📋 Table of Contents
1. [Assignment Overview](#assignment-overview)
2. [Dataset Description](#dataset-description)
3. [Machine Learning Algorithms](#machine-learning-algorithms)
4. [Methodology & Implementation](#methodology--implementation)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [Usage Instructions](#usage-instructions)
8. [Results & Performance](#results--performance)
9. [Web Application](#web-application)
10. [Technologies Used](#technologies-used)
11. [Conclusion](#conclusion)

---

## 📋 Assignment Overview

This project implements a complete Machine Learning pipeline for predicting heart disease using **6 different classification algorithms**. The system includes data preprocessing, model training, evaluation, and deployment through an interactive web application built with Streamlit.

### Problem Statement
Given various clinical parameters of a patient, predict whether they have heart disease or not (binary classification: 0 = No Disease, 1 = Disease Present).

### Objectives
1. ✅ Implement 6 different classification algorithms
2. ✅ Compare model performances
3. ✅ Create a complete ML pipeline from data to deployment
4. ✅ Build an interactive web application
5. ✅ Document the entire process

---

## 📊 Dataset Description

### Source
- **Dataset Name**: Heart Disease Dataset
- **Source**: UCI Machine Learning Repository / Kaggle
- **Total Samples**: 303 patients (293 after preprocessing)
- **Features**: 13 clinical attributes
- **Target Variable**: Binary (0 = No disease, 1 = Disease present)
- **Class Distribution**: Relatively balanced

### Features Description

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| age | Age in years | Continuous | 20-100 |
| sex | Gender (1 = male, 0 = female) | Binary | 0-1 |
| cp | Chest pain type (0-3) | Categorical | 0-3 |
| trestbps | Resting blood pressure (mm Hg) | Continuous | 80-200 |
| chol | Serum cholesterol (mg/dl) | Continuous | 100-600 |
| fbs | Fasting blood sugar > 120 mg/dl | Binary | 0-1 |
| restecg | Resting ECG results (0-2) | Categorical | 0-2 |
| thalach | Maximum heart rate achieved | Continuous | 60-220 |
| exang | Exercise induced angina | Binary | 0-1 |
| oldpeak | ST depression induced by exercise | Continuous | 0-6 |
| slope | Slope of peak exercise ST segment | Categorical | 0-2 |
| ca | Number of major vessels (0-3) | Discrete | 0-3 |
| thal | Thalassemia (1-3) | Categorical | 1-3 |
| **target** | **Heart disease diagnosis** | **Binary** | **0-1** |

---

## 🤖 Machine Learning Algorithms

This project implements and compares **6 different classification algorithms**:

### Model Comparison Table - All Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Naive Bayes | 0.8644 | 0.9336 | 0.8571 | 0.9091 | 0.8824 | 0.7244 |
| kNN | 0.8475 | 0.9260 | 0.8529 | 0.8788 | 0.8657 | 0.6897 |
| Random Forest (Ensemble) | 0.8305 | 0.9307 | 0.7949 | 0.9394 | 0.8611 | 0.6625 |
| Logistic Regression | 0.7966 | 0.9021 | 0.7838 | 0.8788 | 0.8286 | 0.5863 |
| XGBoost (Ensemble) | 0.7797 | 0.8695 | 0.7632 | 0.8788 | 0.8169 | 0.5523 |
| Decision Tree | 0.6949 | 0.6865 | 0.7143 | 0.7576 | 0.7353 | 0.3769 |

**Metrics Explanation:**
- **Accuracy**: Overall correctness of predictions (higher is better)
- **AUC**: Area Under ROC Curve - model's ability to distinguish between classes (higher is better)
- **Precision**: Proportion of positive predictions that are correct (higher is better)
- **Recall**: Proportion of actual positives correctly identified (higher is better)
- **F1**: Harmonic mean of Precision and Recall (higher is better)
- **MCC**: Matthews Correlation Coefficient - quality of binary classification (range: -1 to 1, higher is better)

**Key Findings:**
- 🥇 **Best Overall**: Naive Bayes achieved 86.44% accuracy with highest AUC (0.9336)
- 🎯 **Highest AUC**: Naive Bayes (0.9336) - best class discrimination
- ⚡ **Best Precision**: Naive Bayes (0.8571) - fewer false positives
- 🔍 **Best Recall**: Random Forest (0.9394) - catches most positive cases
- 📊 **Best F1-Score**: Naive Bayes (0.8824) - best balance
- ✅ **Best MCC**: Naive Bayes (0.7244) - most reliable predictions

---

### Detailed Algorithm Descriptions

### 1. Logistic Regression
- **Type**: Linear classifier
- **Description**: Statistical model for binary classification using logistic function
- **Advantages**: Fast, interpretable, good baseline
- **Hyperparameters**: max_iter=1000, random_state=42

### 2. Decision Tree Classifier
- **Type**: Tree-based non-linear classifier
- **Description**: Splits data based on feature values forming a tree structure
- **Advantages**: Easy to visualize, no feature scaling needed, handles non-linearity
- **Hyperparameters**: random_state=42

### 3. k-Nearest Neighbors (kNN)
- **Type**: Instance-based learning
- **Description**: Classifies based on majority vote of K nearest neighbors
- **Advantages**: Non-parametric, simple, captures local patterns
- **Hyperparameters**: n_neighbors=5

### 4. Naive Bayes (Gaussian)
- **Type**: Probabilistic classifier
- **Description**: Based on Bayes theorem with independence assumption
- **Advantages**: Fast training, works well with small datasets, probabilistic
- **Hyperparameters**: Default (var_smoothing=1e-9)

### 5. Random Forest (Ensemble)
- **Type**: Ensemble learning (Bagging)
- **Description**: Multiple decision trees combined through voting
- **Advantages**: Reduces overfitting, handles non-linear relationships, robust
- **Hyperparameters**: n_estimators=100, random_state=42

### 6. XGBoost (Ensemble)
- **Type**: Gradient boosting ensemble
- **Description**: Advanced gradient boosting algorithm that builds trees sequentially
- **Advantages**: High performance, handles missing values, regularization to prevent overfitting
- **Hyperparameters**: n_estimators=100, random_state=42, eval_metric='logloss'

---

## 🛠️ Methodology & Implementation

### Complete ML Pipeline

#### Step 1: Data Loading
```python
- Load heart.csv dataset using pandas
- Verify data integrity and dimensions
- Check for missing values
```

#### Step 2: Data Preprocessing
```python
- Separate features (X) and target variable (y)
- Split data: 80% training (234 samples), 20% testing (59 samples)
- Use stratified split to maintain class distribution
- Apply StandardScaler for feature normalization
- Save scaler for deployment
```

#### Step 3: Model Training
```python
- Initialize all 6 classification algorithms
- Train each model on training data
- Use consistent random_state=42 for reproducibility
- Measure training time for each model
```

#### Step 4: Model Evaluation
```python
- Test on held-out test set
- Calculate accuracy score
- Generate classification report (Precision, Recall, F1-Score)
- Compare all models
```

#### Step 5: Model Persistence
```python
- Save trained models using pickle
- Save StandardScaler for consistent preprocessing
- Store in models/ directory
```

#### Step 6: Deployment
```python
- Build interactive Streamlit web application
- Load saved models and scaler
- Enable real-time predictions
- Multi-model comparison feature
```

---

## 📁 Project Structure

```
mlassignment/
│
├── app.py                     # Streamlit web application
├── train_models.py           # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation (this file)
│
├── data/                     # Dataset directory
│   ├── heart.csv            # Heart disease dataset (303 samples)
│   └── README.md            # Dataset documentation
│
├── models/                   # Saved trained models
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest_(ensemble).pkl
│   ├── xgboost_(ensemble).pkl
│   ├── scaler.pkl           # Feature scaler
│   ├── model_metrics.csv    # Model evaluation metrics
│   └── README.md            # Models documentation
│
├── .gitignore               # Git ignore file
└── .github/
    └── copilot-instructions.md  # Project guidelines
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download Repository
```bash
cd mlassignment
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- streamlit==1.30.0
- scikit-learn==1.3.2
- pandas==2.1.4
- numpy==1.26.2
- matplotlib==3.8.2
- seaborn==0.13.0

---

## 💻 Usage Instructions

### Training the Models

Run the training script to train all 6 models:

```bash
python train_models.py
```

**What happens:**
1. Loads `data/heart.csv`
2. Preprocesses and splits data (80/20)
3. Trains all 6 classification models
4. Evaluates each model on test set
5. Saves models to `models/` directory
6. Displays accuracy comparison

**Expected Output:**
```
================================================================================
Heart Disease Classification - Model Training Pipeline
================================================================================
Loading data from data/heart.csv...
Dataset shape: (293, 14)

Preprocessing data...
Training set size: 234
Testing set size: 59

Training and Evaluating Models...

[Model training with detailed metrics for each model...]

Logistic Regression:
Accuracy:  0.7966
AUC:       0.9021
Precision: 0.7838
Recall:    0.8788
F1-Score:  0.8286
MCC:       0.5863

[Similar output for all 6 models...]

================================================================================
Model Comparison Table - All Evaluation Metrics
================================================================================

ML Model Name               Accuracy        AUC  Precision     Recall         F1        MCC
-----------------------------------------------------------------------------------------------
Naive Bayes                   0.8644     0.9336     0.8571     0.9091     0.8824     0.7244
kNN                           0.8475     0.9260     0.8529     0.8788     0.8657     0.6897
Random Forest (Ensemble)      0.8305     0.9307     0.7949     0.9394     0.8611     0.6625
Logistic Regression           0.7966     0.9021     0.7838     0.8788     0.8286     0.5863
XGBoost (Ensemble)            0.7797     0.8695     0.7632     0.8788     0.8169     0.5523
Decision Tree                 0.6949     0.6865     0.7143     0.7576     0.7353     0.3769

✓ Training completed successfully!
✓ All models and scaler saved to 'models/' directory
✓ Metrics saved to 'models/model_metrics.csv'
```

### Running the Web Application

Launch the Streamlit web app:

```bash
streamlit run app.py
```

The application will open in your default browser at: `http://localhost:8501`

### Using the Web Application

1. **Enter Patient Information**: Use sidebar inputs for 13 clinical features
   - Age, Sex, Chest Pain Type
   - Blood Pressure, Cholesterol
   - ECG results, Heart Rate
   - And more...

2. **Select Model**: Choose from 6 trained models

3. **Get Prediction**: Click "🔍 Predict" button
   - View risk assessment
   - See confidence scores
   - Get health recommendations

4. **Compare Models**: View predictions from all models simultaneously

---

## 📊 Results & Performance

### Model Performance Summary

**Dataset**: 293 samples (234 training, 59 testing) | **Evaluation**: Test set performance

| Rank | Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|------|-------|----------|-----|-----------|--------|----------|-----|
| 🥇 1 | **Naive Bayes** | **0.8644** | **0.9336** | **0.8571** | 0.9091 | **0.8824** | **0.7244** |
| 🥈 2 | **kNN** | 0.8475 | 0.9260 | 0.8529 | 0.8788 | 0.8657 | 0.6897 |
| 🥉 3 | **Random Forest (Ensemble)** | 0.8305 | 0.9307 | 0.7949 | **0.9394** | 0.8611 | 0.6625 |
| 4 | Logistic Regression | 0.7966 | 0.9021 | 0.7838 | 0.8788 | 0.8286 | 0.5863 |
| 5 | XGBoost (Ensemble) | 0.7797 | 0.8695 | 0.7632 | 0.8788 | 0.8169 | 0.5523 |
| 6 | Decision Tree | 0.6949 | 0.6865 | 0.7143 | 0.7576 | 0.7353 | 0.3769 |

### Comprehensive Analysis

#### 🏆 Overall Performance Winner: **Naive Bayes**
- **Highest Accuracy**: 86.44% (best overall correctness)
- **Highest AUC**: 0.9336 (best ability to distinguish between classes)
- **Highest Precision**: 0.8571 (fewest false positives)
- **Highest F1-Score**: 0.8824 (best balance of precision and recall)
- **Highest MCC**: 0.7244 (most reliable binary classification)
- **Fastest Training**: Ideal for real-time applications

#### 🥈 Strong Performer: **kNN (k-Nearest Neighbors)**
- **Second-Best Accuracy**: 84.75%
- **High AUC**: 0.9260 (excellent class discrimination)
- **Well-Balanced**: Consistent across all metrics
- **Non-parametric**: Flexible for various data distributions

#### 🥉 High Recall: **Random Forest (Ensemble)**
- **Highest Recall**: 0.9394 (catches 93.94% of disease cases - critical for healthcare)
- **Excellent AUC**: 0.9307 (second-highest after Naive Bayes)
- **Ensemble Strength**: Robust predictions through voting
- **Use Case**: When missing disease cases is most critical

#### 📊 Model-by-Model Insights

**1. Naive Bayes - Best Overall** 🏆
- ✅ Winner across 5 out of 6 metrics
- ✅ Highest AUC (0.9336) - best class separation
- ✅ Highest precision - fewest false positives
- ✅ Extremely fast training and prediction
- ✅ Good probabilistic interpretability
- **Use Case**: Production deployment, real-time screening

**2. kNN - Consistent Performer**
- ✅ Second-best overall performance
- ✅ Balanced performance across all metrics  
- ✅ Non-parametric, flexible
- ⚠️ Sensitive to k value and distance metric
- ⚠️ Slower with large datasets
- **Use Case**: Baseline comparison, exploratory analysis

**3. Random Forest (Ensemble) - High Recall**
- ✅ Highest recall (93.94%) - catches most disease cases
- ✅ High AUC (0.9307)
- ✅ Robust to overfitting through ensemble
- ⚠️ Lower precision (more false positives)
- **Use Case**: When missing disease cases must be minimized

**4. Logistic Regression - Good Baseline**
- ✅ Fast training and prediction
- ✅ Interpretable coefficients
- ✅ Good AUC (0.9021)
- ⚠️ Lower accuracy (79.66%)
- **Use Case**: Baseline model, explainability required

**5. XGBoost (Ensemble) - Needs Tuning**
- ✅ Good recall (0.8788)
- ⚠️ Lower accuracy (77.97%) than expected for gradient boosting
- ⚠️ Lower AUC (0.8695)
- **Improvement Needed**: Hyperparameter tuning (learning rate, max_depth, subsample)
- **Potential**: With proper tuning, could outperform other models
- **Use Case**: After hyperparameter optimization

**6. Decision Tree - Needs Improvement**
- ⚠️ Lowest performance across all metrics
- ⚠️ Likely overfitting to training data
- ⚠️ Low MCC (0.3769) indicates poor reliability
- **Improvement Needed**: Pruning, max_depth tuning
- **Use Case**: Educational visualization only

### Key Findings

1. 🎯 **Best Model**: Naive Bayes is the clear winner (86.44% accuracy, 0.9336 AUC, 0.8824 F1)
2. 📈 **All Models Have Good AUC**: Even XGBoost achieved 0.8695 AUC (>0.85 threshold)
3. 🔍 **High Recall Priority**: Random Forest catches 93.94% of disease cases
4. ⚖️ **Precision-Recall Tradeoff**: Naive Bayes has highest precision (0.8571)
5. ✅ **Strong MCC Values**: Top 3 models show MCC > 0.66 (reliable classification)
6. ⚠️ **Decision Tree Underperforms**: Needs hyperparameter tuning
7. 🔧 **XGBoost Potential**: Currently underperforming but has room for improvement with tuning

### Clinical Significance

For **heart disease prediction**, the choice of model depends on priorities:

- **Best Overall Performance**: Use **Naive Bayes** (highest accuracy, AUC, precision, F1, MCC)
- **Minimize False Negatives** (missing disease cases): Use **Random Forest** (recall: 93.94%)
- **Balance Speed & Accuracy**: Use **Naive Bayes** (fastest with best performance)
- **Ensemble Approach**: Combine Naive Bayes + Random Forest for maximum reliability

---

## 🌐 Web Application

### Features

1. **Interactive Input Form**
   - 13 clinical parameter inputs
   - User-friendly sliders and dropdowns
   - Real-time validation

2. **Model Selection**
   - Choose from 6 trained models
   - View model descriptions
   - Compare model characteristics

3. **Prediction Display**
   - Clear risk assessment
   - Confidence scores
   - Visual indicators
   - Health recommendations

4. **Multi-Model Comparison**
   - See predictions from all models
   - Compare results side-by-side
   - Identify consensus

### Screenshot / Interface

```
❤️ Heart Disease Prediction System
══════════════════════════════════════════════════════════

Sidebar:
  📋 Patient Information
  ├── Age: [slider]
  ├── Sex: [dropdown]
  ├── Chest Pain Type: [dropdown]
  └── ... (13 features)
  
  🤖 Model Selection
  └── Choose Model: [dropdown]
  
  [🔍 Predict] Button

Main Area:
  📊 Patient Data Summary (table)
  
  🎯 Prediction Results
  ├── Risk Assessment (color-coded)
  ├── Confidence Score
  └── Recommendation
  
  📈 Compare with All Models (table)
```

---

## 🔧 Technologies Used

### Programming & Libraries
- **Python 3.13** - Core programming language
- **scikit-learn 1.3.2** - Machine learning algorithms
- **pandas 2.1.4** - Data manipulation
- **numpy 1.26.2** - Numerical computations
- **matplotlib 3.8.2** - Visualization
- **seaborn 0.13.0** - Statistical visualization

### Web Framework
- **Streamlit 1.30.0** - Interactive web application

### Tools & Utilities
- **pickle** - Model serialization
- **StandardScaler** - Feature normalization
- **train_test_split** - Data splitting
- **classification_report** - Model evaluation

### Development
- **VS Code** - IDE
- **Git** - Version control
- **Python virtual environment** - Dependency management

---

## 📝 Conclusion

### Summary
This project successfully implements a complete Machine Learning pipeline for heart disease classification, demonstrating:

1. ✅ **Data Preprocessing**: Proper handling of medical data with standardization
2. ✅ **Multiple Algorithms**: Comparative analysis of 6 ML models with comprehensive metrics
3. ✅ **Best Performance**: Naive Bayes achieved best overall (Accuracy: 86.44%, AUC: 0.9336, F1: 0.8824, MCC: 0.7244)
4. ✅ **Robust Evaluation**: 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC) for thorough assessment
5. ✅ **Deployment**: Interactive web application for real-world use
6. ✅ **Documentation**: Comprehensive project documentation with detailed analysis

### Key Achievements
- ✅ Implemented 6 classification algorithms (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost)
- ✅ Achieved 86.44% accuracy with Naive Bayes (best overall performer)
- ✅ Excellent AUC scores: Naive Bayes (0.9336), Random Forest (0.9307), kNN (0.9260)
- ✅ High recall (93.94%) with Random Forest - critical for not missing disease cases
- ✅ Strong MCC values (>0.66) for top 3 models - reliable classification
- ✅ Created user-friendly Streamlit web interface
- ✅ Modular and reusable code structure
- ✅ Complete MLOps pipeline with model persistence
- ✅ Comprehensive evaluation metrics for informed model selection

### Model Selection Recommendation

**For Production Deployment:**
- **Primary Model**: Naive Bayes (best accuracy, AUC, precision, F1, and MCC + fastest)
- **High Recall Requirement**: Random Forest (Ensemble) - catches 93.94% of disease cases
- **Ensemble Approach**: Combine Naive Bayes + Random Forest predictions for maximum reliability
- **XGBoost Optimization**: With proper hyperparameter tuning, XGBoost could be competitive

### Future Improvements
1. 🔄 **Hyperparameter Tuning**: GridSearchCV/RandomizedSearchCV for optimization
2. 📊 **Feature Engineering**: Create interaction terms, polynomial features
3. 🧬 **Feature Selection**: Use SelectKBest, RFE to identify most important features
4. 🎯 **Class Balancing**: SMOTE/ADASYN if imbalance exists
5. 🧪 **Cross-Validation**: K-fold CV for more robust performance estimates
6. 📈 **Advanced Visualization**: ROC curves, Precision-Recall curves, confusion matrices
7. 🤖 **Advanced Models**: XGBoost, LightGBM, Neural Networks
8. 🌐 **Cloud Deployment**: Streamlit Cloud, AWS, Azure, or Heroku
9. 🔒 **Security**: Add user authentication and data encryption
10. 💾 **Database Integration**: PostgreSQL/MongoDB for prediction history tracking
11. 📱 **Mobile App**: React Native or Flutter mobile application
12. 🔔 **Alert System**: Email/SMS notifications for high-risk predictions

### Learning Outcomes
- ✅ Understanding of 6 different ML algorithms and their characteristics
- ✅ Experience with end-to-end ML pipeline (data → training → deployment)
- ✅ Web application development using Streamlit
- ✅ Comprehensive model evaluation using 6 different metrics
- ✅ Model comparison and selection based on multiple criteria
- ✅ Best practices in ML project structure and documentation
- ✅ Handling medical/healthcare data responsibly
- ✅ Understanding precision-recall tradeoffs in classification

---

## ⚠️ Disclaimer

This application is for **educational and demonstration purposes only**. It is an academic assignment and should **NOT** be used for actual medical diagnosis or healthcare decisions. 

⚕️ **Always consult with qualified healthcare professionals for medical advice, diagnosis, and treatment.**

---

## 📧 Contact & Support

For questions about this assignment or project:
- Review the code documentation
- Check the inline comments in Python files
- Refer to scikit-learn documentation
- Contact course instructor

---

## 📄 License

This project is created for educational purposes as part of an ML Assignment.

---

**Made with ❤️ for Machine Learning Education**

**Assignment Submitted**: February 2026
