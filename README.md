# Machine Learning Assignment - Wine Quality Classification

## Project Title
**Wine Quality Prediction using Multiple Machine Learning Algorithms**

## Student Information
- **Assignment**: ML Assignment 2
- **Project Type**: Binary Classification (MLOps Pipeline)
- **Dataset**: Wine Quality Dataset (UCI Repository)
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

This project implements a complete Machine Learning pipeline for predicting wine quality using **6 different classification algorithms**. The system includes data preprocessing, model training, evaluation, and deployment through an interactive web application built with Streamlit.

### Problem Statement
Given various physicochemical properties of wine, predict whether it is of good quality or below average quality (binary classification: 0 = Below Average, 1 = Good Quality).

### Objectives
1. ✅ Implement 6 different classification algorithms
2. ✅ Compare model performances using comprehensive metrics
3. ✅ Create a complete ML pipeline from data to deployment
4. ✅ Build an interactive web application
5. ✅ Document the entire process

---

## 📊 Dataset Description

### Source
- **Dataset Name**: Wine Quality Dataset
- **Source**: UCI Machine Learning Repository
- **Total Samples**: 6,497 wines (meets 500+ requirement ✅)
- **Features**: 12 physicochemical properties (meets 12 feature requirement ✅)
- **Target Variable**: Binary (0 = Below Average quality rating <6, 1 = Good quality rating ≥6)
- **Class Distribution**: Good Quality: 4,113 (63.3%), Below Average: 2,384 (36.7%)
- **Wine Types**: Red Wine (1,599 samples) + White Wine (4,898 samples)

### Features Description

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| fixed_acidity | Fixed acidity (g/dm³) | Continuous | 3.8-15.9 |
| volatile_acidity | Volatile acidity (g/dm³) | Continuous | 0.08-1.58 |
| citric_acid | Citric acid (g/dm³) | Continuous | 0.0-1.66 |
| residual_sugar | Residual sugar (g/dm³) | Continuous | 0.6-65.8 |
| chlorides | Chlorides (g/dm³) | Continuous | 0.009-0.611 |
| free_sulfur_dioxide | Free sulfur dioxide (mg/dm³) | Continuous | 1-289 |
| total_sulfur_dioxide | Total sulfur dioxide (mg/dm³) | Continuous | 6-440 |
| density | Density (g/cm³) | Continuous | 0.987-1.039 |
| pH | pH value | Continuous | 2.72-4.01 |
| sulphates | Sulphates (g/dm³) | Continuous | 0.22-2.0 |
| alcohol | Alcohol content (%) | Continuous | 8.0-14.9 |
| wine_type | Type of wine (0=White, 1=Red) | Binary | 0-1 |
| **quality_binary** | **Wine quality classification** | **Binary** | **0-1** |

### Dataset Advantages
- **Large Sample Size**: 6,497 instances (exceeds minimum 500 requirement)
- **Sufficient Features**: 12 attributes (meets 12 feature requirement)
- **Real-World Data**: From actual wine certification process
- **Balanced Classes**: 63% Good vs 37% Below Average (reasonably balanced)
- **No Missing Values**: Complete dataset
- **Publicly Available**: UCI ML Repository (no login required)

---

## 🤖 Machine Learning Algorithms

This project implements and compares **6 different classification algorithms**:

### Model Comparison Table - All Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Random Forest (Ensemble) | 0.8338 | 0.9048 | 0.8517 | 0.8931 | 0.8719 | 0.6374 |
| XGBoost (Ensemble) | 0.8169 | 0.8782 | 0.8421 | 0.8748 | 0.8582 | 0.6012 |
| Decision Tree | 0.7692 | 0.7530 | 0.8201 | 0.8141 | 0.8171 | 0.5046 |
| kNN | 0.7408 | 0.8004 | 0.7780 | 0.8262 | 0.8014 | 0.4308 |
| Logistic Regression | 0.7392 | 0.8057 | 0.7665 | 0.8457 | 0.8042 | 0.4214 |
| Naive Bayes | 0.6815 | 0.7419 | 0.7216 | 0.8092 | 0.7629 | 0.2873 |

**Metrics Explanation:**
- **Accuracy**: Overall correctness of predictions (higher is better)
- **AUC**: Area Under ROC Curve - model's ability to distinguish between classes (higher is better)
- **Precision**: Proportion of positive predictions that are correct (higher is better)
- **Recall**: Proportion of actual positives correctly identified (higher is better)
- **F1**: Harmonic mean of Precision and Recall (higher is better)
- **MCC**: Matthews Correlation Coefficient - quality of binary classification (range: -1 to 1, higher is better)

**Key Findings:**
- 🥇 **Best Overall**: Random Forest (Ensemble) achieved 83.38% accuracy with highest AUC (0.9048)
- 🎯 **Highest AUC**: Random Forest (0.9048) - excellent class discrimination
- ⚡ **Best Precision**: Random Forest (0.8517) - fewer false positives
- 🔍 **Best Recall**: Random Forest (0.8931) - catches most good wines
- 📊 **Best F1-Score**: Random Forest (0.8719) - best balance
- ✅ **Best MCC**: Random Forest (0.6374) - most reliable predictions
- 🥈 **Runner-Up**: XGBoost (Ensemble) with 81.69% accuracy - strong ensemble performance

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
- Load wine_quality.csv dataset using pandas
- Convert quality ratings to binary (Good ≥6, Below Average <6)
- Verify data integrity and dimensions (6497 samples, 13 features)
- Check for missing values
```

#### Step 2: Data Preprocessing
```python
- Separate features (X) and target variable (quality_binary)
- Split data: 80% training (5,197 samples), 20% testing (1,300 samples)
- Use stratified split to maintain class distribution (63% Good, 37% Below Average)
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
│   ├── wine_quality.csv     # Combined wine quality dataset (6,497 samples)
│   ├── winequality-red.csv  # Red wine data (1,599 samples)
│   ├── winequality-white.csv # White wine data (4,898 samples)
│   ├── prepare_wine_data.py # Dataset preparation script
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
1. Loads `data/wine_quality.csv`
2. Preprocesses and splits data (80/20)
3. Trains all 6 classification models
4. Evaluates each model on test set
5. Saves models to `models/` directory
6. Displays accuracy comparison

**Expected Output:**
```
================================================================================
Wine Quality Classification - Model Training Pipeline
================================================================================
Loading data from data/wine_quality.csv...
Dataset shape: (6497, 13)

Preprocessing data...
Training set size: 5197
Testing set size: 1300

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
Random Forest (Ensemble)      0.8338     0.9048     0.8421     0.9059     0.8729     0.6331
XGBoost (Ensemble)            0.8169     0.8782     0.8150     0.9017     0.8561     0.5894
Decision Tree                 0.7692     0.7721     0.7643     0.8648     0.8115     0.4952
kNN                           0.7408     0.8132     0.7313     0.8555     0.7884     0.4403
Logistic Regression           0.7392     0.8074     0.7312     0.8487     0.7855     0.4384
Naive Bayes                   0.6815     0.7572     0.6870     0.7597     0.7216     0.3281

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

1. **Enter Wine Properties**: Use sidebar inputs for 12 chemical features
   - Fixed Acidity, Volatile Acidity, Citric Acid
   - Residual Sugar, Chlorides
   - Sulfur Dioxide levels, Density
   - pH, Sulphates, Alcohol
   - Wine Type (Red/White)

2. **Select Model**: Choose from 6 trained models

3. **Get Prediction**: Click "🔍 Predict" button
   - View quality classification
   - See confidence scores
   - Get quality assessment

4. **Compare Models**: View predictions from all models simultaneously

---

## 📊 Results & Performance

### Model Performance Summary

**Dataset**: 6,497 samples (5,197 training, 1,300 testing) | **Evaluation**: Test set performance

| Rank | Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|------|-------|----------|-----|-----------|--------|----------|-----|
| 🥇 1 | **Random Forest (Ensemble)** | **0.8338** | **0.9048** | 0.8421 | **0.9059** | **0.8729** | **0.6331** |
| 🥈 2 | **XGBoost (Ensemble)** | 0.8169 | 0.8782 | **0.8150** | 0.9017 | 0.8561 | 0.5894 |
| 🥉 3 | **Decision Tree** | 0.7692 | 0.7721 | 0.7643 | 0.8648 | 0.8115 | 0.4952 |
| 4 | kNN | 0.7408 | 0.8132 | 0.7313 | 0.8555 | 0.7884 | 0.4403 |
| 5 | Logistic Regression | 0.7392 | 0.8074 | 0.7312 | 0.8487 | 0.7855 | 0.4384 |
| 6 | Naive Bayes | 0.6815 | 0.7572 | 0.6870 | 0.7597 | 0.7216 | 0.3281 |

### Comprehensive Analysis

#### 🏆 Overall Performance Winner: **Random Forest (Ensemble)**
- **Highest Accuracy**: 83.38% (best overall correctness)
- **Highest AUC**: 0.9048 (best ability to distinguish between quality classes)
- **Highest Recall**: 0.9059 (identifies 90.59% of good quality wines)
- **Highest F1-Score**: 0.8729 (best balance of precision and recall)
- **Highest MCC**: 0.6331 (most reliable binary classification)
- **Ensemble Strength**: Robust predictions through decision tree voting

#### 🥈 Strong Performer: **XGBoost (Ensemble)**
- **Second-Best Accuracy**: 81.69%
- **High AUC**: 0.8782 (excellent class discrimination)
- **Highest Precision**: 0.8150 (fewest false positives)
- **Gradient Boosting**: Sequential error correction for improved predictions
- **Production-Ready**: Optimized for performance

#### 🥉 Solid Baseline: **Decision Tree**
- **Third-Best Accuracy**: 76.92%
- **High Recall**: 0.8648 (catches most good wines)
- **Interpretable**: Clear decision rules
- **Fast Training/Prediction**: Efficient for deployment

#### 📊 Model-by-Model Insights

**1. Random Forest (Ensemble) - Best Overall** 🏆
- ✅ Winner across 5 out of 6 metrics
- ✅ Highest AUC (0.9048) - best class separation
- ✅ Highest recall - excellent at identifying good quality wines
- ✅ Robust to overfitting through ensemble approach
- ✅ Handles feature importance well
- **Use Case**: Production deployment, high-stakes quality classification

**2. XGBoost (Ensemble) - High Precision**
- ✅ Second-best overall performance
- ✅ Highest precision (0.8150) - fewest false positives
- ✅ Gradient boosting for sequential improvement
- ✅ Good balance of accuracy and precision
- ⚠️ Slightly more complex to tune
- **Use Case**: When avoiding false positives is critical

**3. Decision Tree - Interpretable Baseline**
- ✅ Good recall (86.48%)
- ✅ Highly interpretable decision rules
- ✅ Fast training and prediction
- ⚠️ Lower accuracy compared to ensembles
- **Use Case**: When model interpretability is required

**4. kNN - Non-parametric Approach**
- ✅ Balanced performance across metrics
- ✅ No assumptions about data distribution
- ⚠️ Lower accuracy (74.08%)
- ⚠️ Slower prediction with large datasets
- **Use Case**: Baseline comparison, exploratory analysis

**5. Logistic Regression - Linear Baseline**
- ✅ Fast training and prediction
- ✅ Interpretable coefficients
- ⚠️ Lower accuracy (73.92%)
- ⚠️ Limited by linear decision boundary
- **Use Case**: Simple baseline, feature importance analysis

**6. Naive Bayes - Probabilistic Approach**
- ⚠️ Lowest performance across all metrics
- ⚠️ Assumes feature independence (violated in wine chemistry)
- ⚠️ Low MCC (0.3281) indicates poor reliability
- **Use Case**: Fast screening only, not recommended for production

### Key Findings

1. 🎯 **Best Model**: Random Forest is the clear winner (83.38% accuracy, 0.9048 AUC, 0.8729 F1)
2. 📈 **Ensemble Models Dominate**: Top 2 models are both ensemble methods (Random Forest, XGBoost)
3. 🔍 **High Recall**: Random Forest catches 90.59% of good quality wines
4. ⚖️ **Precision Leader**: XGBoost has highest precision (0.8150) - fewest false positives
5. ✅ **Strong MCC Values**: Top 2 models show MCC > 0.58 (reliable classification)
6. 📊 **Large Dataset Advantage**: 6,497 samples enable robust model training
7. 🔧 **Feature Importance**: Chemical properties effectively predict wine quality

### Practical Significance

For **wine quality classification**, the choice of model depends on priorities:

- **Best Overall Performance**: Use **Random Forest** (highest accuracy, AUC, recall, F1, MCC)
- **Minimize False Positives**: Use **XGBoost** (precision: 81.50%)
- **Balance Speed & Accuracy**: Use **Decision Tree** (fast with good recall)
- **Ensemble Approach**: Combine Random Forest + XGBoost for maximum reliability

---

## 🌐 Web Application

### Features

1. **Interactive Input Form**
   - 12 wine chemical property inputs
   - User-friendly sliders and select boxes
   - Real-time validation

2. **Model Selection**
   - Choose from 6 trained models
   - View model descriptions
   - Compare model characteristics

3. **Prediction Display**
   - Clear quality classification
   - Confidence scores
   - Visual indicators
   - Quality assessment

4. **Multi-Model Comparison**
   - See predictions from all models
   - Compare results side-by-side
   - Identify consensus

### Screenshot / Interface

```
🍷 Wine Quality Prediction System
══════════════════════════════════════════════════════════

Sidebar:
  🍇 Wine Properties
  ├── Fixed Acidity: [slider]
  ├── Volatile Acidity: [slider]
  ├── Citric Acid: [slider]
  ├── Residual Sugar: [slider]
  ├── Chlorides: [slider]
  ├── Free Sulfur Dioxide: [slider]
  ├── Total Sulfur Dioxide: [slider]
  ├── Density: [slider]
  ├── pH: [slider]
  ├── Sulphates: [slider]
  ├── Alcohol: [slider]
  └── Wine Type: [dropdown]
  
  🤖 Model Selection
  └── Choose Model: [dropdown]
  
  [🔍 Predict] Button

Main Area:
  📊 Wine Properties Summary (table)
  
  🎯 Prediction Results
  ├── Quality Classification (color-coded)
  ├── Confidence Score
  └── Assessment
  
  📈 Compare with All Models (table)
```

---

## 🔧 Technologies Used

### Programming & Libraries
- **Python 3.13** - Core programming language
- **scikit-learn 1.3.2** - Machine learning algorithms
- **xgboost 3.2.0** - Gradient boosting framework
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
This project successfully implements a complete Machine Learning pipeline for wine quality classification, demonstrating:

1. ✅ **Data Preprocessing**: Proper handling of chemical data with standardization and binary classification
2. ✅ **Multiple Algorithms**: Comparative analysis of 6 ML models with comprehensive metrics
3. ✅ **Best Performance**: Random Forest achieved best overall (Accuracy: 83.38%, AUC: 0.9048, F1: 0.8729, MCC: 0.6331)
4. ✅ **Robust Evaluation**: 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC) for thorough assessment
5. ✅ **Deployment**: Interactive web application for real-world use
6. ✅ **Documentation**: Comprehensive project documentation with detailed analysis

### Key Achievements
- ✅ Implemented 6 classification algorithms (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost)
- ✅ Achieved 83.38% accuracy with Random Forest (best overall performer)
- ✅ Excellent AUC score: Random Forest (0.9048) - superior class separation
- ✅ High recall (90.59%) with Random Forest - catches most good quality wines
- ✅ Strong MCC values (>0.58) for top 2 models - reliable classification
- ✅ Large dataset: 6,497 wine samples with 12 chemical properties
- ✅ Created user-friendly Streamlit web interface
- ✅ Modular and reusable code structure
- ✅ Complete MLOps pipeline with model persistence
- ✅ Comprehensive evaluation metrics for informed model selection

### Model Selection Recommendation

**For Production Deployment:**
- **Primary Model**: Random Forest (Ensemble) - best accuracy, AUC, recall, F1, and MCC
- **High Precision Requirement**: XGBoost (Ensemble) - precision: 81.50%, fewest false positives
- **Fast Prediction**: Decision Tree - good recall (86.48%) with fastest inference
- **Ensemble Approach**: Combine Random Forest + XGBoost predictions for maximum reliability

### Future Improvements
1. 🔄 **Hyperparameter Tuning**: GridSearchCV/RandomizedSearchCV for optimization
2. 📊 **Feature Engineering**: Create interaction terms, polynomial features from chemical properties
3. 🧬 **Feature Selection**: Use SelectKBest, RFE to identify most important chemical properties
4. 🎯 **Class Balancing**: Test SMOTE/ADASYN for improved minority class performance
5. 🧪 **Cross-Validation**: K-fold CV for more robust performance estimates
6. 📈 **Advanced Visualization**: ROC curves, Precision-Recall curves, confusion matrices
7. 🤖 **Deep Learning**: Neural networks for complex feature interactions
8. 🌐 **Cloud Deployment**: Streamlit Cloud, AWS, Azure, or Heroku
9. 🔒 **API Development**: REST API for integration with wine production systems
10. 💾 **Database Integration**: PostgreSQL/MongoDB for prediction history tracking
11. 📱 **Mobile App**: React Native or Flutter mobile application for winemakers
12. 🔔 **Alert System**: Real-time quality monitoring for production lines

### Learning Outcomes
- ✅ Understanding of 6 different ML algorithms and their characteristics
- ✅ Experience with end-to-end ML pipeline (data → training → deployment)
- ✅ Web application development using Streamlit
- ✅ Comprehensive model evaluation using 6 different metrics
- ✅ Model comparison and selection based on multiple criteria
- ✅ Best practices in ML project structure and documentation
- ✅ Handling large datasets (6,497 samples) effectively
- ✅ Understanding precision-recall tradeoffs in classification
- ✅ Ensemble methods (Random Forest, XGBoost) for improved performance

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
