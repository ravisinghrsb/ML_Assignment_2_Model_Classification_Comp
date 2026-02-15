# Wine Quality Classification Using Machine Learning

## 1. Problem Statement

The wine industry relies heavily on quality assessment for pricing, marketing, and production decisions. Traditional wine quality evaluation depends on human expert tasters, which can be subjective, time-consuming, and expensive. This project aims to develop an automated machine learning system that predicts wine quality based on physicochemical properties.

**Objective:** Build and compare six different machine learning classification models to predict whether a wine is of "Good Quality" (rating ≥ 6) or "Below Average Quality" (rating < 6) based on 12 chemical features.

**Key Goals:**
- Implement six different classification algorithms
- Compare model performance using multiple evaluation metrics
- Deploy the best model in an interactive web application
- Provide actionable insights for wine quality prediction

## 2. Dataset Description

**Dataset Name:** Wine Quality Dataset  
**Source:** UCI Machine Learning Repository  
**Total Samples:** 6,497 wine samples
- Red wine: 1,599 samples
- White wine: 4,898 samples

**Number of Features:** 12 physicochemical properties

**Target Variable:** Binary classification
- **Good Quality:** Wine rating ≥ 6
- **Below Average:** Wine rating < 6

**Class Distribution:**
- Good Quality: 4,113 samples (63.3%)
- Below Average: 2,384 samples (36.7%)

**Dataset Split:**
- Training set: 5,197 samples (80%)
- Test set: 1,300 samples (20%)
- Stratified sampling to maintain class balance

### Features Description

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| Fixed Acidity | Tartaric acid content | g/dm³ | 3.8 - 15.9 |
| Volatile Acidity | Acetic acid content (vinegar taste) | g/dm³ | 0.08 - 1.58 |
| Citric Acid | Adds freshness and flavor | g/dm³ | 0.0 - 1.66 |
| Residual Sugar | Sweetness after fermentation | g/dm³ | 0.6 - 65.8 |
| Chlorides | Salt content | g/dm³ | 0.009 - 0.611 |
| Free Sulfur Dioxide | Prevents microbial growth | mg/dm³ | 1 - 289 |
| Total Sulfur Dioxide | Total SO₂ content | mg/dm³ | 6 - 440 |
| Density | Wine density | g/cm³ | 0.987 - 1.039 |
| pH | Acidity level | - | 2.72 - 4.01 |
| Sulphates | Wine additive (potassium sulphate) | g/dm³ | 0.22 - 2.0 |
| Alcohol | Alcohol percentage | % vol | 8.0 - 14.9 |
| Wine Type | Red (1) or White (0) | Binary | 0 or 1 |

## 3. Models Used

Six machine learning classification algorithms were implemented and evaluated:

### Model Comparison Table - Evaluation Metrics

Results on test set (1,300 samples):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.7392 | 0.8057 | 0.7665 | 0.8457 | 0.8042 | 0.4214 |
| Decision Tree | 0.7692 | 0.7530 | 0.8201 | 0.8141 | 0.8171 | 0.5046 |
| kNN | 0.7408 | 0.8004 | 0.7780 | 0.8262 | 0.8014 | 0.4308 |
| Naive Bayes | 0.6815 | 0.7419 | 0.7216 | 0.8092 | 0.7629 | 0.2873 |
| Random Forest (Ensemble) | 0.8338 | 0.9048 | 0.8517 | 0.8931 | 0.8719 | 0.6374 |
| XGBoost (Ensemble) | 0.8169 | 0.8782 | 0.8421 | 0.8748 | 0.8582 | 0.6012 |

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieves moderate accuracy (73.92%) with good recall (84.57%), making it suitable as a baseline model. The linear decision boundary limits its ability to capture complex patterns in wine chemistry. Strong interpretability through coefficient analysis. Best used for understanding linear relationships between chemical properties and quality. |
| Decision Tree | Delivers decent accuracy (76.92%) with high precision (82.01%), resulting in fewer false positives. The model is highly interpretable with clear decision rules based on chemical thresholds. However, prone to overfitting on training data. Good balance between performance and explainability, suitable when understanding decision logic is important. |
| kNN | Shows moderate performance (74.08% accuracy) with good recall (82.62%). As an instance-based learner, it makes no assumptions about data distribution. Performance is sensitive to the choice of k value and feature scaling. Slower prediction time compared to other models due to distance calculations. Works well for wines with similar chemical profiles in the training set. |
| Naive Bayes | Lowest overall performance (68.15% accuracy, 0.2873 MCC) among all models. The assumption of feature independence is violated in wine chemistry where features are correlated (e.g., acidity measures). Fast training and prediction but poor reliability. Not recommended for production use in this application. Useful only for quick baseline comparisons. |
| Random Forest (Ensemble) | **Best overall performer** with highest accuracy (83.38%), AUC (0.9048), recall (89.31%), F1-score (0.8719), and MCC (0.6374). The ensemble approach combines multiple decision trees to reduce overfitting and improve generalization. Excellent at capturing complex non-linear relationships between chemical properties. Robust to outliers and handles feature interactions well. **Recommended for production deployment**. |
| XGBoost (Ensemble) | **Second-best performer** with strong accuracy (81.69%) and highest precision (84.21%), minimizing false positives. Gradient boosting sequentially corrects errors from previous trees. Excellent AUC score (0.8782) indicates strong discriminative ability. More complex to tune than Random Forest but offers better precision when avoiding false positives is critical. **Alternative choice for production** when precision is prioritized over recall. |

### Key Findings

1. **Ensemble Methods Dominate:** Random Forest and XGBoost outperform all individual classifiers by 5-15%, demonstrating the power of ensemble learning.

2. **Accuracy Range:** Model accuracy ranges from 68.15% (Naive Bayes) to 83.38% (Random Forest), showing significant variation in performance.

3. **Best Discriminative Power:** Random Forest achieves the highest AUC (0.9048), indicating superior ability to distinguish between good and below-average wines.

4. **Precision-Recall Trade-off:** 
   - XGBoost prioritizes precision (84.21%) - fewer false positives
   - Random Forest prioritizes recall (89.31%) - catches more good wines

5. **Model Reliability:** MCC scores show Random Forest (0.6374) and XGBoost (0.6012) have the most reliable predictions, while Naive Bayes (0.2873) is least reliable.

6. **Feature Complexity:** Linear models (Logistic Regression) struggle compared to tree-based models, suggesting non-linear relationships in wine chemistry.

---

## Project Structure

```
mlassignment/
│
├── app.py                     # Streamlit web application (enhanced with 5 tabs)
├── train_models.py           # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── data/                     # Dataset directory
│   ├── wine_quality.csv     # Combined wine quality dataset (6,497 samples)
│   ├── winequality-red.csv  # Red wine data (1,599 samples)
│   ├── winequality-white.csv # White wine data (4,898 samples)
│   └── prepare_wine_data.py # Dataset preparation script
│
├── models/                   # Saved trained models
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest_(ensemble).pkl
│   ├── xgboost_(ensemble).pkl
│   ├── scaler.pkl           # StandardScaler for feature normalization
└── model_metrics.csv    # Model evaluation metrics
```

## Implementation Pipeline

The project follows a complete machine learning workflow:

1. **Data Loading:** Load wine quality dataset and convert quality scores to binary labels
2. **Preprocessing:** Split data (80/20), apply StandardScaler for feature normalization
3. **Training:** Train all six models with random seed (42) for reproducibility
4. **Evaluation:** Test models on held-out test set, calculate 6 performance metrics
5. **Model Persistence:** Save trained models and scaler using pickle
6. **Deployment:** Build interactive Streamlit web application with 5 feature tabs

---

## Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. Clone or download the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- streamlit>=1.30.0
- scikit-learn>=1.5.0
- pandas>=2.1.0
- numpy>=1.26.0
- matplotlib>=3.8.0
- seaborn>=0.13.0
- xgboost>=3.2.0
- plotly>=5.18.0

---

## Usage

### Training Models

Run the training script to train all six models:

```bash
python train_models.py
```

This will:
- Load the wine quality dataset
- Preprocess the data
- Train all six classification models
- Evaluate and save the models
- Generate model_metrics.csv with performance results

### Running the Web Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The application will open in your browser at https://ravi-bhadauria-mlassignment2-modelclassificationcomp.streamlit.app/

### Web App Features

The Streamlit application provides 5 comprehensive tabs:

1. **Model Performance Tab**
   - Complete metrics table for all six models
   - Interactive comparison charts
   - Download metrics as CSV

2. **Predictions Tab**
   - Input wine properties using sliders
   - Get predictions from all models simultaneously
   - Consensus prediction with confidence scores
   - Detailed model analysis with probability gauges

3. **Visualizations Tab**
   - Radar charts comparing model metrics
   - Interactive Plotly charts
   - Accuracy and F1-score comparison graphs

4. **Test Your Data Tab**
   - Upload CSV files for batch predictions
   - Download sample template
   - Model agreement analysis with distribution charts
   - Export prediction results

5. **About Tab**
   - Dataset information and statistics
   - Download complete dataset
   - Dataset preview

---

## Technologies Used

- **Python 3.13** - Programming language
- **scikit-learn** - Machine learning library (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest)
- **XGBoost** - Gradient boosting framework
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive visualizations

---

## Results Summary

**Best Performing Model:** Random Forest (Ensemble)
- **Accuracy:** 83.38%
- **AUC:** 0.9048
- **F1-Score:** 0.8719
- **MCC:** 0.6374

**Key Insights:**
- Ensemble methods (Random Forest, XGBoost) significantly outperform individual classifiers
- Random Forest achieves 89.31% recall, successfully identifying most good quality wines
- XGBoost offers highest precision (84.21%), minimizing false positives
- The 6,497 sample dataset provides sufficient data for robust model training
- 12 chemical properties are effective predictors of wine quality

---

## Future Enhancements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Feature engineering to create interaction terms between chemical properties
- K-fold cross-validation for more robust performance estimates
- SHAP values for model interpretability and feature importance analysis
- Deep learning models (Neural Networks) for complex pattern recognition
- REST API development for production deployment
- Cloud deployment on AWS, Azure, or Heroku
- Mobile application for real-time quality assessment

---

## License

This project was created for educational purposes as part of a Machine Learning assignment.

---

*For questions or issues, refer to the code documentation and inline comments in the Python files.*
