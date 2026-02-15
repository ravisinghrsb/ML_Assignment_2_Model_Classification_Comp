# Heart Disease Dataset

This directory contains the heart disease dataset (`heart.csv`).

## Dataset Information

The dataset has the following structure:

### Features (13 attributes):
1. **age**: Age in years
2. **sex**: Sex (1 = male; 0 = female)
3. **cp**: Chest pain type
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps**: Resting blood pressure (in mm Hg on admission to the hospital)
5. **chol**: Serum cholesterol in mg/dl
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **restecg**: Resting electrocardiographic results
   - 0: Normal
   - 1: Having ST-T wave abnormality
   - 2: Showing probable or definite left ventricular hypertrophy
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes; 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: The slope of the peak exercise ST segment
    - 0: Upsloping
    - 1: Flat
    - 2: Downsloping
12. **ca**: Number of major vessels (0-3) colored by fluoroscopy
13. **thal**: Thalassemia
    - 1: Normal
    - 2: Fixed defect
    - 3: Reversible defect

### Target Variable:
- **target**: Diagnosis of heart disease (1 = presence; 0 = absence)

## Dataset Characteristics

- **Total Samples**: 303 patients
- **Features**: 13 clinical features + 1 target
- **Classes**: Binary (0 = no disease, 1 = disease)
- **No Missing Values**

## Dataset Sources

You can obtain the heart disease dataset from:

1. **UCI Machine Learning Repository**:
   - [Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

2. **Kaggle**:
   - [Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)

## Usage

With the `heart.csv` file in this directory, you can:
1. Train the models: `python train_models.py`
2. Run the web app: `streamlit run app.py`
