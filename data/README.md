# Wine Quality Dataset

This directory contains the wine quality dataset (`wine_quality.csv`).

## Dataset Information

The dataset combines red and white wine samples with the following structure:

### Features (12 chemical attributes):
1. **fixed_acidity**: Fixed acidity (g/dm³) - tartaric acid content
2. **volatile_acidity**: Volatile acidity (g/dm³) - acetic acid content (vinegar-like)
3. **citric_acid**: Citric acid (g/dm³) - freshness and flavor
4. **residual_sugar**: Residual sugar (g/dm³) - sweetness after fermentation
5. **chlorides**: Chlorides (g/dm³) - salt content
6. **free_sulfur_dioxide**: Free SO₂ (mg/dm³) - prevents microbial growth and oxidation
7. **total_sulfur_dioxide**: Total SO₂ (mg/dm³) - free + bound forms
8. **density**: Density (g/cm³) - depends on alcohol and sugar content
9. **pH**: pH level (0-14 scale) - acidity measure
10. **sulphates**: Sulphates (g/dm³) - wine additive (antimicrobial and antioxidant)
11. **alcohol**: Alcohol content (% by volume)
12. **wine_type**: Wine type (Red = 0; White = 1)

### Target Variable:
- **quality_binary**: Wine quality classification
  - 1 = Good Quality (original quality score ≥ 6)
  - 0 = Below Average (original quality score < 6)

## Dataset Characteristics

- **Total Samples**: 6,497 wines
  - Red wines: 1,599 samples
  - White wines: 4,898 samples
- **Features**: 12 chemical properties
- **Classes**: Binary (0 = Below Average, 1 = Good Quality)
- **Quality Distribution**:
  - Good Quality: 4,113 samples (63.3%)
  - Below Average: 2,384 samples (36.7%)
- **No Missing Values**

## Dataset Sources

The wine quality dataset is obtained from:

1. **UCI Machine Learning Repository**:
   - [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - Red Wine: `winequality-red.csv`
   - White Wine: `winequality-white.csv`

2. **Related Studies**:
   - P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
   - "Modeling wine preferences by data mining from physicochemical properties."
   - Decision Support Systems, Elsevier, 47(4):547-553, 2009.

## Data Preparation

The `prepare_wine_data.py` script combines red and white wine datasets:
```bash
python data/prepare_wine_data.py
```

This creates `wine_quality.csv` by:
1. Loading red and white wine CSVs
2. Adding a `wine_type` column (Red=0, White=1)
3. Combining into a single dataset
4. Saving to `wine_quality.csv`

## Usage

With the `wine_quality.csv` file in this directory, you can:
1. Train the models: `python train_models.py`
2. Run the web app: `streamlit run app.py`
