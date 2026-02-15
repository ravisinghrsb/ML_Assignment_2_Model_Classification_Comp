"""
Prepare Wine Quality Dataset for ML Assignment
Combines red and white wine datasets into a single dataset
"""

import pandas as pd

# Read red and white wine datasets
red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

# Add wine type column
red_wine['wine_type'] = 1  # 1 for red wine
white_wine['wine_type'] = 0  # 0 for white wine

# Combine datasets
wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)

# Rename columns to remove spaces and special characters
wine_data.columns = wine_data.columns.str.replace(' ', '_')

# Save combined dataset
wine_data.to_csv('wine_quality.csv', index=False)

print(f"Combined Wine Quality Dataset Created!")
print(f"Total instances: {len(wine_data)}")
print(f"Total features: {len(wine_data.columns) - 1}")  # excluding target
print(f"\nDataset shape: {wine_data.shape}")
print(f"\nFeatures: {list(wine_data.columns)}")
print(f"\nQuality distribution:\n{wine_data['quality'].value_counts().sort_index()}")
print(f"\nWine type distribution:\n{wine_data['wine_type'].value_counts()}")
