"""
Train all 6 classification models for heart disease prediction
Models: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, Naive Bayes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef)
import pickle
import os

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_data(filepath='data/heart.csv'):
    """Load the heart disease dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the data: split features and target, scale features"""
    print("\nPreprocessing data...")
    
    # Assuming 'target' is the column name for the disease indicator
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and evaluate their performance"""
    
    # Define all models
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        'XGBoost (Ensemble)': XGBClassifier(random_state=RANDOM_STATE, n_estimators=100, eval_metric='logloss')
    }
    
    trained_models = {}
    results = {}
    
    print("\n" + "="*80)
    print("Training and Evaluating Models")
    print("="*80)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions for AUC calculation
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred  # For models without predict_proba
        
        # Calculate all metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = accuracy  # Fallback if AUC can't be calculated
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"AUC:       {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"MCC:       {mcc:.4f}")
        
        # Store results
        trained_models[name] = model
        results[name] = {
            'Accuracy': accuracy,
            'AUC': auc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'MCC': mcc
        }
    
    return trained_models, results

def save_models(models, scaler):
    """Save trained models and scaler to disk"""
    print("\n" + "="*80)
    print("Saving Models")
    print("="*80)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save each model
    for name, model in models.items():
        filename = f"models/{name.lower().replace(' ', '_')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved: {filename}")
    
    # Save scaler
    scaler_file = 'models/scaler.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved: {scaler_file}")

def main():
    """Main function to execute the training pipeline"""
    print("="*80)
    print("Heart Disease Classification - Model Training Pipeline")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train and evaluate models
    trained_models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Save models
    save_models(trained_models, scaler)
    
    # Print summary table
    print("\n" + "="*80)
    print("Model Comparison Table - All Evaluation Metrics")
    print("="*80)
    print(f"\n{'ML Model Name':<25} {'Accuracy':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MCC':>10}")
    print("-" * 95)
    
    # Sort by accuracy
    for name in sorted(results.keys(), key=lambda x: results[x]['Accuracy'], reverse=True):
        metrics = results[name]
        print(f"{name:<25} {metrics['Accuracy']:>10.4f} {metrics['AUC']:>10.4f} "
              f"{metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} "
              f"{metrics['F1']:>10.4f} {metrics['MCC']:>10.4f}")
    
    print("\n✓ Training completed successfully!")
    print("✓ All models and scaler saved to 'models/' directory")
    
    # Save metrics to CSV for README
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv('models/model_metrics.csv')
    print("✓ Metrics saved to 'models/model_metrics.csv'")

if __name__ == "__main__":
    main()
