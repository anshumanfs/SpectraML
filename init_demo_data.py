#!/usr/bin/env python
"""
Initialize demo datasets for SpectraML to test feature engineering functionality.
This script creates synthetic data that demonstrates various feature engineering operations.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import uuid

# Settings
UPLOAD_DIR = 'uploads'
DB_PATH = 'datalab.db'
CREATE_EXPERIMENT = True
EXPERIMENT_NAME = 'Feature Engineering Demo'

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

def create_demo_dataset():
    """Create a demo dataset with various feature types suitable for feature engineering"""
    # Number of samples
    n_samples = 1000
    
    # Create random data
    np.random.seed(42)  # For reproducibility
    
    # Generate numeric features
    df = pd.DataFrame({
        'id': range(1, n_samples + 1),
        'numeric_normal': np.random.normal(0, 1, n_samples),
        'numeric_uniform': np.random.uniform(0, 100, n_samples),
        'numeric_skewed': np.random.exponential(5, n_samples),
        'integer': np.random.randint(1, 100, n_samples)
    })
    
    # Generate categorical features
    categories = ['A', 'B', 'C', 'D', 'E']
    df['category_balanced'] = np.random.choice(categories, n_samples)
    df['category_imbalanced'] = np.random.choice(categories, n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05])
    
    # Generate binary feature
    df['binary'] = np.random.choice([0, 1], n_samples)
    
    # Generate datetime feature
    start_date = datetime(2020, 1, 1)
    df['date'] = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_samples)]
    
    # Generate text feature
    words = ['machine', 'learning', 'data', 'science', 'artificial', 'intelligence', 'feature', 'engineering', 
             'model', 'algorithm', 'neural', 'network', 'deep', 'regression', 'classification']
    
    def generate_text(min_words=3, max_words=10):
        n_words = np.random.randint(min_words, max_words + 1)
        return ' '.join(np.random.choice(words, n_words))
    
    df['text'] = [generate_text() for _ in range(n_samples)]
    
    # Add some missing values to various columns (about 5%)
    for col in ['numeric_normal', 'numeric_uniform', 'category_balanced', 'text']:
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    
    # Add a few outliers to numeric columns (about 1%)
    for col in ['numeric_normal', 'numeric_uniform', 'numeric_skewed']:
        mask = np.random.random(n_samples) < 0.01
        df.loc[mask, col] = df[col].max() * 5
    
    # Generate target variables (for demo ML tasks)
    # - Regression target (based on some features with noise)
    df['target_regression'] = (
        2 * df['numeric_normal'] - 
        0.5 * df['numeric_uniform'] + 
        5 * (df['category_balanced'] == 'A').astype(int) +
        np.random.normal(0, 2, n_samples)
    )
    
    # - Classification target (based on some features with noise)
    df['target_classification'] = np.random.choice(['positive', 'negative'], n_samples)
    
    # Add spectral-like features (e.g., signal intensities at different wavelengths)
    wavelengths = np.linspace(400, 900, 20)  # 20 wavelength points from 400-900nm
    
    # Generate spectral patterns for each sample
    for i, wl in enumerate(wavelengths):
        # Create a feature name like 'intensity_450nm'
        feature_name = f'intensity_{int(wl)}nm'
        
        # Base spectrum pattern depends on the 'category_balanced' feature
        base_intensity = np.zeros(n_samples)
        
        # Different spectral profiles for different categories
        for cat, multiplier in zip(categories, [1.0, 0.8, 1.2, 0.7, 1.5]):
            mask = df['category_balanced'] == cat
            # Gaussian-shaped peak centered at different wavelengths for each category
            center = 400 + 125 * categories.index(cat)
            base_intensity[mask] = multiplier * np.exp(-(wl - center)**2 / 5000)
        
        # Add noise and baseline
        df[feature_name] = base_intensity + 0.05 * np.random.random(n_samples) + 0.1
    
    return df

def save_dataset(df, filename='feature_engineering_demo.csv'):
    """Save the dataset to the uploads directory"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    return filepath

def register_in_database(filepath, experiment_id=None):
    """Register the dataset in the database"""
    with sqlite3.connect(DB_PATH) as conn:
        # Create experiment if needed
        if CREATE_EXPERIMENT and not experiment_id:
            experiment_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            conn.execute(
                "INSERT INTO experiments (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (experiment_id, EXPERIMENT_NAME, "Demo for testing feature engineering operations", now, now)
            )
            print(f"Created experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
        
        # Register dataset
        if experiment_id:
            dataset_id = str(uuid.uuid4())
            filename = os.path.basename(filepath)
            file_ext = os.path.splitext(filename)[1].lower()[1:]  # Remove the '.'
            
            conn.execute(
                "INSERT INTO datasets (id, experiment_id, filename, filetype, created_at) VALUES (?, ?, ?, ?, ?)",
                (dataset_id, experiment_id, filename, file_ext, datetime.now().isoformat())
            )
            print(f"Registered dataset: {filename} (ID: {dataset_id})")
            
            return dataset_id
    
    return None

def main():
    """Main function to create and register demo data"""
    print("Creating demo dataset for feature engineering...")
    df = create_demo_dataset()
    
    print(f"Created dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Save and register
    filepath = save_dataset(df)
    dataset_id = register_in_database(filepath)
    
    if dataset_id:
        print("Dataset ready for feature engineering!")
        print(f"Access it at: http://localhost:5500/experiment/{experiment_id}/feature-engineering?dataset={dataset_id}")

if __name__ == "__main__":
    main()
