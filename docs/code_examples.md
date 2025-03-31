# Code Examples for Machine Learning with Spectral Data

[← Back to Index](index.md)

This page provides practical code examples for various machine learning tasks with spectral data using Python.

## Table of Contents

- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Feature Engineering for Spectral Data](#feature-engineering-for-spectral-data)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Advanced Techniques](#advanced-techniques)

## Data Loading and Preprocessing

### Loading Spectral Data from CSV

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load spectral data with wavelengths as columns
def load_spectral_data(file_path):
    """Load spectral data from CSV file."""
    df = pd.read_csv(file_path)
    
    # Extract wavelengths from column names (assuming format like 'wl_850.0')
    wavelengths = [float(col.split('_')[1]) for col in df.columns if col.startswith('wl_')]
    
    # Extract spectral data
    spectra = df[[col for col in df.columns if col.startswith('wl_')]].values
    
    # Extract metadata if available
    metadata = df[[col for col in df.columns if not col.startswith('wl_')]]
    
    return spectra, wavelengths, metadata

# Example usage
spectra, wavelengths, metadata = load_spectral_data('spectral_data.csv')

# Plot some example spectra
plt.figure(figsize=(10, 6))
for i in range(min(5, len(spectra))):
    plt.plot(wavelengths, spectra[i], label=f'Sample {i+1}')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.legend()
plt.title('Example Spectra')
plt.show()
```

### Basic Preprocessing for Spectral Data

```python
from scipy.signal import savgol_filter

def preprocess_spectra(spectra, wavelengths):
    """Apply common preprocessing steps to spectral data."""
    # 1. Savitzky-Golay smoothing
    spectra_smoothed = np.array([savgol_filter(spectrum, window_length=11, polyorder=3) 
                                for spectrum in spectra])
    
    # 2. Baseline correction (simple linear)
    spectra_baseline = np.zeros_like(spectra_smoothed)
    for i, spectrum in enumerate(spectra_smoothed):
        # Simple linear baseline using first and last points
        baseline = np.linspace(spectrum[0], spectrum[-1], len(spectrum))
        spectra_baseline[i] = spectrum - baseline
    
    # 3. Standard Normal Variate (SNV) normalization
    spectra_normalized = np.zeros_like(spectra_baseline)
    for i, spectrum in enumerate(spectra_baseline):
        spectra_normalized[i] = (spectrum - np.mean(spectrum)) / np.std(spectrum)
    
    return spectra_normalized

# Example usage
preprocessed_spectra = preprocess_spectra(spectra, wavelengths)

# Plot original vs preprocessed
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(wavelengths, spectra[0])
plt.title('Original Spectrum')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')

plt.subplot(2, 1, 2)
plt.plot(wavelengths, preprocessed_spectra[0])
plt.title('Preprocessed Spectrum')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Intensity')
plt.tight_layout()
plt.show()
```

## Feature Engineering for Spectral Data

### Extracting Peak Features

```python
from scipy.signal import find_peaks

def extract_peak_features(spectra, wavelengths, prominence=0.1, width=3):
    """Extract features from peaks in spectral data."""
    peak_features = []
    
    for spectrum in spectra:
        # Find peaks
        peaks, properties = find_peaks(spectrum, prominence=prominence, width=width)
        
        if len(peaks) == 0:
            # No peaks found, return zeros
            peak_features.append([0, 0, 0, 0, 0])
            continue
        
        # Extract peak wavelengths
        peak_wavelengths = [wavelengths[i] for i in peaks]
        
        # Extract peak intensities
        peak_intensities = [spectrum[i] for i in peaks]
        
        # Calculate peak features
        features = [
            len(peaks),  # Number of peaks
            np.mean(peak_intensities),  # Mean peak intensity
            np.max(peak_intensities),  # Max peak intensity
            np.mean(properties['widths']),  # Mean peak width
            np.mean(properties['prominences'])  # Mean peak prominence
        ]
        
        peak_features.append(features)
    
    feature_names = ['peak_count', 'mean_peak_intensity', 'max_peak_intensity', 
                     'mean_peak_width', 'mean_peak_prominence']
    
    return np.array(peak_features), feature_names

# Example usage
peak_features, feature_names = extract_peak_features(preprocessed_spectra, wavelengths)
peak_df = pd.DataFrame(peak_features, columns=feature_names)
print(peak_df.head())
```

### Dimensionality Reduction with PCA

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(spectra, n_components=10):
    """Apply PCA to reduce dimensionality of spectral data."""
    # Scale the data
    scaler = StandardScaler()
    spectra_scaled = scaler.fit_transform(spectra)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(spectra_scaled)
    
    # Get component information
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"Explained variance by {n_components} components: {cumulative_variance[-1]:.4f}")
    
    # Component analysis
    loadings = pca.components_
    
    return pca_result, pca, scaler, loadings, explained_variance

# Example usage
pca_features, pca_model, scaler, loadings, explained_variance = apply_pca(preprocessed_spectra)

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), 'r-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.xticks(range(1, len(explained_variance) + 1))
plt.tight_layout()
plt.show()
```

## Model Training and Evaluation

### Training a Classification Model for Spectral Data

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def train_classifier(features, labels):
    """Train a random forest classifier on spectral features."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Create and train the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Feature importance
    if hasattr(clf, 'feature_importances_'):
        feature_importance = clf.feature_importances_
        return clf, feature_importance
    
    return clf, None

# Example usage (assuming we have labels for our samples)
labels = metadata['class_label'].values  # Replace with your actual label column
model, feature_importance = train_classifier(pca_features, labels)

# Plot feature importance if available
if feature_importance is not None:
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
```

### Partial Least Squares Regression for Spectral Data

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_pls_regression(spectra, target_values, n_components=10):
    """Train PLS regression model on spectral data."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, target_values, test_size=0.3, random_state=42
    )
    
    # Create and train PLS model
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = pls.predict(X_train)
    y_test_pred = pls.predict(X_test)
    
    # Evaluate model
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('PLS Regression: Actual vs Predicted')
    plt.tight_layout()
    plt.show()
    
    return pls, test_rmse, test_r2

# Example usage (assuming we have continuous target values)
target_values = metadata['concentration'].values  # Replace with your actual target column
pls_model, rmse, r2 = train_pls_regression(preprocessed_spectra, target_values)
```

## Advanced Techniques

### 1D Convolutional Neural Network for Spectral Classification

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def build_1d_cnn(input_shape, num_classes):
    """Build a 1D CNN for spectral classification."""
    model = Sequential([
        # First convolutional layer
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        # Second convolutional layer
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Third convolutional layer
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Flatten and fully connected layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_cnn_classifier(spectra, labels):
    """Train a 1D CNN classifier on spectral data."""
    # Reshape spectra for CNN input (samples, timesteps, features)
    X = spectra.reshape(spectra.shape[0], spectra.shape[1], 1)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y = to_categorical(y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Build model
    model = build_1d_cnn(input_shape=(X.shape[1], 1), num_classes=y.shape[1])
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return model, label_encoder, history

# Example usage
# cnn_model, label_encoder, history = train_cnn_classifier(preprocessed_spectra, labels)
```

### Transfer Learning with Pretrained Models

```python
def apply_transfer_learning(source_spectra, source_labels, target_spectra, target_labels):
    """Apply transfer learning from a source domain to a target domain."""
    # This is a simplified example. In a real scenario, you would:
    # 1. Train a model on source data
    # 2. Freeze some layers
    # 3. Fine-tune on target data
    
    # Step 1: Train source model (using a simple RandomForest as example)
    from sklearn.ensemble import RandomForestClassifier
    
    source_model = RandomForestClassifier(n_estimators=100, random_state=42)
    source_model.fit(source_spectra, source_labels)
    
    # Step 2: Use source model predictions as features for target model
    source_predictions = source_model.predict_proba(target_spectra)
    
    # Combine original features with source model predictions
    enhanced_features = np.hstack([target_spectra, source_predictions])
    
    # Step 3: Train target model with enhanced features
    target_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        enhanced_features, target_labels, test_size=0.3, random_state=42, stratify=target_labels
    )
    
    target_model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = target_model.score(X_test, y_test)
    print(f"Transfer learning model accuracy: {accuracy:.4f}")
    
    return target_model

# Example usage would require both source and target domain data
```

---

## Navigation

**Next**: [References](references.md)  
**Previous**: [Glossary of Terms](glossary.md)  

**Related Topics**:
- [Feature Engineering](feature_engineering.md)
- [Model Selection](model_selection.md)
- [Spectral Data Analysis](spectral_data.md)
