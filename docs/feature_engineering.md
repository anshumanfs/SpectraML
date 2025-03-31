# Feature Engineering for Machine Learning

[← Back to Index](index.md)

Feature engineering is the process of transforming raw data into features that better represent the underlying problem, resulting in improved model performance. It's often considered the most critical step in the machine learning pipeline and can be the difference between a mediocre model and an excellent one.

## Importance of Feature Engineering

- **Improves Model Performance**: Well-crafted features can dramatically enhance predictive accuracy
- **Reduces Model Complexity**: Good features can simplify the learning task
- **Accelerates Training**: Relevant features can speed up convergence
- **Makes Models More Interpretable**: Thoughtful features can make models easier to understand
- **Domain Knowledge Integration**: Allows incorporation of expert knowledge into the model

## The Feature Engineering Process

1. **Feature Understanding**: Analyze the relationship between features and the target
2. **Feature Creation**: Generate new features from existing ones
3. **Feature Transformation**: Convert features to more useful forms
4. **Feature Selection**: Choose the most relevant features
5. **Feature Validation**: Evaluate the impact of engineered features

## Numeric Data Transformation Techniques

### Scaling and Normalization

Scaling ensures that features with different ranges contribute equally to the model.

#### Min-Max Scaling
Scales features to a fixed range, typically [0, 1]:

```python
X_scaled = (X - X.min()) / (X.max() - X.min())
```

#### Standardization (Z-score normalization)
Transforms features to have zero mean and unit variance:

```python
X_standardized = (X - X.mean()) / X.std()
```

#### Robust Scaling
Uses median and interquartile range instead of mean and standard deviation, making it robust to outliers:

```python
X_robust = (X - X.median()) / (X.quantile(0.75) - X.quantile(0.25))
```

### Binning

Converts continuous features into discrete bins.

#### Equal-width binning
Creates bins of equal width:

```python
pd.cut(X, bins=10)
```

#### Equal-frequency binning
Creates bins with equal number of observations:

```python
pd.qcut(X, q=10)
```

#### Custom binning
Creates bins based on domain knowledge:

```python
bins = [0, 18, 35, 50, 65, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Middle-aged', 'Senior']
pd.cut(df['age'], bins=bins, labels=labels)
```

### Mathematical Transformations

#### Log Transform
Handles skewed distributions:

```python
X_log = np.log1p(X)  # log(1+x) to handle zeros
```

#### Square Root Transform
Another option for handling right-skewed data:

```python
X_sqrt = np.sqrt(X)
```

#### Box-Cox Transform
A generalized power transformation:

```python
from scipy import stats
X_boxcox, lambda_value = stats.boxcox(X)
```

#### Yeo-Johnson Transform
Similar to Box-Cox but can handle negative values:

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
X_yj = pt.fit_transform(X)
```

## Categorical Data Transformation

### One-Hot Encoding
Converts categorical variables into binary vectors:

```python
pd.get_dummies(df['category'])
```

### Label Encoding
Assigns a unique integer to each category:

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_encoded = le.fit_transform(X)
```

### Target Encoding
Replaces categories with their target statistics:

```python
target_means = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(target_means)
```

### Binary Encoding
Represents categories as binary digits:

```python
from category_encoders import BinaryEncoder
encoder = BinaryEncoder()
X_binary = encoder.fit_transform(X)
```

### Feature Hashing
Maps categories to vector indices using hash functions:

```python
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=10)
X_hashed = hasher.transform(X)
```

## Feature Creation Techniques

### Interaction Features
Create new features by combining existing ones:

```python
df['area'] = df['height'] * df['width']
df['bmi'] = df['weight'] / (df['height'] ** 2)
```

### Polynomial Features
Generate polynomial and interaction terms:

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

### Domain-Specific Features
Create features based on domain knowledge:

```python
# E-commerce example
df['avg_session_value'] = df['total_purchase'] / df['session_count']
df['days_since_last_purchase'] = (today - df['last_purchase_date']).dt.days
```

### Aggregation Features
Create summary statistics from grouped data:

```python
# For each customer, compute statistics across all their purchases
customer_stats = df.groupby('customer_id')['purchase_amount'].agg(['mean', 'min', 'max', 'sum', 'count'])
```

## Time-Based Feature Engineering

### Extracting Components
Break down datetime into components:

```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour
```

### Cyclical Features
Convert cyclical time features to continuous representation:

```python
# For month, which cycles every 12 months
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
```

### Time Differences
Calculate time elapsed between events:

```python
df['days_since_previous'] = df.groupby('customer_id')['date'].diff().dt.days
```

### Lag Features
Create features based on previous values:

```python
df['sales_prev_day'] = df['sales'].shift(1)
df['sales_prev_week'] = df['sales'].shift(7)
```

### Rolling Windows
Calculate statistics over moving time periods:

```python
df['sales_rolling_mean_7d'] = df['sales'].rolling(window=7).mean()
df['sales_rolling_std_7d'] = df['sales'].rolling(window=7).std()
```

## Text Feature Engineering

### Bag of Words
Count the frequency of each word:

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(text_data)
```

### TF-IDF (Term Frequency-Inverse Document Frequency)
Weights terms by their importance:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(text_data)
```

### Word Embeddings
Represent words as dense vectors:

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
```

### Text Statistics
Extract statistical features from text:

```python
df['text_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]))
```

## Feature Selection Methods

### Filter Methods
Select features based on statistical tests:

#### Variance Threshold
Remove features with low variance:

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)
```

#### Correlation
Remove highly correlated features:

```python
# Remove features with correlation > 0.9
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_uncorrelated = X.drop(to_drop, axis=1)
```

#### Statistical Tests
Select features based on relationship with target:

```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_best = selector.fit_transform(X, y)
```

### Wrapper Methods
Use model performance to select features:

#### Recursive Feature Elimination
Recursively remove features:

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
```

#### Forward/Backward Selection
Add or remove features iteratively:

```python
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
sfs = SequentialFeatureSelector(LinearRegression(), 
                               k_features=10, 
                               forward=True, 
                               scoring='r2')
X_selected = sfs.fit_transform(X, y)
```

### Embedded Methods
Models that perform feature selection during training:

#### LASSO Regression
Adds L1 regularization that can zero out unimportant features:

```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
# Features with non-zero coefficients
selected_features = X.columns[lasso.coef_ != 0]
```

#### Tree-based Feature Importance
Use feature importance from tree models:

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
importances = rf.feature_importances_
```

## Feature Engineering for Spectral Data

Spectral data presents unique challenges and opportunities for feature engineering:

### Peak Detection
Identify and characterize spectral peaks:

```python
from scipy.signal import find_peaks
peaks, _ = find_peaks(spectrum, height=0.1, distance=10)
```

### Baseline Correction
Remove background interference:

```python
from pybaselines import Baseline
baseline_fitter = Baseline()
baseline = baseline_fitter.mor(spectrum)
corrected_spectrum = spectrum - baseline
```

### Normalization
Account for variations in overall intensity:

```python
# Standard Normal Variate (SNV)
def snv(spectra):
    return (spectra - spectra.mean(axis=1, keepdims=True)) / spectra.std(axis=1, keepdims=True)
```

### Derivative Spectroscopy
Enhance subtle spectral features:

```python
from scipy.signal import savgol_filter
# First derivative
first_derivative = savgol_filter(spectrum, window_length=15, polyorder=2, deriv=1)
# Second derivative
second_derivative = savgol_filter(spectrum, window_length=15, polyorder=2, deriv=2)
```

### Dimensionality Reduction
Reduce the high dimensionality of spectral data:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(spectra)
```

## Best Practices for Feature Engineering

1. **Start Simple**: Begin with basic transformations and gradually add complexity
2. **Use Domain Knowledge**: Incorporate expert insights about the data
3. **Validate Features**: Measure the impact of each feature on model performance
4. **Document Everything**: Keep track of all transformations and their effects
5. **Feature Store**: Consider implementing a feature store for reusability
6. **Feature Versioning**: Version control your features like code
7. **Iterate**: Feature engineering is an iterative process requiring experimentation

## Common Pitfalls

- **Data Leakage**: Inadvertently including target information in features
- **Overcomplicating**: Creating too many features can lead to overfitting
- **Correlation Blindness**: Not considering correlations between features
- **Ignoring Outliers**: Forgetting to handle outliers before transformations
- **Improper Scaling**: Using inappropriate scaling methods
- **Forgetting Missing Values**: Not addressing missing values in features

## Tools for Feature Engineering

- **Scikit-learn**: Comprehensive library with preprocessing tools
- **Feature-engine**: Specialized feature engineering library
- **FeatureTools**: Automated feature engineering
- **Pandas**: Essential for data manipulation
- **Category Encoders**: Advanced categorical encoding techniques

---

## Navigation

**Next**: [Model Selection](model_selection.md)  
**Previous**: [Introduction to Machine Learning](introduction.md)

**Related Topics**:
- [Spectral Data Analysis](spectral_data.md)
- [Model Evaluation](model_evaluation.md)
- [Advanced Topics](advanced_topics.md)
