import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_trainer')

class ModelTrainer:
    """
    Handles training of machine learning models.
    """
    
    def __init__(self):
        """Initialize the model trainer with supported models"""
        # Dictionary of supported model types and their default configurations
        self.supported_models = {
            'linear_regression': {
                'type': 'regression',
                'model': 'sklearn.linear_model.LinearRegression',
                'default_params': {}
            },
            'logistic_regression': {
                'type': 'classification',
                'model': 'sklearn.linear_model.LogisticRegression',
                'default_params': {'max_iter': 1000, 'C': 1.0}
            },
            'random_forest': {
                'type': 'classification',
                'model': 'sklearn.ensemble.RandomForestClassifier',
                'default_params': {'n_estimators': 100, 'max_depth': 10}
            },
            'random_forest_regressor': {
                'type': 'regression',
                'model': 'sklearn.ensemble.RandomForestRegressor',
                'default_params': {'n_estimators': 100, 'max_depth': 10}
            },
            'svm': {
                'type': 'classification',
                'model': 'sklearn.svm.SVC',
                'default_params': {'kernel': 'rbf', 'C': 1.0}
            },
            'gradient_boosting': {
                'type': 'classification',
                'model': 'sklearn.ensemble.GradientBoostingClassifier',
                'default_params': {'n_estimators': 100, 'learning_rate': 0.1}
            },
            'gradient_boosting_regressor': {
                'type': 'regression',
                'model': 'sklearn.ensemble.GradientBoostingRegressor',
                'default_params': {'n_estimators': 100, 'learning_rate': 0.1}
            }
        }
        
        # Storage directory for saved models
        self.model_dir = os.path.join('storage', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train(self, file_path, model_type, config):
        """
        Train a model on the provided dataset
        
        Parameters:
        -----------
        file_path : str
            Path to the dataset file
        model_type : str
            Type of model to train
        config : dict
            Configuration for model training
            
        Returns:
        --------
        dict
            Results of model training including metrics
        """
        logger.info(f"Starting model training for {model_type} on {file_path}")
        
        try:
            # Validate model type
            if model_type not in self.supported_models:
                raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(self.supported_models.keys())}")
            
            # Load dataset with robust error handling
            try:
                # Determine file type from extension
                file_ext = os.path.splitext(file_path)[1].lower()
                
                # Use robust CSV loading with error handling
                if file_ext == '.csv':
                    # First try with standard options
                    try:
                        df = pd.read_csv(file_path)
                    except pd.errors.ParserError as e:
                        logger.warning(f"CSV parsing error: {str(e)}")
                        logger.info("Trying with error handling options...")
                        
                        # Try to detect CSV dialect
                        import csv
                        with open(file_path, 'r', newline='', errors='replace') as f:
                            sample = f.read(4096)
                        try:
                            dialect = csv.Sniffer().sniff(sample)
                            logger.info(f"Detected CSV dialect: delimiter='{dialect.delimiter}', quotechar='{dialect.quotechar}'")
                            
                            # Try to read with detected dialect
                            df = pd.read_csv(
                                file_path,
                                delimiter=dialect.delimiter,
                                quotechar=dialect.quotechar,
                                error_bad_lines=False,
                                warn_bad_lines=True,
                                on_bad_lines='skip'  # For pandas >= 1.3
                            )
                        except:
                            # Fallback to most permissive options
                            logger.warning("Dialect detection failed, using fallback options")
                            df = pd.read_csv(
                                file_path,
                                error_bad_lines=False,
                                warn_bad_lines=True,
                                on_bad_lines='skip',
                                low_memory=False
                            )
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                elif file_ext == '.json':
                    df = pd.read_json(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
                
                logger.info(f"Successfully loaded dataset with shape {df.shape}")
            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                raise
            
            # Extract configuration parameters
            target_column = config.get('target_column')
            if not target_column:
                raise ValueError("Target column must be specified in configuration")
                
            # Validate target column exists
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}")
            
            # Get feature columns (all except target)
            ignore_columns = config.get('ignore_columns', [])
            feature_columns = [col for col in df.columns if col != target_column and col not in ignore_columns]
            
            if not feature_columns:
                raise ValueError("No feature columns available for training")
            
            # Extract features and target
            X = df[feature_columns]
            y = df[target_column]
            
            # Log data info
            logger.info(f"Feature columns: {feature_columns}")
            logger.info(f"Target column: {target_column}")
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Target shape: {y.shape}")
            
            # Check for nulls in target column
            if y.isnull().any():
                logger.warning(f"Target column contains {y.isnull().sum()} null values")
                logger.info("Removing rows with null target values")
                non_null_indices = y.notnull()
                X = X[non_null_indices]
                y = y[non_null_indices]
            
            # Identify numeric and categorical columns for preprocessing
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            logger.info(f"Numeric columns: {numeric_cols}")
            logger.info(f"Categorical columns: {categorical_cols}")
            
            # Create preprocessing pipeline
            preprocessor = self._create_preprocessor(numeric_cols, categorical_cols)
            
            # Split data into train/test sets
            test_size = float(config.get('test_size', 0.2))
            random_state = int(config.get('random_state', 42))
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                logger.info(f"Data split: train={X_train.shape}, test={X_test.shape}")
            except Exception as e:
                logger.error(f"Error splitting data: {str(e)}")
                raise
            
            # Get model class from string
            model_info = self.supported_models[model_type]
            model_path = model_info['model'].split('.')
            model_module = __import__('.'.join(model_path[:-1]), fromlist=[model_path[-1]])
            ModelClass = getattr(model_module, model_path[-1])
            
            # Merge default params with user-provided params
            default_params = model_info['default_params'].copy()
            user_params = config.get('model_params', {})
            model_params = {**default_params, **user_params}
            
            # Log model parameters
            logger.info(f"Training {model_type} with parameters: {model_params}")
            
            # Create and train model pipeline
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('model', ModelClass(**model_params))
            ])
            
            # Train the model with more robust error handling
            try:
                model.fit(X_train, y_train)
                logger.info("Model training completed successfully")
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                
                # Try to diagnose what went wrong
                if "could not convert string to float" in str(e):
                    # Check for non-numeric data in columns expected to be numeric
                    for col in numeric_cols:
                        if pd.api.types.is_object_dtype(X[col]):
                            sample_values = X[col].dropna().head().tolist()
                            logger.error(f"Column '{col}' contains non-numeric values: {sample_values}")
                
                # Check for categorical columns with many unique values
                for col in categorical_cols:
                    n_unique = X[col].nunique()
                    if n_unique > 100:
                        logger.warning(f"Column '{col}' has {n_unique} unique values, which may cause issues for one-hot encoding")
                
                # Add more diagnostics as needed
                raise
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            
            # Calculate appropriate metrics based on problem type
            metrics = {}
            if model_info['type'] == 'classification':
                # Check if binary or multiclass
                if len(np.unique(y)) <= 2:
                    # Binary classification
                    metrics = {
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                        'f1': float(f1_score(y_test, y_pred, zero_division=0))
                    }
                else:
                    # Multiclass - use macro averaging
                    metrics = {
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
                        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
                        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0))
                    }
            else:  # regression
                metrics = {
                    'mse': float(mean_squared_error(y_test, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    'r2': float(r2_score(y_test, y_pred))
                }
            
            logger.info(f"Model evaluation metrics: {metrics}")
            
            # Save model
            model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_dir, f"{model_type}_{model_id}.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Create and return results
            result = {
                'success': True,
                'model_id': model_id,
                'model_type': model_type,
                'metrics': metrics,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'model_path': model_path,
                'training_config': config
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_type': model_type
            }
    
    def _create_preprocessor(self, numeric_cols, categorical_cols):
        """
        Create a column transformer for preprocessing data
        
        Parameters:
        -----------
        numeric_cols : list
            List of numeric column names
        categorical_cols : list
            List of categorical column names
            
        Returns:
        --------
        ColumnTransformer
            Preprocessor for the data
        """
        # For numeric columns: impute missing values and scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # For categorical columns: impute missing values and one-hot encode
        # Only use OneHotEncoder if there are categorical columns
        if categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Create column transformer with both numeric and categorical pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )
        else:
            # If no categorical columns, only use numeric pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols)
                ]
            )
        
        return preprocessor
    
    def load_model(self, model_path):
        """
        Load a trained model from file
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
            
        Returns:
        --------
        object
            Loaded model
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, model_path, data):
        """
        Make predictions with a trained model
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        data : pandas.DataFrame
            Data to make predictions on
            
        Returns:
        --------
        array
            Predictions
        """
        try:
            model = self.load_model(model_path)
            return model.predict(data)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise