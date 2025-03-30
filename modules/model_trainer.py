import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, StackingClassifier, StackingRegressor
from sklearn.svm import SVC
import warnings

class ModelTrainer:
    """
    Handles training of machine learning models for classification and regression tasks.
    """
    
    def __init__(self):
        self.supported_models = [
            'logistic_regression', 'linear_regression', 'random_forest_classifier',
            'random_forest_regressor', 'gradient_boosting', 'stacking_classifier',
            'stacking_regressor'
        ]
    
    def train(self, file_path, model_type, config):
        """
        Train a machine learning model
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        model_type : str
            Type of model to train
        config : dict
            Configuration for the model
            
        Returns:
        --------
        dict
            Results of the training including metrics and trained model
        """
        # Load data
        df = pd.read_csv(file_path)
        target_column = config.get('target_column')
        
        if not target_column or target_column not in df.columns:
            raise ValueError("A valid target column must be specified")
        
        # Split data into features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data into training and testing sets
        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Get feature names and class names
        feature_names = X.columns.tolist()
        class_names = y.unique().tolist() if y.nunique() <= 10 else None
        
        # Train model based on type
        if model_type == 'logistic_regression':
            return self._train_logistic_regression(X_train, X_test, y_train, y_test, config, feature_names, class_names)
        elif model_type == 'linear_regression':
            return self._train_linear_regression(X_train, X_test, y_train, y_test, config, feature_names)
        elif model_type == 'random_forest_classifier':
            return self._train_random_forest_classifier(X_train, X_test, y_train, y_test, config, feature_names, class_names)
        elif model_type == 'random_forest_regressor':
            return self._train_random_forest_regressor(X_train, X_test, y_train, y_test, config, feature_names)
        elif model_type == 'gradient_boosting':
            return self._train_gradient_boosting(X_train, X_test, y_train, y_test, config, feature_names)
        elif model_type == 'stacking_classifier':
            return self._train_stacking_classifier(X_train, X_test, y_train, y_test, config, feature_names, class_names)
        elif model_type == 'stacking_regressor':
            return self._train_stacking_regressor(X_train, X_test, y_train, y_test, config, feature_names)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {self.supported_models}")
    
    def _train_logistic_regression(self, X_train, X_test, y_train, y_test, config, feature_names, class_names):
        """Train a logistic regression model"""
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_prob, 'classification')
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': None
        }
    
    def _train_linear_regression(self, X_train, X_test, y_train, y_test, config, feature_names):
        """Train a linear regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, None, 'regression')
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': None
        }
    
    def _train_random_forest_classifier(self, X_train, X_test, y_train, y_test, config, feature_names, class_names):
        """Train a random forest classifier"""
        n_estimators = config.get('n_estimators', 100)
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_prob, 'classification')
        
        # Get feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    def _train_random_forest_regressor(self, X_train, X_test, y_train, y_test, config, feature_names):
        """Train a random forest regressor"""
        n_estimators = config.get('n_estimators', 100)
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, None, 'regression')
        
        # Get feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    def _train_gradient_boosting(self, X_train, X_test, y_train, y_test, config, feature_names):
        """Train a gradient boosting regressor"""
        n_estimators = config.get('n_estimators', 100)
        model = GradientBoostingRegressor(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, None, 'regression')
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': None
        }
    
    def _train_stacking_classifier(self, X_train, X_test, y_train, y_test, config, feature_names, class_names):
        """Train a stacking ensemble of classifiers"""
        # Get ensemble configuration
        ensemble_config = config.get('ensemble_config', {})
        estimators = ensemble_config.get('estimators', [
            ('lr', LogisticRegression()),
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('svm', SVC(probability=True))
        ])
        final_estimator = ensemble_config.get('final_estimator', LogisticRegression())
        cv = ensemble_config.get('cv', 5)
        
        # Create and train stacking classifier
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_prob, 'classification')
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': None
        }
    
    def _train_stacking_regressor(self, X_train, X_test, y_train, y_test, config, feature_names):
        """Train a stacking ensemble of regressors"""
        # Get ensemble configuration
        ensemble_config = config.get('ensemble_config', {})
        estimators = ensemble_config.get('estimators', [
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(n_estimators=100)),
            ('gbr', GradientBoostingRegressor())
        ])
        final_estimator = ensemble_config.get('final_estimator', LinearRegression())
        cv = ensemble_config.get('cv', 5)
        
        # Create and train stacking regressor
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, None, 'regression')
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': None
        }
    
    def _calculate_metrics(self, y_true, y_pred, y_prob, task_type):
        """
        Calculate evaluation metrics
        
        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted target values
        y_prob : array-like, optional
            Predicted probabilities (for classification)
        task_type : str
            Type of task ('classification' or 'regression')
            
        Returns:
        --------
        dict
            Calculated metrics
        """
        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            if y_prob is not None:
                metrics['roc_auc'] = None  # Add ROC AUC calculation if needed
        elif task_type == 'regression':
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred)
            }
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return metrics