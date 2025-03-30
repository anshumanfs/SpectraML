import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
import umap
import warnings
from modules.data_loader import DataLoader
import os

class FeatureEngineer:
    """
    Performs various feature engineering operations on data.
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.supported_operations = [
            'scaling', 'normalization', 'imputation', 'one_hot_encoding',
            'binning', 'polynomial_features', 'log_transform', 'box_cox',
            'pca', 'tsne', 'umap', 'feature_selection', 'datetime_features',
            'text_vectorization', 'outlier_detection', 'binning', 'clustering'
        ]
    
    def apply_operations(self, file_path, operations):
        """
        Apply a sequence of feature engineering operations to the data
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        operations : list of dict
            List of operations to apply, each with type and parameters
            
        Returns:
        --------
        dict
            Results of the operations including transformed data
        """
        # Load data
        df = self.data_loader.load_data(file_path)
        file_type = file_path.split('.')[-1].lower()
        
        # Track applied operations and their results
        applied_operations = []
        
        # Apply each operation in sequence
        for op in operations:
            op_type = op.get('type')
            params = op.get('params', {})
            
            if op_type not in self.supported_operations:
                warnings.warn(f"Unsupported operation: {op_type}. Skipping.")
                continue
            
            # Apply the operation
            try:
                result = self._apply_operation(df, op_type, params)
                
                # Update the dataframe with the transformed one
                df = result.get('data')
                
                # Save operation details
                applied_operations.append({
                    'type': op_type,
                    'params': params,
                    'metrics': result.get('metrics', {}),
                    'columns_added': result.get('columns_added', []),
                    'columns_removed': result.get('columns_removed', [])
                })
                
            except Exception as e:
                applied_operations.append({
                    'type': op_type,
                    'params': params,
                    'error': str(e)
                })
        
        # Save the transformed data
        output_dir = 'storage/processed'
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"processed_{os.path.basename(file_path)}")
        self.data_loader.save_data(df, output_file, file_type)
        
        # Return results
        return {
            'success': True,
            'data_info': self.data_loader.get_data_info(output_file, file_type),
            'operations': applied_operations,
            'output_file': output_file
        }
    
    def preview_operations(self, file_path, operations):
        """
        Preview the results of feature engineering operations without saving
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        operations : list of dict
            List of operations to apply, each with type and parameters
            
        Returns:
        --------
        dict
            Preview results including sample of transformed data and summary
        """
        # Load data
        df = self.data_loader.load_data(file_path)
        original_shape = df.shape
        
        # Track applied operations and their results
        applied_operations = []
        
        # Apply each operation in sequence
        for op in operations:
            op_type = op.get('type')
            params = op.get('params', {})
            
            if op_type not in self.supported_operations:
                warnings.warn(f"Unsupported operation: {op_type}. Skipping.")
                continue
            
            # Apply the operation
            try:
                result = self._apply_operation(df, op_type, params)
                
                # Update the dataframe with the transformed one
                df = result.get('data')
                
                # Save operation details
                applied_operations.append({
                    'type': op_type,
                    'params': params,
                    'metrics': result.get('metrics', {}),
                    'columns_added': result.get('columns_added', []),
                    'columns_removed': result.get('columns_removed', [])
                })
                
            except Exception as e:
                applied_operations.append({
                    'type': op_type,
                    'params': params,
                    'error': str(e)
                })
        
        # Create a summary of changes
        new_shape = df.shape
        
        # Get first 5 rows for preview (convert to dict for JSON serialization)
        preview_sample = df.head(5).to_dict(orient='records')
        
        # Get column information
        columns_info = []
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing': int(df[col].isnull().sum()),
                'missing_pct': float(df[col].isnull().sum() / len(df) * 100)
            }
            columns_info.append(col_info)
        
        return {
            'success': True,
            'original_shape': {
                'rows': original_shape[0],
                'columns': original_shape[1]
            },
            'new_shape': {
                'rows': new_shape[0],
                'columns': new_shape[1]
            },
            'operations': applied_operations,
            'preview_sample': preview_sample,
            'columns_info': columns_info
        }
    
    def _apply_operation(self, df, op_type, params):
        """
        Apply a single feature engineering operation
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data to transform
        op_type : str
            Type of operation to apply
        params : dict
            Parameters for the operation
            
        Returns:
        --------
        dict
            Result of the operation
        """
        if op_type == 'scaling':
            return self._apply_scaling(df, params)
        elif op_type == 'imputation':
            return self._apply_imputation(df, params)
        elif op_type == 'one_hot_encoding':
            return self._apply_one_hot_encoding(df, params)
        elif op_type == 'pca':
            return self._apply_pca(df, params)
        elif op_type == 'tsne':
            return self._apply_tsne(df, params)
        elif op_type == 'umap':
            return self._apply_umap(df, params)
        elif op_type == 'feature_selection':
            return self._apply_feature_selection(df, params)
        elif op_type == 'datetime_features':
            return self._apply_datetime_features(df, params)
        elif op_type == 'log_transform':
            return self._apply_log_transform(df, params)
        elif op_type == 'polynomial_features':
            return self._apply_polynomial_features(df, params)
        elif op_type == 'outlier_detection':
            return self._apply_outlier_detection(df, params)
        elif op_type == 'binning':
            return self._apply_binning(df, params)
        elif op_type == 'clustering':
            return self._apply_clustering(df, params)
        else:
            raise ValueError(f"Operation '{op_type}' not implemented")
    
    def _apply_scaling(self, df, params):
        """Apply scaling to numeric columns"""
        method = params.get('method', 'standard')
        columns = params.get('columns')
        
        # If no columns specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Select scaler based on method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Apply scaling
        result_df[columns] = scaler.fit_transform(df[columns])
        
        return {
            'data': result_df,
            'metrics': {'columns_scaled': len(columns)},
            'columns_added': []
        }
    
    def _apply_imputation(self, df, params):
        """Impute missing values"""
        method = params.get('method', 'mean')
        columns = params.get('columns')
        
        # If no columns specified, use all columns with missing values
        if not columns:
            columns = df.columns[df.isnull().any()].tolist()
        
        if not columns:
            return {'data': df, 'metrics': {'missing_values_filled': 0}}
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Select imputer based on method
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif method == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
        elif method == 'constant':
            imputer = SimpleImputer(strategy='constant', fill_value=params.get('fill_value', 0))
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=params.get('n_neighbors', 5))
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        # For numeric columns, use the imputer
        numeric_cols = list(set(columns) & set(df.select_dtypes(include=['number']).columns))
        if numeric_cols:
            result_df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # For categorical columns, use mode imputation
        cat_cols = list(set(columns) - set(numeric_cols))
        for col in cat_cols:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            result_df[col] = df[col].fillna(mode_val)
        
        return {
            'data': result_df,
            'metrics': {
                'missing_values_filled': (df[columns].isnull().sum().sum() - result_df[columns].isnull().sum().sum())
            }
        }
    
    def _apply_one_hot_encoding(self, df, params):
        """Apply one-hot encoding to categorical columns"""
        columns = params.get('columns')
        drop_first = params.get('drop_first', False)
        
        # If no columns specified, use all object and category columns
        if not columns:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not columns:
            return {'data': df, 'metrics': {'columns_encoded': 0}, 'columns_added': []}
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Apply one-hot encoding
        encoded = pd.get_dummies(result_df[columns], drop_first=drop_first)
        
        # Drop original columns if specified
        if params.get('drop_original', True):
            result_df = result_df.drop(columns, axis=1)
        
        # Add encoded columns
        result_df = pd.concat([result_df, encoded], axis=1)
        
        return {
            'data': result_df,
            'metrics': {'columns_encoded': len(columns)},
            'columns_added': encoded.columns.tolist(),
            'columns_removed': columns if params.get('drop_original', True) else []
        }
    
    def _apply_pca(self, df, params):
        """Apply PCA dimensionality reduction"""
        columns = params.get('columns')
        n_components = params.get('n_components', 2)
        
        # If no columns specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(columns) < 2:
            raise ValueError("At least 2 numeric columns required for PCA")
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Prepare data
        X = result_df[columns].dropna()
        
        # Apply PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_scaled)
        
        # Create new columns with PCA components
        component_cols = [f'pca_component_{i+1}' for i in range(n_components)]
        for i, col in enumerate(component_cols):
            result_df.loc[X.index, col] = components[:, i]
        
        return {
            'data': result_df,
            'metrics': {
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'total_variance_explained': pca.explained_variance_ratio_.sum()
            },
            'columns_added': component_cols
        }
    
    def _apply_tsne(self, df, params):
        """Apply t-SNE dimensionality reduction"""
        columns = params.get('columns')
        n_components = params.get('n_components', 2)
        perplexity = params.get('perplexity', 30.0)
        
        # If no columns specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(columns) < 2:
            raise ValueError("At least 2 numeric columns required for t-SNE")
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Prepare data
        X = result_df[columns].dropna()
        
        # Apply t-SNE
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        components = tsne.fit_transform(X_scaled)
        
        # Create new columns with t-SNE components
        component_cols = [f'tsne_component_{i+1}' for i in range(n_components)]
        for i, col in enumerate(component_cols):
            result_df.loc[X.index, col] = components[:, i]
        
        return {
            'data': result_df,
            'metrics': {
                'kl_divergence': tsne.kl_divergence_
            },
            'columns_added': component_cols
        }
    
    def _apply_umap(self, df, params):
        """Apply UMAP dimensionality reduction"""
        columns = params.get('columns')
        n_components = params.get('n_components', 2)
        n_neighbors = params.get('n_neighbors', 15)
        min_dist = params.get('min_dist', 0.1)
        
        # If no columns specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(columns) < 2:
            raise ValueError("At least 2 numeric columns required for UMAP")
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Prepare data
        X = result_df[columns].dropna()
        
        # Apply UMAP
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                          min_dist=min_dist, random_state=42)
        components = reducer.fit_transform(X_scaled)
        
        # Create new columns with UMAP components
        component_cols = [f'umap_component_{i+1}' for i in range(n_components)]
        for i, col in enumerate(component_cols):
            result_df.loc[X.index, col] = components[:, i]
        
        return {
            'data': result_df,
            'metrics': {},
            'columns_added': component_cols
        }
    
    def _apply_feature_selection(self, df, params):
        """Apply feature selection methods"""
        columns = params.get('columns')
        target = params.get('target')
        method = params.get('method', 'f_classif')
        k = params.get('k', 10)
        
        if not target or target not in df.columns:
            raise ValueError("A valid target column must be specified")
        
        # If no columns specified, use all numeric columns except target
        if not columns:
            columns = [col for col in df.select_dtypes(include=['number']).columns 
                      if col != target]
        
        if len(columns) < 2:
            raise ValueError("At least 2 feature columns required for feature selection")
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Prepare data
        X = result_df[columns].dropna()
        y = result_df.loc[X.index, target]
        
        # Select feature selection method
        if method == 'chi2':
            selector = SelectKBest(chi2, k=min(k, len(columns)))
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=min(k, len(columns)))
        elif method == 'mutual_info_classif':
            selector = SelectKBest(mutual_info_classif, k=min(k, len(columns)))
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Apply feature selection
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        mask = selector.get_support()
        selected_columns = [columns[i] for i in range(len(columns)) if mask[i]]
        
        # Update dataframe if drop_unselected is True
        if params.get('drop_unselected', False):
            unselected_columns = [col for col in columns if col not in selected_columns]
            result_df = result_df.drop(unselected_columns, axis=1)
        
        return {
            'data': result_df,
            'metrics': {
                'selected_features': selected_columns,
                'scores': selector.scores_.tolist() if hasattr(selector, 'scores_') else None
            },
            'columns_removed': [col for col in columns if col not in selected_columns] 
                              if params.get('drop_unselected', False) else []
        }
    
    def _apply_datetime_features(self, df, params):
        """Extract features from datetime columns"""
        columns = params.get('columns')
        features = params.get('features', ['year', 'month', 'day', 'dayofweek'])
        
        # If no columns specified, try to identify datetime columns
        if not columns:
            columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            # Try to convert string columns to datetime
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    pd.to_datetime(df[col])
                    columns.append(col)
                except:
                    pass
        
        if not columns:
            return {'data': df, 'metrics': {'datetime_features_added': 0}, 'columns_added': []}
        
        # Create a copy of the dataframe
        result_df = df.copy()
        added_columns = []
        
        # Process each datetime column
        for col in columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(result_df[col]):
                try:
                    result_df[col] = pd.to_datetime(result_df[col])
                except:
                    continue
            
            # Extract datetime features
            prefix = f"{col}_"
            if 'year' in features:
                col_name = f"{prefix}year"
                result_df[col_name] = result_df[col].dt.year
                added_columns.append(col_name)
            
            if 'month' in features:
                col_name = f"{prefix}month"
                result_df[col_name] = result_df[col].dt.month
                added_columns.append(col_name)
            
            if 'day' in features:
                col_name = f"{prefix}day"
                result_df[col_name] = result_df[col].dt.day
                added_columns.append(col_name)
            
            if 'dayofweek' in features:
                col_name = f"{prefix}dayofweek"
                result_df[col_name] = result_df[col].dt.dayofweek
                added_columns.append(col_name)
            
            if 'hour' in features:
                col_name = f"{prefix}hour"
                result_df[col_name] = result_df[col].dt.hour
                added_columns.append(col_name)
            
            if 'quarter' in features:
                col_name = f"{prefix}quarter"
                result_df[col_name] = result_df[col].dt.quarter
                added_columns.append(col_name)
            
            if 'is_weekend' in features:
                col_name = f"{prefix}is_weekend"
                result_df[col_name] = (result_df[col].dt.dayofweek >= 5).astype(int)
                added_columns.append(col_name)
        
        return {
            'data': result_df,
            'metrics': {'datetime_features_added': len(added_columns)},
            'columns_added': added_columns
        }
    
    def _apply_log_transform(self, df, params):
        """Apply logarithmic transformation to numeric columns"""
        columns = params.get('columns')
        base = params.get('base', 'natural')  # 'natural', '10', or '2'
        
        # If no columns specified, use all positive numeric columns
        if not columns:
            numeric_df = df.select_dtypes(include=['number'])
            columns = [col for col in numeric_df.columns if (numeric_df[col] > 0).all()]
        
        if not columns:
            return {'data': df, 'metrics': {'columns_transformed': 0}, 'columns_added': []}
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Apply log transformation based on specified base
        for col in columns:
            # Skip columns with non-positive values
            if (df[col] <= 0).any():
                continue
                
            if base == 'natural':
                result_df[f"{col}_log"] = np.log(df[col])
            elif base == '10':
                result_df[f"{col}_log10"] = np.log10(df[col])
            elif base == '2':
                result_df[f"{col}_log2"] = np.log2(df[col])
        
        # Get list of added columns
        added_columns = [f"{col}_log" if base == 'natural' else
                         f"{col}_log10" if base == '10' else
                         f"{col}_log2" for col in columns 
                         if (df[col] > 0).all()]
        
        return {
            'data': result_df,
            'metrics': {'columns_transformed': len(added_columns)},
            'columns_added': added_columns
        }
    
    def _apply_polynomial_features(self, df, params):
        """Add polynomial features for numeric columns"""
        from sklearn.preprocessing import PolynomialFeatures
        
        columns = params.get('columns')
        degree = params.get('degree', 2)
        interaction_only = params.get('interaction_only', False)
        
        # If no columns specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(columns) < 1:
            return {'data': df, 'metrics': {'features_added': 0}, 'columns_added': []}
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Apply polynomial transformation
        poly = PolynomialFeatures(degree=degree, 
                                  interaction_only=interaction_only, 
                                  include_bias=False)
        
        X = result_df[columns].fillna(0)  # Handle NaN values
        poly_features = poly.fit_transform(X)
        
        # Get feature names
        if hasattr(poly, 'get_feature_names_out'):
            feature_names = poly.get_feature_names_out(columns)
        else:
            feature_names = poly.get_feature_names(columns)
        
        # Remove original features which are included in poly_features
        feature_names = feature_names[len(columns):]
        poly_features = poly_features[:, len(columns):]
        
        # Add polynomial features to dataframe
        for i, name in enumerate(feature_names):
            result_df[f"poly_{name}"] = poly_features[:, i]
        
        added_columns = [f"poly_{name}" for name in feature_names]
        
        return {
            'data': result_df,
            'metrics': {'features_added': len(added_columns)},
            'columns_added': added_columns
        }
    
    def _apply_outlier_detection(self, df, params):
        """Detect and handle outliers in numeric data"""
        columns = params.get('columns')
        method = params.get('method', 'zscore')
        threshold = params.get('threshold', 3.0)
        handling = params.get('handling', 'flag')  # 'flag', 'remove', or 'clip'
        
        # If no columns specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if not columns:
            return {'data': df, 'metrics': {'outliers_detected': 0}, 'columns_added': []}
        
        # Create a copy of the dataframe
        result_df = df.copy()
        outliers_detected = 0
        
        # Process each column
        for col in columns:
            # Skip columns with NaN values
            if df[col].isnull().any():
                continue
                
            # Detect outliers based on method
            if method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold
            elif method == 'iqr':
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            # Count outliers
            outliers_count = outliers.sum()
            outliers_detected += outliers_count
            
            # Handle outliers
            if handling == 'flag':
                result_df[f"{col}_is_outlier"] = outliers.astype(int)
            elif handling == 'remove':
                # Replace outliers with NaN
                result_df.loc[outliers, col] = np.nan
            elif handling == 'clip':
                if method == 'zscore':
                    mean, std = df[col].mean(), df[col].std()
                    result_df[col] = result_df[col].clip(
                        lower=mean - threshold * std,
                        upper=mean + threshold * std
                    )
                elif method == 'iqr':
                    result_df[col] = result_df[col].clip(
                        lower=lower_bound,
                        upper=upper_bound
                    )
        
        # Get list of added columns (only for 'flag' handling)
        added_columns = [f"{col}_is_outlier" for col in columns] if handling == 'flag' else []
        
        return {
            'data': result_df,
            'metrics': {'outliers_detected': outliers_detected},
            'columns_added': added_columns
        }
    
    def _apply_binning(self, df, params):
        """Bin numeric values into discrete intervals"""
        columns = params.get('columns')
        method = params.get('method', 'equal_width')
        bins = params.get('bins', 10)
        labels = params.get('labels', None)
        
        # If no columns specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if not columns:
            return {'data': df, 'metrics': {'columns_binned': 0}, 'columns_added': []}
        
        # Create a copy of the dataframe
        result_df = df.copy()
        added_columns = []
        
        # Process each column
        for col in columns:
            binned_col = f"{col}_binned"
            
            # Apply binning based on method
            if method == 'equal_width':
                result_df[binned_col] = pd.cut(
                    df[col], bins=bins, labels=labels,
                    include_lowest=True, duplicates='drop'
                )
            elif method == 'equal_freq':
                result_df[binned_col] = pd.qcut(
                    df[col], q=bins, labels=labels,
                    duplicates='drop'
                )
            elif method == 'custom':
                bin_edges = params.get('bin_edges')
                if not bin_edges:
                    continue
                result_df[binned_col] = pd.cut(
                    df[col], bins=bin_edges, labels=labels,
                    include_lowest=True, duplicates='drop'
                )
            else:
                raise ValueError(f"Unknown binning method: {method}")
                
            added_columns.append(binned_col)
        
        return {
            'data': result_df,
            'metrics': {'columns_binned': len(added_columns)},
            'columns_added': added_columns
        }
    
    def _apply_clustering(self, df, params):
        """Add cluster membership as a feature"""
        columns = params.get('columns')
        method = params.get('method', 'kmeans')
        n_clusters = params.get('n_clusters', 3)
        
        # If no columns specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(columns) < 2:
            raise ValueError("At least 2 numeric columns required for clustering")
        
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Prepare data
        X = result_df[columns].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'dbscan':
            eps = params.get('eps', 0.5)
            min_samples = params.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Fit and predict clusters
        clusters = clusterer.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        cluster_col = f"cluster_{method}"
        result_df.loc[X.index, cluster_col] = clusters
        
        # Calculate clustering metrics if available
        metrics = {}
        if method == 'kmeans':
            metrics['inertia'] = clusterer.inertia_
        
        return {
            'data': result_df,
            'metrics': metrics,
            'columns_added': [cluster_col]
        }
