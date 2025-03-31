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
from datetime import datetime

class FeatureEngineer:
    """
    Provides operations for feature engineering on datasets
    """
    
    def __init__(self):
        """Initialize feature engineering operations"""
        # Dictionary of supported operations with their metadata
        self.supported_operations = {
            'filter_rows': {
                'name': 'Filter Rows',
                'description': 'Remove rows based on conditions',
                'parameters': {
                    'column': {'type': 'column', 'required': True},
                    'operation': {'type': 'select', 'options': ['>', '<', '==', '!=', '>=', '<=', 'contains', 'starts_with', 'ends_with', 'is_null', 'is_not_null']},
                    'value': {'type': 'text', 'required': False}
                }
            },
            'drop_columns': {
                'name': 'Drop Columns',
                'description': 'Remove columns from the dataset',
                'parameters': {
                    'columns': {'type': 'multicolumn', 'required': True}
                }
            },
            'impute_missing': {
                'name': 'Impute Missing Values',
                'description': 'Fill missing values using a strategy',
                'parameters': {
                    'columns': {'type': 'multicolumn', 'required': True},
                    'strategy': {'type': 'select', 'options': ['mean', 'median', 'most_frequent', 'constant']},
                    'constant_value': {'type': 'text', 'required': False}
                }
            },
            'scale_normalize': {
                'name': 'Scale/Normalize',
                'description': 'Scale numeric columns to a standard range',
                'parameters': {
                    'columns': {'type': 'multicolumn', 'required': True},
                    'method': {'type': 'select', 'options': ['standard', 'minmax', 'robust']}
                }
            },
            'one_hot_encode': {
                'name': 'One-Hot Encode',
                'description': 'Convert categorical columns to one-hot encoded columns',
                'parameters': {
                    'columns': {'type': 'multicolumn', 'required': True}
                }
            },
            'bin_numeric': {
                'name': 'Bin Numeric Values',
                'description': 'Convert numeric columns to categorical bins',
                'parameters': {
                    'column': {'type': 'column', 'required': True},
                    'num_bins': {'type': 'number', 'default': 5},
                    'strategy': {'type': 'select', 'options': ['uniform', 'quantile']}
                }
            },
            'log_transform': {
                'name': 'Log Transform',
                'description': 'Apply log transformation to numeric columns',
                'parameters': {
                    'columns': {'type': 'multicolumn', 'required': True},
                    'base': {'type': 'number', 'default': 'e'}
                }
            },
            'pca': {
                'name': 'Principal Component Analysis',
                'description': 'Reduce dimensionality using PCA',
                'parameters': {
                    'columns': {'type': 'multicolumn', 'required': True},
                    'n_components': {'type': 'number', 'default': 2}
                }
            },
            'create_datetime_features': {
                'name': 'Extract DateTime Features',
                'description': 'Extract components from datetime columns',
                'parameters': {
                    'column': {'type': 'column', 'required': True},
                    'components': {'type': 'multiselect', 'options': ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'quarter']}
                }
            },
            'text_extraction': {
                'name': 'Text Feature Extraction',
                'description': 'Extract features from text columns',
                'parameters': {
                    'column': {'type': 'column', 'required': True},
                    'extract': {'type': 'multiselect', 'options': ['length', 'word_count', 'uppercase_count', 'lowercase_count', 'digit_count', 'special_char_count']}
                }
            },
            'polynomial_features': {
                'name': 'Create Polynomial Features',
                'description': 'Generate polynomial features from numeric columns',
                'parameters': {
                    'columns': {'type': 'multicolumn', 'required': True},
                    'degree': {'type': 'number', 'default': 2}
                }
            }
        }
        
        # Storage directory for processed datasets
        self.output_dir = os.path.join('uploads')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def apply_operations(self, file_path, operations, output_path=None):
        """
        Apply a sequence of feature engineering operations to a dataset
        
        Parameters:
        -----------
        file_path : str
            Path to the input dataset
        operations : list
            List of operations to apply
        output_path : str, optional
            Path to save the processed dataset
            
        Returns:
        --------
        dict
            Results of the feature engineering process
        """
        try:
            # Load the dataset
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Store original shape
            original_shape = df.shape
            
            # Apply each operation in sequence
            operation_results = []
            for i, operation in enumerate(operations):
                op_type = operation['type']
                params = operation.get('params', {})
                
                if op_type not in self.supported_operations:
                    raise ValueError(f"Unsupported operation: {op_type}")
                
                # Apply the operation
                df, op_result = self._apply_operation(df, op_type, params)
                operation_results.append(op_result)
            
            # Generate output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.basename(file_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(self.output_dir, f"{name}_processed_{timestamp}{ext}")
            
            # Save the processed dataset
            if file_ext == '.csv':
                df.to_csv(output_path, index=False)
            elif file_ext in ['.xls', '.xlsx']:
                df.to_excel(output_path, index=False)
            elif file_ext == '.json':
                df.to_json(output_path, orient='records')
            
            # Generate data info for preview
            columns_info = []
            for col in df.columns:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'missing': int(df[col].isnull().sum()),
                    'missing_pct': float((df[col].isnull().sum() / len(df)) * 100)
                }
                
                # Add some basic stats for numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info.update({
                        'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                    })
                
                columns_info.append(col_info)
                
            # Return results
            return {
                'success': True,
                'output_file': output_path,
                'original_shape': {'rows': original_shape[0], 'columns': original_shape[1]},
                'final_shape': {'rows': df.shape[0], 'columns': df.shape[1]},
                'operations': operation_results,
                'data_info': {
                    'num_rows': df.shape[0],
                    'num_columns': df.shape[1],
                    'columns': columns_info
                }
            }
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def preview_operations(self, file_path, operations, max_rows=10):
        """
        Preview the results of applying operations without saving the dataset
        
        Parameters:
        -----------
        file_path : str
            Path to the input dataset
        operations : list
            List of operations to apply
        max_rows : int, optional
            Maximum number of rows to return in the preview
            
        Returns:
        --------
        dict
            Preview of the processed data
        """
        try:
            # Load the dataset
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Store original shape
            original_shape = df.shape
            
            # Apply each operation in sequence
            operation_results = []
            for i, operation in enumerate(operations):
                op_type = operation['type']
                params = operation.get('params', {})
                
                if op_type not in self.supported_operations:
                    raise ValueError(f"Unsupported operation: {op_type}")
                
                # Apply the operation
                df, op_result = self._apply_operation(df, op_type, params)
                operation_results.append(op_result)
            
            # Generate column information
            columns = []
            for col in df.columns:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'sample_values': df[col].dropna().head(3).tolist(),
                    'null_count': int(df[col].isna().sum()),
                }
                columns.append(col_info)
            
            # Convert dataframe to records for preview
            preview_rows = df.head(max_rows).replace({np.nan: None}).to_dict(orient='records')
            
            # Return preview data
            return {
                'success': True,
                'preview': {
                    'columns': columns,
                    'rows': preview_rows,
                    'total_rows': df.shape[0],
                    'original_shape': {'rows': original_shape[0], 'columns': original_shape[1]},
                    'final_shape': {'rows': df.shape[0], 'columns': df.shape[1]}
                },
                'operations': operation_results
            }
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _apply_operation(self, df, op_type, params):
        """
        Apply a single operation to the dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        op_type : str
            Type of operation to apply
        params : dict
            Parameters for the operation
            
        Returns:
        --------
        tuple
            (Modified dataframe, Operation result metadata)
        """
        result = {
            'type': op_type,
            'name': self.supported_operations[op_type]['name'],
            'params': params
        }
        
        original_shape = df.shape
        
        # Filter rows operation
        if op_type == 'filter_rows':
            column = params.get('column')
            operation = params.get('operation')
            value = params.get('value')
            
            if not column or column not in df.columns:
                raise ValueError(f"Invalid column for filter_rows: {column}")
            
            if not operation:
                raise ValueError("Operation is required for filter_rows")
            
            # Different filtering conditions
            if operation == 'is_null':
                filtered_df = df[df[column].isna()]
            elif operation == 'is_not_null':
                filtered_df = df[df[column].notna()]
            else:
                if value is None:
                    raise ValueError("Value is required for this filter operation")
                
                # Convert value to appropriate type based on column data type
                col_type = df[column].dtype
                if pd.api.types.is_numeric_dtype(col_type):
                    try:
                        value = float(value)
                    except:
                        raise ValueError(f"Invalid numeric value: {value}")
                        
                # Apply the filtering operation
                if operation == '>':
                    filtered_df = df[df[column] > value]
                elif operation == '<':
                    filtered_df = df[df[column] < value]
                elif operation == '==':
                    filtered_df = df[df[column] == value]
                elif operation == '!=':
                    filtered_df = df[df[column] != value]
                elif operation == '>=':
                    filtered_df = df[df[column] >= value]
                elif operation == '<=':
                    filtered_df = df[df[column] <= value]
                elif operation == 'contains':
                    filtered_df = df[df[column].astype(str).str.contains(str(value), na=False)]
                elif operation == 'starts_with':
                    filtered_df = df[df[column].astype(str).str.startswith(str(value), na=False)]
                elif operation == 'ends_with':
                    filtered_df = df[df[column].astype(str).str.endswith(str(value), na=False)]
                else:
                    raise ValueError(f"Unsupported filter operation: {operation}")
            
            result['rows_removed'] = len(df) - len(filtered_df)
            df = filtered_df
        
        # Drop columns operation
        elif op_type == 'drop_columns':
            columns = params.get('columns', [])
            
            if not columns:
                raise ValueError("No columns specified for drop_columns")
            
            # Filter to only include columns that exist in the dataframe
            columns_to_drop = [col for col in columns if col in df.columns]
            
            if not columns_to_drop:
                raise ValueError("None of the specified columns exist in the dataframe")
            
            df = df.drop(columns=columns_to_drop)
            result['columns_dropped'] = columns_to_drop
            
        # Impute missing values operation
        elif op_type == 'impute_missing':
            columns = params.get('columns', [])
            strategy = params.get('strategy', 'mean')
            
            if not columns:
                raise ValueError("No columns specified for impute_missing")
            
            # Validate columns exist in the dataframe
            columns_to_impute = [col for col in columns if col in df.columns]
            
            if not columns_to_impute:
                raise ValueError("None of the specified columns exist in the dataframe")
            
            # If strategy is constant, we need a constant value
            if strategy == 'constant':
                constant_value = params.get('constant_value')
                if constant_value is None:
                    raise ValueError("Constant value is required for constant imputation strategy")
                    
                # Try to convert to numeric if columns are numeric
                for col in columns_to_impute:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            constant_value = float(constant_value)
                        except:
                            raise ValueError(f"Invalid numeric value for constant imputation: {constant_value}")
                            
                    # Impute with constant value
                    df[col] = df[col].fillna(constant_value)
            else:
                # For other strategies, use SimpleImputer
                for col in columns_to_impute:
                    # Skip columns that are not appropriate for the chosen strategy
                    if strategy in ['mean', 'median'] and not pd.api.types.is_numeric_dtype(df[col]):
                        continue
                        
                    # Create and fit imputer
                    imputer = SimpleImputer(strategy=strategy)
                    df[col] = imputer.fit_transform(df[[col]])
            
            result['columns_imputed'] = columns_to_impute
            result['strategy'] = strategy
            
        # Scale/normalize operation
        elif op_type == 'scale_normalize':
            columns = params.get('columns', [])
            method = params.get('method', 'standard')
            
            if not columns:
                raise ValueError("No columns specified for scale_normalize")
            
            # Filter to numeric columns that exist in the dataframe
            numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_columns:
                raise ValueError("None of the specified columns are numeric and exist in the dataframe")
            
            # Choose scaler based on method
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {method}")
            
            # Apply scaling
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            
            result['columns_scaled'] = numeric_columns
            result['method'] = method
            
        # One-hot encode operation
        elif op_type == 'one_hot_encode':
            columns = params.get('columns', [])
            
            if not columns:
                raise ValueError("No columns specified for one_hot_encode")
            
            # Filter to categorical columns that exist in the dataframe
            cat_columns = [col for col in columns if col in df.columns]
            
            if not cat_columns:
                raise ValueError("None of the specified columns exist in the dataframe")
            
            # One-hot encode each column
            df_encoded = pd.get_dummies(df, columns=cat_columns, prefix=cat_columns)
            
            result['columns_encoded'] = cat_columns
            result['new_columns_count'] = len(df_encoded.columns) - len(df.columns)
            df = df_encoded
            
        # Bin numeric values operation
        elif op_type == 'bin_numeric':
            column = params.get('column')
            num_bins = int(params.get('num_bins', 5))
            strategy = params.get('strategy', 'uniform')
            
            if not column or column not in df.columns:
                raise ValueError(f"Invalid column for bin_numeric: {column}")
            
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Column {column} is not numeric and cannot be binned")
            
            # Create bins based on strategy
            if strategy == 'uniform':
                df[f"{column}_binned"] = pd.cut(df[column], bins=num_bins)
            elif strategy == 'quantile':
                df[f"{column}_binned"] = pd.qcut(df[column], q=num_bins, duplicates='drop')
            else:
                raise ValueError(f"Unsupported binning strategy: {strategy}")
            
            result['column_binned'] = column
            result['num_bins'] = num_bins
            result['strategy'] = strategy
            
        # Log transform operation
        elif op_type == 'log_transform':
            columns = params.get('columns', [])
            base = params.get('base', 'e')
            
            if not columns:
                raise ValueError("No columns specified for log_transform")
            
            # Filter to numeric columns that exist in the dataframe
            numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_columns:
                raise ValueError("None of the specified columns are numeric and exist in the dataframe")
            
            # Determine base
            if base == 'e':
                log_func = np.log
                prefix = 'ln'
            elif base == '10':
                log_func = np.log10
                prefix = 'log10'
            elif base == '2':
                log_func = np.log2
                prefix = 'log2'
            else:
                try:
                    base = float(base)
                    from math import log
                    log_func = lambda x: np.log(x) / np.log(base)
                    prefix = f"log{base}"
                except:
                    raise ValueError(f"Invalid log base: {base}")
            
            # Apply log transform (add 1 to avoid log(0))
            for col in numeric_columns:
                # Only transform columns with all positive values
                if (df[col] <= 0).any():
                    # Add minimum offset to make all values positive
                    min_val = df[col].min()
                    offset = abs(min_val) + 1 if min_val <= 0 else 0
                    df[f"{prefix}_{col}"] = log_func(df[col] + offset)
                else:
                    df[f"{prefix}_{col}"] = log_func(df[col])
            
            result['columns_transformed'] = numeric_columns
            result['base'] = base
            
        # PCA operation
        elif op_type == 'pca':
            columns = params.get('columns', [])
            n_components = int(params.get('n_components', 2))
            
            if not columns:
                raise ValueError("No columns specified for PCA")
            
            # Filter to numeric columns that exist in the dataframe
            numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_columns:
                raise ValueError("None of the specified columns are numeric and exist in the dataframe")
            
            if len(numeric_columns) < n_components:
                n_components = len(numeric_columns)
                
            # Handle missing values before PCA
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(df[numeric_columns])
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(X)
            
            # Add PCA components to dataframe
            for i in range(n_components):
                df[f"PC{i+1}"] = pca_result[:, i]
            
            result['columns_used'] = numeric_columns
            result['n_components'] = n_components
            result['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
            
        # Create datetime features operation
        elif op_type == 'create_datetime_features':
            column = params.get('column')
            components = params.get('components', ['year', 'month', 'day'])
            
            if not column or column not in df.columns:
                raise ValueError(f"Invalid column for create_datetime_features: {column}")
            
            # Convert to datetime if not already
            try:
                df[column] = pd.to_datetime(df[column])
            except:
                raise ValueError(f"Column {column} cannot be converted to datetime")
            
            # Extract requested components
            component_map = {
                'year': ('year', lambda x: x.dt.year),
                'month': ('month', lambda x: x.dt.month),
                'day': ('day', lambda x: x.dt.day),
                'hour': ('hour', lambda x: x.dt.hour),
                'minute': ('minute', lambda x: x.dt.minute),
                'second': ('second', lambda x: x.dt.second),
                'dayofweek': ('dayofweek', lambda x: x.dt.dayofweek),
                'quarter': ('quarter', lambda x: x.dt.quarter)
            }
            
            added_columns = []
            for component in components:
                if component in component_map:
                    suffix, func = component_map[component]
                    new_col = f"{column}_{suffix}"
                    df[new_col] = func(df[column])
                    added_columns.append(new_col)
            
            result['column'] = column
            result['components_added'] = added_columns
            
        # Text feature extraction operation
        elif op_type == 'text_extraction':
            column = params.get('column')
            extract = params.get('extract', ['length', 'word_count'])
            
            if not column or column not in df.columns:
                raise ValueError(f"Invalid column for text_extraction: {column}")
            
            # Convert column to string type
            df[column] = df[column].astype(str)
            
            # Extract requested features
            feature_map = {
                'length': ('length', lambda x: x.str.len()),
                'word_count': ('word_count', lambda x: x.str.split().str.len()),
                'uppercase_count': ('uppercase_count', lambda x: x.str.count(r'[A-Z]')),
                'lowercase_count': ('lowercase_count', lambda x: x.str.count(r'[a-z]')),
                'digit_count': ('digit_count', lambda x: x.str.count(r'[0-9]')),
                'special_char_count': ('special_char_count', lambda x: x.str.count(r'[^\w\s]'))
            }
            
            added_columns = []
            for feature in extract:
                if feature in feature_map:
                    suffix, func = feature_map[feature]
                    new_col = f"{column}_{suffix}"
                    df[new_col] = func(df[column])
                    added_columns.append(new_col)
            
            result['column'] = column
            result['features_added'] = added_columns
            
        # Polynomial features operation
        elif op_type == 'polynomial_features':
            columns = params.get('columns', [])
            degree = int(params.get('degree', 2))
            
            if not columns:
                raise ValueError("No columns specified for polynomial_features")
            
            # Filter to numeric columns that exist in the dataframe
            numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
            if not numeric_columns:
                raise ValueError("None of the specified columns are numeric and exist in the dataframe")
            
            # Generate polynomial features (interactions)
            from itertools import combinations_with_replacement
            
            added_columns = []
            for combo in combinations_with_replacement(numeric_columns, degree):
                # Create new column name
                if len(set(combo)) == 1:
                    # For powers of the same feature (e.g., x^2)
                    new_col = f"{combo[0]}^{degree}"
                else:
                    # For interactions (e.g., x*y)
                    new_col = "*".join(combo)
                
                # Create the polynomial feature
                df[new_col] = 1.0
                for col in combo:
                    df[new_col] *= df[col]
                    
                added_columns.append(new_col)
            
            result['columns_used'] = numeric_columns
            result['degree'] = degree
            result['features_added'] = added_columns
        
        # Add more operations as needed...
        
        result['rows_before'] = original_shape[0]
        result['rows_after'] = df.shape[0]
        result['columns_before'] = original_shape[1]
        result['columns_after'] = df.shape[1]
        
        return df, result
