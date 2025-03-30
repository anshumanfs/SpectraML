import pandas as pd
import numpy as np
import json
import os

class DataLoader:
    """
    Handles loading data from various file formats and provides
    information about the data structure.
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'json']
    
    def load_data(self, file_path, file_type=None):
        """
        Load data from a file into a pandas DataFrame
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        file_type : str, optional
            Type of file (csv, xlsx, json). If None, inferred from extension
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_type is None:
            file_type = file_path.split('.')[-1].lower()
        
        if file_type not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_type}. Supported formats: {self.supported_formats}")
        
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'xlsx':
            return pd.read_excel(file_path)
        elif file_type == 'json':
            return pd.read_json(file_path)
    
    def get_data_info(self, file_path, file_type=None):
        """
        Get information about the data in the file
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        file_type : str, optional
            Type of file (csv, xlsx, json). If None, inferred from extension
            
        Returns:
        --------
        dict
            Information about the data
        """
        df = self.load_data(file_path, file_type)
        
        # Convert NumPy int64/float64 to Python native types to ensure JSON serialization
        def convert_to_native_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Get basic information
        info = {
            'num_rows': int(len(df)),  # Convert explicitly to Python int
            'num_columns': int(len(df.columns)),  # Convert explicitly to Python int
            'columns': [],
            'memory_usage': float(df.memory_usage(deep=True).sum() / (1024 * 1024)),  # Convert to Python float
            'missing_values': int(df.isnull().sum().sum()),  # Convert to Python int
            'missing_percentage': float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100)  # Convert to Python float
        }
        
        # Get column-specific information
        for col in df.columns:
            col_info = {
                'name': str(col),
                'dtype': str(df[col].dtype),
                'missing': int(df[col].isnull().sum()),  # Convert to Python int
                'missing_pct': float((df[col].isnull().sum() / len(df)) * 100)  # Convert to Python float
            }
            
            # Add numeric column stats
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'min': convert_to_native_types(df[col].min() if not pd.isna(df[col].min()) else None),
                    'max': convert_to_native_types(df[col].max() if not pd.isna(df[col].max()) else None),
                    'mean': convert_to_native_types(df[col].mean() if not pd.isna(df[col].mean()) else None),
                    'median': convert_to_native_types(df[col].median() if not pd.isna(df[col].median()) else None),
                    'std': convert_to_native_types(df[col].std() if not pd.isna(df[col].std()) else None)
                })
            
            # Add categorical column stats
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                value_counts = df[col].value_counts().to_dict()
                # Convert any non-serializable keys to strings and values to native Python types
                value_counts = {str(k): convert_to_native_types(v) for k, v in value_counts.items()}
                
                col_info.update({
                    'unique_values': int(df[col].nunique()),  # Convert to Python int
                    'top_values': dict(sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10])
                })
            
            info['columns'].append(col_info)
        
        return info
    
    def save_data(self, df, file_path, file_type=None):
        """
        Save DataFrame to a file
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data to save
        file_path : str
            Path to save the file
        file_type : str, optional
            Type of file (csv, xlsx, json). If None, inferred from extension
        """
        if file_type is None:
            file_type = file_path.split('.')[-1].lower()
        
        if file_type not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_type}. Supported formats: {self.supported_formats}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_type == 'csv':
            df.to_csv(file_path, index=False)
        elif file_type == 'xlsx':
            df.to_excel(file_path, index=False)
        elif file_type == 'json':
            df.to_json(file_path, orient='records')
        
        return file_path
