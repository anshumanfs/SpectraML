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
        Load data from a file
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        file_type : str, optional
            Type of file (inferred from extension if not provided)
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        if not file_type:
            file_type = os.path.splitext(file_path)[1].lstrip('.').lower()
        
        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type in ['xls', 'xlsx']:
                df = pd.read_excel(file_path)
            elif file_type == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Attempt to convert string columns that might actually be numeric
            # This helps with plotting numeric data that was loaded as strings
            for col in df.columns:
                # Only try to convert if it's an object/string type
                if df[col].dtype == 'object':
                    try:
                        # Try to convert to numeric, but keep non-numeric values as is
                        numeric_col = pd.to_numeric(df[col], errors='coerce')
                        
                        # If most values converted successfully, replace the column
                        if numeric_col.notna().sum() > 0.5 * len(numeric_col):
                            df[col] = numeric_col
                    except:
                        # Keep as is if conversion fails
                        pass
            
            # Print information about the loaded data
            print(f"Loaded dataframe from {file_path}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Data types:\n{df.dtypes}")
            
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
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
    
    def get_preview_data(self, file_path, file_type, max_rows=10):
        """
        Get a preview of the data from a file
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        file_type : str
            Type of the file
        max_rows : int, optional
            Maximum number of rows to return
            
        Returns:
        --------
        dict
            Preview data and summary information
        """
        try:
            # Handle different file types
            if file_type.lower() in ['csv', 'txt']:
                # Try different encodings
                try:
                    df = pd.read_csv(file_path, nrows=max_rows)
                except UnicodeDecodeError:
                    # Try with latin1 encoding for non-UTF8 files
                    df = pd.read_csv(file_path, encoding='latin1', nrows=max_rows)
            elif file_type.lower() in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, nrows=max_rows)
            elif file_type.lower() == 'json':
                # For JSON, read the whole file then slice
                df = pd.read_json(file_path)
                df = df.head(max_rows)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Get column information
            columns = []
            for col in df.columns:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'sample_values': df[col].dropna().head(3).tolist(),
                    'null_count': int(df[col].isna().sum()),
                }
                columns.append(col_info)
            
            # Convert dataframe to records for JSON serialization
            records = df.replace({np.nan: None}).to_dict(orient='records')
            
            return {
                'rows': records,
                'columns': columns,
                'total_rows': len(df),
                'total_columns': len(df.columns)
            }
        except Exception as e:
            raise Exception(f"Error getting data preview: {str(e)}")
