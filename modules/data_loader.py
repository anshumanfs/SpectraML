import pandas as pd
import numpy as np
import json
import os
import csv

class DataLoader:
    """
    Handles loading data from various file formats and provides
    information about the data structure.
    """
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'json']
    
    def load_data(self, file_path, file_type=None, **kwargs):
        """
        Load data from a file
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        file_type : str, optional
            Type of file (inferred from extension if not provided)
        **kwargs : dict
            Additional parameters for data loading:
            - csv_delimiter: str, delimiter for CSV files (default: ',')
            - csv_quotechar: str, character for quoting fields (default: '"')
            - skip_bad_lines: bool, skip lines with parsing errors (default: False)
            - max_rows: int, maximum number of rows to load
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        if not file_type:
            file_type = os.path.splitext(file_path)[1].lstrip('.').lower()
        
        try:
            if file_type == 'csv':
                # Extract CSV-specific parameters
                csv_delimiter = kwargs.get('csv_delimiter', ',')
                csv_quotechar = kwargs.get('csv_quotechar', '"')
                skip_bad_lines = kwargs.get('skip_bad_lines', False)
                max_rows = kwargs.get('max_rows', None)
                
                # Detect if we should skip bad lines based on error
                try:
                    # First attempt standard parsing
                    df = pd.read_csv(
                        file_path,
                        delimiter=csv_delimiter,
                        quotechar=csv_quotechar,
                        nrows=max_rows
                    )
                except pd.errors.ParserError as e:
                    print(f"CSV parsing error: {str(e)}")
                    
                    if skip_bad_lines or kwargs.get('auto_detect_issues', True):
                        print("Attempting to load with error handling options...")
                        
                        # Enable error skipping and try different dialect detection
                        df = pd.read_csv(
                            file_path,
                            delimiter=csv_delimiter,
                            quotechar=csv_quotechar,
                            error_bad_lines=False,  # Don't raise error on bad lines
                            warn_bad_lines=True,    # Warn about bad lines
                            on_bad_lines='skip',    # Skip bad lines (for pandas >= 1.3)
                            low_memory=False,       # Better for inconsistent data types
                            nrows=max_rows
                        )
                        
                        print(f"Successfully loaded CSV with error handling ({len(df)} rows)")
                    else:
                        # If we detect CSV dialect issues, let's try to auto-detect delimiter
                        with open(file_path, 'r', newline='', errors='replace') as f:
                            sample = f.read(4096)  # Read a sample to detect dialect
                            
                        dialect = csv.Sniffer().sniff(sample)
                        print(f"Detected CSV dialect: delimiter='{dialect.delimiter}', quotechar='{dialect.quotechar}'")
                        
                        df = pd.read_csv(
                            file_path,
                            delimiter=dialect.delimiter,
                            quotechar=dialect.quotechar,
                            nrows=max_rows
                        )
                        print(f"Successfully loaded CSV with auto-detected dialect ({len(df)} rows)")
                        
            elif file_type in ['xls', 'xlsx']:
                max_rows = kwargs.get('max_rows', None)
                df = pd.read_excel(file_path, nrows=max_rows)
            elif file_type == 'json':
                df = pd.read_json(file_path)
                # Apply max_rows limit if specified
                if kwargs.get('max_rows'):
                    df = df.head(kwargs.get('max_rows'))
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
    
    def get_data_info(self, file_path, file_type=None, **kwargs):
        """
        Get information about the data in the file
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        file_type : str, optional
            Type of file (csv, xlsx, json). If None, inferred from extension
        **kwargs : dict
            Additional parameters for data loading, see load_data method
            
        Returns:
        --------
        dict
            Information about the data
        """
        # Pass all kwargs to load_data for flexible CSV handling
        kwargs['skip_bad_lines'] = kwargs.get('skip_bad_lines', True)  # Default to skipping bad lines for info
        df = self.load_data(file_path, file_type, **kwargs)
        
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
    
    def get_preview_data(self, file_path, file_type, max_rows=10, **kwargs):
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
        **kwargs : dict
            Additional parameters for data loading, see load_data method
            
        Returns:
        --------
        dict
            Preview data and summary information
        """
        try:
            # Set default options for preview
            kwargs['max_rows'] = max_rows
            kwargs['skip_bad_lines'] = kwargs.get('skip_bad_lines', True)  # Default to skipping bad lines for preview
            
            # Use the enhanced load_data method with all parameters
            df = self.load_data(file_path, file_type, **kwargs)
            
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
