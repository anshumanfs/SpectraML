"""
Schemas for feature engineering operations.
This file provides structured parameter definitions for operations to ensure 
consistent validation between frontend and backend.
"""

OPERATION_SCHEMAS = {
    'filter_rows': {
        'name': 'Filter Rows',
        'description': 'Filter dataset rows based on conditions',
        'parameters': {
            'column': {
                'type': 'column',
                'required': True,
                'description': 'Column to filter on'
            },
            'operation': {
                'type': 'select',
                'options': ['>', '<', '==', '!=', '>=', '<=', 'contains', 'starts_with', 'ends_with', 'is_null', 'is_not_null'],
                'required': True,
                'description': 'Comparison operation to apply'
            },
            'value': {
                'type': 'text',
                'required': False,
                'description': 'Value to compare against (not needed for is_null/is_not_null)'
            }
        }
    },
    
    'drop_columns': {
        'name': 'Drop Columns',
        'description': 'Remove columns from the dataset',
        'parameters': {
            'columns': {
                'type': 'multicolumn',
                'required': True,
                'description': 'Columns to remove from dataset'
            }
        }
    },
    
    'impute_missing': {
        'name': 'Impute Missing Values',
        'description': 'Fill missing values using a strategy',
        'parameters': {
            'columns': {
                'type': 'multicolumn',
                'required': True,
                'description': 'Columns to impute missing values in'
            },
            'strategy': {
                'type': 'select',
                'options': ['mean', 'median', 'most_frequent', 'constant'],
                'required': True,
                'description': 'Method for filling missing values'
            },
            'constant_value': {
                'type': 'text',
                'required': False,
                'description': 'Value to use when strategy is "constant"'
            }
        }
    },
    
    'scale_normalize': {
        'name': 'Scale/Normalize',
        'description': 'Scale numeric columns to a standard range',
        'parameters': {
            'columns': {
                'type': 'multicolumn',
                'required': True,
                'description': 'Numeric columns to scale'
            },
            'method': {
                'type': 'select',
                'options': ['standard', 'minmax', 'robust'],
                'required': True,
                'description': 'Scaling method to apply'
            }
        }
    },
    
    'one_hot_encode': {
        'name': 'One-Hot Encode',
        'description': 'Convert categorical columns to one-hot encoded columns',
        'parameters': {
            'columns': {
                'type': 'multicolumn',
                'required': True,
                'description': 'Categorical columns to one-hot encode'
            }
        }
    },
    
    'bin_numeric': {
        'name': 'Bin Numeric Values',
        'description': 'Convert numeric columns to categorical bins',
        'parameters': {
            'column': {
                'type': 'column',
                'required': True,
                'description': 'Numeric column to bin'
            },
            'num_bins': {
                'type': 'number',
                'default': 5,
                'required': True,
                'description': 'Number of bins to create'
            },
            'strategy': {
                'type': 'select',
                'options': ['uniform', 'quantile'],
                'required': True,
                'description': 'Method to determine bin edges'
            }
        }
    },
    
    'log_transform': {
        'name': 'Log Transform',
        'description': 'Apply logarithmic transformation to numeric columns',
        'parameters': {
            'columns': {
                'type': 'multicolumn',
                'required': True,
                'description': 'Numeric columns to transform'
            },
            'base': {
                'type': 'select',
                'options': ['e', '2', '10'],
                'default': 'e',
                'required': True,
                'description': 'Logarithm base to use'
            }
        }
    },
    
    'pca': {
        'name': 'Principal Component Analysis',
        'description': 'Reduce dimensionality using PCA',
        'parameters': {
            'columns': {
                'type': 'multicolumn',
                'required': True,
                'description': 'Numeric columns to use for PCA'
            },
            'n_components': {
                'type': 'number',
                'default': 2,
                'required': True,
                'description': 'Number of principal components to extract'
            }
        }
    },
    
    'create_datetime_features': {
        'name': 'Extract DateTime Features',
        'description': 'Extract components from datetime columns',
        'parameters': {
            'column': {
                'type': 'column',
                'required': True,
                'description': 'Datetime column to extract features from'
            },
            'components': {
                'type': 'multiselect',
                'options': ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'quarter'],
                'required': True,
                'description': 'Datetime components to extract'
            }
        }
    },
    
    'text_extraction': {
        'name': 'Text Feature Extraction',
        'description': 'Extract features from text columns',
        'parameters': {
            'column': {
                'type': 'column',
                'required': True,
                'description': 'Text column to extract features from'
            },
            'extract': {
                'type': 'multiselect',
                'options': ['length', 'word_count', 'uppercase_count', 'lowercase_count', 'digit_count', 'special_char_count'],
                'required': True,
                'description': 'Text features to extract'
            }
        }
    },
    
    'polynomial_features': {
        'name': 'Create Polynomial Features',
        'description': 'Generate polynomial features from numeric columns',
        'parameters': {
            'columns': {
                'type': 'multicolumn',
                'required': True,
                'description': 'Numeric columns to use for generating polynomial features'
            },
            'degree': {
                'type': 'number',
                'default': 2,
                'required': True,
                'description': 'Polynomial degree'
            }
        }
    }
}

def get_operation_schema(operation_type):
    """
    Get schema for a specific operation type
    
    Args:
        operation_type (str): The type of operation
        
    Returns:
        dict: Schema definition or None if not found
    """
    return OPERATION_SCHEMAS.get(operation_type)

def validate_operation_params(operation_type, params):
    """
    Validate operation parameters against schema
    
    Args:
        operation_type (str): The type of operation
        params (dict): Parameters to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    schema = get_operation_schema(operation_type)
    if not schema:
        return False, f"Unknown operation type: {operation_type}"
    
    # Check required parameters
    for param_name, param_def in schema['parameters'].items():
        if param_def.get('required', False):
            if param_name not in params:
                return False, f"Missing required parameter '{param_name}' for operation '{operation_type}'"
            
            param_value = params[param_name]
            param_type = param_def.get('type')
            
            # Handle null values
            if param_value is None:
                return False, f"Parameter '{param_name}' cannot be null for operation '{operation_type}'"
            
            # Handle special case for multiselect/multicolumn
            if param_type in ['multicolumn', 'multiselect'] and isinstance(param_value, list) and len(param_value) == 0:
                return False, f"Parameter '{param_name}' must have at least one value for operation '{operation_type}'"
    
    # Special case for filter_rows
    if operation_type == 'filter_rows':
        operation = params.get('operation')
        value = params.get('value')
        
        if operation in ['is_null', 'is_not_null']:
            # Value not required for these operations
            pass
        elif not value and operation not in ['is_null', 'is_not_null']:
            return False, "Value is required for this filter operation"
    
    return True, None
