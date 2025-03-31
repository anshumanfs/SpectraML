import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import json
import os
from scipy import stats

class Visualizer:
    """
    Provides visualization capabilities for different types of data.
    """
    
    def __init__(self):
        # Basic visualization types
        self.basic_viz_types = {
            'scatter': 'Scatter Plot',
            'line': 'Line Chart',
            'bar': 'Bar Chart',
            'histogram': 'Histogram',
            'box': 'Box Plot',
            'pie': 'Pie Chart',
        }
        
        # Advanced visualization types
        self.advanced_viz_types = {
            'violin': 'Violin Plot',
            'heatmap': 'Heatmap',
            'scatter_3d': '3D Scatter Plot',
            'surface_3d': '3D Surface Plot',
            'contour': 'Contour Plot',
            'density_contour': 'Density Contour',
            'parallel_coordinates': 'Parallel Coordinates',
            'parallel_categories': 'Parallel Categories',
            'scatter_matrix': 'Scatter Plot Matrix',
            'sunburst': 'Sunburst Chart',
            'treemap': 'Treemap',
            'funnel': 'Funnel Chart',
            'indicator': 'Indicator/Gauge',
            'sankey': 'Sankey Diagram',
            'candlestick': 'Candlestick Chart',
            'ohlc': 'OHLC Chart',
            'radar': 'Radar/Polar Chart',
            'choropleth': 'Choropleth Map',
        }
        
        # Statistical visualization types
        self.statistical_viz_types = {
            'correlation': 'Correlation Matrix',
            'distribution': 'Distribution Plot',
            'qq_plot': 'Q-Q Plot',
            'residual_plot': 'Residual Plot',
            'pair_plot': 'Pair Plot',
            'time_series_decomposition': 'Time Series Decomposition',
            'cluster_map': 'Cluster Map',
        }
        
        # All visualization types combined
        self.viz_types = {
            **self.basic_viz_types,
            **self.advanced_viz_types,
            **self.statistical_viz_types
        }
    
    def create_visualization(self, file_path, viz_type, params=None):
        """
        Create a visualization for the data in the file
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        viz_type : str
            Type of visualization to create
        params : dict, optional
            Additional parameters for the visualization
            
        Returns:
        --------
        dict
            Visualization data and metadata
        """
        if params is None:
            params = {}
        
        # Load data based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Print dataframe info for debugging
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame data types:\n{df.dtypes}")
        
        try:
            # Create visualization based on type
            if viz_type in self.basic_viz_types:
                fig = self.create_basic_viz(df, viz_type, params)
            elif viz_type in self.advanced_viz_types:
                fig = self.create_advanced_viz(df, viz_type, params)
            elif viz_type in self.statistical_viz_types:
                fig = self.create_statistical_viz(df, viz_type, params)
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
            
            # Ensure figure has proper data structure
            if not hasattr(fig, 'to_dict'):
                raise ValueError("Invalid figure object - missing to_dict method")
            
            # Convert Plotly figure to JSON for the frontend
            fig_dict = fig.to_dict()
            
            # Verify the figure dictionary has the expected structure
            if 'data' not in fig_dict or not isinstance(fig_dict['data'], list):
                # Create a minimal valid plotly figure if structure is incorrect
                fig_dict = {
                    'data': [],
                    'layout': {'title': {'text': 'Empty Plot'}}
                }
            
            # Check if the data array contains actual traces with data
            if not fig_dict['data'] or all(not trace.get('x') for trace in fig_dict['data']):
                print("Warning: Plotly figure has no data traces")
                # Add a warning annotation to the plot
                if 'annotations' not in fig_dict['layout']:
                    fig_dict['layout']['annotations'] = []
                
                fig_dict['layout']['annotations'].append({
                    'text': 'No data available for the selected parameters',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 14, 'color': 'red'}
                })
            
            # Log the figure structure for debugging
            print(f"Figure structure: {list(fig_dict.keys())}")
            print(f"Number of data traces: {len(fig_dict['data'])}")
            for i, trace in enumerate(fig_dict['data']):
                data_length = len(trace.get('x', [])) if 'x' in trace else 'N/A'
                print(f"Trace {i}: Type={trace.get('type')}, Points={data_length}")
            
            # Properly serialize plot data for frontend - return as a string
            plot_json = json.dumps(fig_dict, cls=PlotlyJSONEncoder)
            
            return {
                'success': True,
                'plot': plot_json,  # Return as JSON string, not parsed object
                'type': viz_type,
                'title': params.get('title', f"{viz_type.replace('_', ' ').title()} Visualization")
            }
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            # Create a basic error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(
                title=f"Error in {viz_type} visualization",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            
            # Convert error figure to JSON
            fig_dict = fig.to_dict()
            plot_json = json.dumps(fig_dict, cls=PlotlyJSONEncoder)
            
            return {
                'success': True,
                'plot': plot_json,
                'type': viz_type,
                'title': f"Error: {viz_type.replace('_', ' ').title()} Visualization"
            }
    
    def _preprocess_dataframe(self, df, columns=None, numeric_only=False):
        """
        Preprocess dataframe for visualization to ensure compatible column types
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        columns : list, optional
            Specific columns to use (if None, use all columns)
        numeric_only : bool, optional
            If True, keep only numeric columns
            
        Returns:
        --------
        pandas.DataFrame
            Processed dataframe with compatible column types
        """
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Filter to specific columns if provided
        if columns is not None:
            # Ensure all specified columns exist
            existing_columns = [col for col in columns if col in processed_df.columns]
            processed_df = processed_df[existing_columns]
        
        # If numeric only, filter to numeric columns
        if numeric_only:
            processed_df = processed_df.select_dtypes(include=['number'])
        
        # For visualizations that require uniform types, convert non-numeric to string
        # This ensures all columns are treated as either numeric or categorical
        for col in processed_df.columns:
            if not pd.api.types.is_numeric_dtype(processed_df[col]):
                processed_df[col] = processed_df[col].astype(str)
        
        return processed_df
    
    def create_basic_viz(self, df, viz_type, params):
        """Create basic visualizations"""
        x = params.get('x')
        y = params.get('y')
        color = params.get('color')
        title = params.get('title', f"{viz_type.replace('_', ' ').title()} Chart")
        use_index = params.get('use_index', False)
        
        # Check if dataframe is empty
        if df.empty:
            raise ValueError("Dataset is empty - cannot create visualization")
        
        # Create a copy of the DataFrame with just the needed columns for better debugging
        plot_df = df.copy()
        
        # Convert DataFrame data to appropriate types
        # Ensure numeric values are actually numeric - pandas sometimes loads them as strings
        for col in plot_df.columns:
            if col in [x, y] and pd.api.types.is_object_dtype(plot_df[col]):
                try:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                    print(f"Converted column '{col}' to numeric type")
                except:
                    print(f"Could not convert column '{col}' to numeric type")
        
        # Special handling for single-column scatter plots or line charts using index
        if (viz_type in ['scatter', 'line']) and use_index and (x or y) and not (x and y):
            # Determine which column to use as y-value (x will be the index)
            column = y if y else x
            
            # Validate column exists
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")
            
            # Log the data type of the column being visualized
            print(f"Column '{column}' data type: {df[column].dtype}")
            
            # Drop rows where the column has NaN values
            plot_df = plot_df.dropna(subset=[column])
            
            # Ensure there's still data after dropping NaN values
            if plot_df.empty:
                raise ValueError("No valid data points after removing NaN values from selected column")
            
            # Print a small sample of the data for debugging
            print(f"Sample data for index {viz_type} visualization:\n{plot_df.head(3)}")
            
            # Create plot with index as x-axis
            plot_df = plot_df.reset_index()  # Convert index to column
            
            # Create a direct trace to ensure data is properly formatted
            fig = go.Figure()
            
            # Choose the appropriate mode based on the visualization type
            mode = 'markers' if viz_type == 'scatter' else 'lines'
            
            # Use a more descriptive name for the trace
            trace_name = f"{column} vs Row Index" if viz_type == 'scatter' else column
            
            fig.add_trace(
                go.Scatter(
                    x=plot_df['index'].tolist(),
                    y=plot_df[column].tolist(),
                    mode=mode,
                    name=trace_name
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title='Row Index',
                yaxis_title=column,
                template='plotly_white'  # Use a cleaner template
            )
            
            return fig
        
        # Regular validation for x and y columns for other chart types
        if not viz_type == 'pie':
            if x and x not in df.columns:
                raise ValueError(f"Column '{x}' not found in dataset")
            if y and y not in df.columns:
                raise ValueError(f"Column '{y}' not found in dataset")
            
            # Add validation for color column
            if color and color not in df.columns:
                print(f"Warning: Color column '{color}' not found in dataset. Ignoring color parameter.")
                color = None
        
        # Log the data types of the columns being visualized
        if x:
            print(f"X column '{x}' data type: {df[x].dtype}")
        if y:
            print(f"Y column '{y}' data type: {df[y].dtype}")
        
        # Drop rows where the key columns (x, y) have NaN values
        if x:
            plot_df = plot_df.dropna(subset=[x])
        if y:
            plot_df = plot_df.dropna(subset=[y])
        
        # Ensure there's still data after dropping NaN values
        if plot_df.empty:
            raise ValueError("No valid data points after removing NaN values from selected columns")
        
        # Print a small sample of the data for debugging
        print(f"Sample data for visualization:\n{plot_df.head(3)}")
        
        try:
            # Direct method for histograms - make more robust
            if viz_type == 'histogram':
                # Check that we have a valid column for histogram
                if not x and not y:
                    raise ValueError("Histogram requires at least one column selection (X or Y)")
                
                # Choose the column to use - prefer X, fall back to Y
                column = x if x else y
                orientation = 'v' if x else 'h'  # Vertical if x-axis used, horizontal if y-axis
                
                # Direct method for histograms
                fig = go.Figure()
                
                # Create histogram with proper orientation
                if orientation == 'v':
                    fig.add_trace(
                        go.Histogram(
                            x=plot_df[column].tolist(),
                            name=column,
                            autobinx=True
                        )
                    )
                    fig.update_layout(
                        title=title,
                        xaxis_title=column,
                        yaxis_title='Count',
                        bargap=0.1
                    )
                else:
                    fig.add_trace(
                        go.Histogram(
                            y=plot_df[column].tolist(),
                            name=column,
                            autobiny=True
                        )
                    )
                    fig.update_layout(
                        title=title,
                        xaxis_title='Count',
                        yaxis_title=column,
                        bargap=0.1
                    )
                
                return fig
            
            # ...existing code for other visualization types...
        except Exception as e:
            # Create a fallback error figure
            print(f"Error creating {viz_type} plot with direct method: {e}")
            print(f"Error details: {str(e)}")
            
            # Create a basic error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating {viz_type} plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(
                title=f"Error in {viz_type} visualization",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
        
        # Debug the created figure to ensure it has data
        fig_dict = fig.to_dict()
        print(f"Figure data trace count: {len(fig_dict.get('data', []))}")
        for i, trace in enumerate(fig_dict.get('data', [])):
            x_data = trace.get('x', [])
            y_data = trace.get('y', [])
            trace_type = trace.get('type', 'unknown')
            print(f"Trace {i}: Type={trace_type}, X points={len(x_data) if x_data else 'None'}, Y points={len(y_data) if y_data else 'None'}")
            
            # Special handling for traces that may not have x/y (like pie charts)
            if trace_type == 'pie':
                values = trace.get('values', [])
                labels = trace.get('labels', [])
                print(f"  Pie chart: Values={len(values) if values else 'None'}, Labels={len(labels) if labels else 'None'}")
        
        return fig
    
    def create_advanced_viz(self, df, viz_type, params):
        """Create advanced visualizations"""
        x = params.get('x')
        y = params.get('y')
        z = params.get('z')
        color = params.get('color')
        title = params.get('title', f"{viz_type.replace('_', ' ').title()} Chart")
        
        # Special preprocessing for visualizations with specific requirements
        if viz_type in ['parallel_coordinates', 'scatter_matrix']:
            # These visualizations need uniform column types
            dimensions = params.get('dimensions')
            if dimensions:
                df = self._preprocess_dataframe(df, columns=dimensions)
            else:
                # Default to first 5 numeric columns if none specified
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) >= 2:
                    df = numeric_df.iloc[:, :min(5, len(numeric_df.columns))]
                else:
                    raise ValueError(f"Insufficient numeric columns for {viz_type}")
        
        # Now continue with existing code for creating the visualization
        if viz_type == 'violin':
            fig = px.violin(df, x=x, y=y, color=color, box=True, title=title)
        elif viz_type == 'heatmap':
            # For heatmap, we need a pivot table or correlation matrix
            if params.get('corr', False):
                # Ensure numeric columns for correlation
                corr_df = df.select_dtypes(include=['number'])
                specified_columns = params.get('columns')
                if specified_columns:
                    # Filter to only numeric columns from the specified list
                    cols = [col for col in specified_columns if col in corr_df.columns]
                    if len(cols) < 2:
                        raise ValueError("Insufficient numeric columns for correlation heatmap")
                    corr_df = corr_df[cols]
                corr_matrix = corr_df.corr()
                fig = px.imshow(corr_matrix, title=title)
            else:
                pivot_index = params.get('pivot_index')
                pivot_columns = params.get('pivot_columns')
                pivot_values = params.get('pivot_values')
                
                if not (pivot_index and pivot_columns and pivot_values):
                    raise ValueError("Pivot table requires index, columns, and values parameters")
                    
                # Create pivot table
                pivot_data = df.pivot_table(
                    index=pivot_index, 
                    columns=pivot_columns, 
                    values=pivot_values,
                    aggfunc='mean'  # Use mean as default aggregation
                )
                fig = px.imshow(pivot_data, title=title)
        # ...existing code for other visualizations...
        
        return fig
    
    def create_statistical_viz(self, df, viz_type, params):
        """Create statistical visualizations"""
        title = params.get('title', f"{viz_type.replace('_', ' ').title()} Chart")
        
        if viz_type == 'correlation':
            # Calculate correlation matrix - ensure numeric data
            numeric_df = df.select_dtypes(include=['number'])
            
            # Check if we have enough numeric columns
            if len(numeric_df.columns) < 2:
                raise ValueError("Insufficient numeric columns for correlation matrix")
            
            # Use specified columns if provided and they are numeric
            specified_columns = params.get('columns')
            if specified_columns:
                # Filter to only numeric columns from the specified list
                cols = [col for col in specified_columns if col in numeric_df.columns]
                if len(cols) < 2:
                    raise ValueError("Insufficient numeric columns in the specified list")
                corr_df = numeric_df[cols].corr()
            else:
                corr_df = numeric_df.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_df,
                text_auto=params.get('show_values', True),
                color_continuous_scale=params.get('colorscale', 'RdBu_r'),
                title=title
            )
        # ...existing code for other statistical visualizations...
        
        return fig
    
    def get_visualization_types(self):
        """Get all available visualization types"""
        return {
            'basic': self.basic_viz_types,
            'advanced': self.advanced_viz_types,
            'statistical': self.statistical_viz_types
        }
    
    def get_visualization_options(self, viz_type):
        """Get options for a specific visualization type"""
        # Common options for most visualizations
        common_options = {
            'title': {
                'type': 'text',
                'label': 'Chart Title',
                'default': f'{viz_type.replace("_", " ").title()} Chart'
            }
        }
        
        # Basic viz options
        if viz_type == 'scatter':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'X-Axis', 'optional': True},
                'y': {'type': 'column', 'label': 'Y-Axis', 'optional': True},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True},
                'use_index': {'type': 'boolean', 'label': 'Use Index for Single Column Plot', 'default': False}
            }
        elif viz_type in ['line', 'bar']:
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'X-Axis'},
                'y': {'type': 'column', 'label': 'Y-Axis'},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True}
            }
        elif viz_type == 'histogram':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'Values'},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True},
                'bin_size': {'type': 'number', 'label': 'Bin Size', 'optional': True}
            }
        elif viz_type == 'box':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'Category'},
                'y': {'type': 'column', 'label': 'Values'},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True}
            }
        elif viz_type == 'pie':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'Names'},
                'y': {'type': 'column', 'label': 'Values'}
            }
        
        # Advanced viz options
        elif viz_type == 'violin':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'Category'},
                'y': {'type': 'column', 'label': 'Values'},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True}
            }
        elif viz_type == 'heatmap':
            return {
                **common_options,
                'corr': {'type': 'boolean', 'label': 'Use Correlation Matrix', 'default': True},
                'columns': {'type': 'multicolumn', 'label': 'Columns for Correlation', 'optional': True},
                'pivot_index': {'type': 'column', 'label': 'Pivot Table Index', 'optional': True},
                'pivot_columns': {'type': 'column', 'label': 'Pivot Table Columns', 'optional': True},
                'pivot_values': {'type': 'column', 'label': 'Pivot Table Values', 'optional': True}
            }
        elif viz_type == 'scatter_3d':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'X-Axis'},
                'y': {'type': 'column', 'label': 'Y-Axis'},
                'z': {'type': 'column', 'label': 'Z-Axis'},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True}
            }
        elif viz_type == 'parallel_coordinates':
            return {
                **common_options,
                'color': {'type': 'column', 'label': 'Color By', 'optional': True},
                'dimensions': {'type': 'multicolumn', 'label': 'Dimensions', 'optional': True}
            }
        
        # Statistical viz options
        elif viz_type == 'correlation':
            return {
                **common_options,
                'columns': {'type': 'multicolumn', 'label': 'Columns', 'optional': True},
                'show_values': {'type': 'boolean', 'label': 'Show Values', 'default': True},
                'colorscale': {'type': 'select', 'label': 'Color Scale', 'options': ['RdBu_r', 'Viridis', 'Plasma', 'Cividis']}
            }
        elif viz_type == 'distribution':
            return {
                **common_options,
                'column': {'type': 'column', 'label': 'Column'},
                'bin_size': {'type': 'number', 'label': 'Bin Size', 'optional': True}
            }
        elif viz_type == 'qq_plot':
            return {
                **common_options,
                'column': {'type': 'column', 'label': 'Column'}
            }
        
        # Default options if specific type not handled
        return common_options


class PlotlyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Plotly figures"""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DatetimeIndex):
            return obj.astype(str).tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):  # Handle NaN/None values
            return None
        return json.JSONEncoder.default(self, obj)
