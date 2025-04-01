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
        try:
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
            
            # Auto-convert numeric columns stored as strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        mask = df[col].notna()
                        if mask.sum() > 0 and mask.sum() / len(df) > 0.5:  # Only convert if >50% converted successfully
                            print(f"Converted column '{col}' to numeric type")
                    except:
                        pass
        
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
            has_data = False
            for trace in fig_dict['data']:
                if trace.get('type') == 'pie' and trace.get('values') and len(trace.get('values')) > 0:
                    has_data = True
                    break
                elif trace.get('type') in ('heatmap', 'contour') and trace.get('z') and len(trace.get('z')) > 0:
                    has_data = True
                    break
                elif ((trace.get('x') and len(trace.get('x')) > 0) or 
                      (trace.get('y') and len(trace.get('y')) > 0)):
                    has_data = True
                    break
            
            if not has_data:
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
                'success': False,
                'error': str(e),
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
        title = params.get('title', f"{{viz_type.replace('_', ' ').title()}} Chart")
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
            # SCATTER PLOT
            if viz_type == 'scatter':
                if x and y:
                    fig = px.scatter(plot_df, x=x, y=y, color=color, title=title)
                else:
                    raise ValueError("Scatter plot requires both X and Y columns (or enable use_index option)")
                
            # LINE CHART
            elif viz_type == 'line':
                if x and y:
                    fig = px.line(plot_df, x=x, y=y, color=color, title=title)
                else:
                    raise ValueError("Line chart requires both X and Y columns (or enable use_index option)")
                
            # BAR CHART
            elif viz_type == 'bar':
                if x and y:
                    # Check if x column is categorical or numeric
                    is_x_categorical = not pd.api.types.is_numeric_dtype(plot_df[x]) or plot_df[x].nunique() < 20
                    
                    if is_x_categorical:
                        # Standard bar chart for categorical data
                        fig = px.bar(plot_df, x=x, y=y, color=color, title=title)
                    else:
                        # For numeric x-values, create a histogram-like bar chart
                        # Group by x value first
                        grouped = plot_df.groupby(x)[y].sum().reset_index()
                        fig = px.bar(grouped, x=x, y=y, title=title)
                else:
                    raise ValueError("Bar chart requires both X (categories) and Y (values) columns")
                    
            # HISTOGRAM
            elif viz_type == 'histogram':
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
                
            # BOX PLOT
            elif viz_type == 'box':
                if y:  # Box plot requires y axis (values)
                    # Use a more direct approach for box plots to ensure compatibility
                    if pd.api.types.is_numeric_dtype(plot_df[y]):
                        # Create box plot with optional grouping by x
                        if x and x in plot_df.columns:
                            # Group by x-axis column (categorical)
                            fig = go.Figure()
                            
                            # Calculate box plot stats for each group
                            groups = plot_df.groupby(x)[y].apply(list).to_dict()
                            
                            for group_name, values in groups.items():
                                fig.add_trace(go.Box(
                                    y=values,
                                    name=str(group_name),
                                    boxpoints='all',  # show all points
                                    jitter=0.3,
                                    pointpos=-1.8
                                ))
                                
                            fig.update_layout(
                                title=title,
                                yaxis_title=y,
                                xaxis_title=x,
                                boxmode='group'
                            )
                        else:
                            # Simple box plot without grouping
                            fig = go.Figure()
                            fig.add_trace(go.Box(
                                y=plot_df[y].dropna().tolist(),
                                name=y,
                                boxpoints='all',  # show all points
                                jitter=0.3,
                                pointpos=-1.8
                            ))
                            fig.update_layout(
                                title=title,
                                yaxis_title=y
                            )
                    else:
                        raise ValueError(f"Column '{y}' must be numeric for box plot")
                else:
                    raise ValueError("Box plot requires Y column (values to analyze)")
                    
            # PIE CHART
            elif viz_type == 'pie':
                if x and y:  # Pie chart needs categories and values
                    # Print detailed info for debugging pie charts
                    print(f"Creating pie chart with x={x}, y={y}")
                    print(f"X column '{x}' exists in df: {x in df.columns}")
                    print(f"Y column '{y}' exists in df: {y in df.columns}")
                    
                    # Check that y column is numeric
                    if not pd.api.types.is_numeric_dtype(plot_df[y]):
                        # Try to convert to numeric one more time
                        try:
                            plot_df[y] = pd.to_numeric(plot_df[y], errors='coerce')
                            if plot_df[y].isna().sum() > 0.5 * len(plot_df):
                                raise ValueError(f"Values column {y} could not be converted to numeric type")
                            print(f"Successfully converted {y} to numeric for pie chart")
                        except Exception as e:
                            raise ValueError(f"Values column {y} must be numeric for pie chart: {str(e)}")
                        
                    # Aggregate data to handle potential duplicates in categories
                    print(f"Aggregating data for pie chart...")
                    pie_data = plot_df.groupby(x)[y].sum().reset_index()
                    print(f"Aggregated data shape: {pie_data.shape}")
                    
                    # Remove any negative values (which cause errors in pie charts)
                    if (pie_data[y] < 0).any():
                        print("Warning: Removing negative values from pie chart data")
                        pie_data = pie_data[pie_data[y] >= 0]
                    
                    # Check if any data remains
                    if len(pie_data) == 0:
                        raise ValueError("No valid data points for pie chart after filtering")
                    
                    # Make sure values are not all zeros
                    if (pie_data[y] == 0).all():
                        raise ValueError("All values are zero, cannot create pie chart")
                    
                    # Print the data for debugging
                    print(f"Pie chart data (first 5 rows):\n{pie_data.head()}")
                    
                    # Create the pie chart
                    fig = px.pie(
                        pie_data, 
                        names=x, 
                        values=y, 
                        title=title,
                        labels={x: "Categories", y: "Values"}
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    
                    # Check if the figure has data
                    if 'data' not in fig.to_dict() or len(fig.to_dict()['data']) == 0:
                        raise ValueError("Plotly failed to generate a valid pie chart")
                    
                    return fig
                else:
                    raise ValueError("Pie chart requires both X (categories) and Y (numeric values) columns")
            
            else:
                # For any other type not specifically handled, try to use plotly express
                fig = px.scatter(plot_df, x=x, y=y, color=color, title=title)
                
            return fig
            
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
            return fig
    
    def create_advanced_viz(self, df, viz_type, params):
        """Create advanced visualizations"""
        x = params.get('x')
        y = params.get('y')
        color = params.get('color')
        title = params.get('title', f"{viz_type.replace('_', ' ').title()} Chart")
        
        # Check if dataframe is empty
        if df.empty:
            raise ValueError("Dataset is empty - cannot create visualization")
            
        # Create a copy of the DataFrame for visualization
        plot_df = df.copy()
        
        # Convert DataFrame data to appropriate types
        for col in plot_df.columns:
            if col in [x, y] and pd.api.types.is_object_dtype(plot_df[col]):
                try:
                    plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                    print(f"Converted column '{col}' to numeric type")
                except:
                    print(f"Could not convert column '{col}' to numeric type")
        
        # Validate column existence
        if x and x not in df.columns:
            raise ValueError(f"Column '{x}' not found in dataset")
        if y and y not in df.columns:
            raise ValueError(f"Column '{y}' not found in dataset")
        
        # Add validation for color column
        if color and color not in df.columns:
            print(f"Warning: Color column '{color}' not found in dataset. Ignoring color parameter.")
            color = None
        
        # Drop rows where the key columns have NaN values
        if x:
            plot_df = plot_df.dropna(subset=[x])
        if y:
            plot_df = plot_df.dropna(subset=[y])
        
        # Ensure there's still data after dropping NaN values
        if plot_df.empty:
            raise ValueError("No valid data points after removing NaN values from selected columns")
        
        try:
            # VIOLIN PLOT
            if viz_type == 'violin':
                if y:  # Violin plot requires y-axis values
                    if not pd.api.types.is_numeric_dtype(plot_df[y]):
                        raise ValueError(f"Column '{y}' must be numeric for violin plot")
                        
                    fig = go.Figure()
                    
                    if x:  # If x is provided, group by x
                        # Group data by x column
                        groups = plot_df.groupby(x)[y].apply(list).to_dict()
                        
                        for group_name, values in groups.items():
                            fig.add_trace(go.Violin(
                                y=values,
                                name=str(group_name),
                                box_visible=True,
                                meanline_visible=True,
                                points="all"
                            ))
                        
                        fig.update_layout(
                            title=title,
                            yaxis_title=y,
                            xaxis_title=x,
                            violinmode='group'
                        )
                    else:  # Simple violin plot for just the y column
                        fig = go.Figure()
                        fig.add_trace(go.Violin(
                            y=plot_df[y].dropna().tolist(),
                            name=y,
                            box_visible=True,
                            meanline_visible=True,
                            points="all"
                        ))
                        fig.update_layout(
                            title=title,
                            yaxis_title=y
                        )
                else:
                    raise ValueError("Violin plot requires a Y column (values to analyze)")
                    
            # HEATMAP
            elif viz_type == 'heatmap':
                if x and y:
                    # Get the value to use for the heatmap cells
                    z_col = params.get('z')  # Optional third variable for cell values
                    
                    print(f"Creating heatmap with x={x}, y={y}, z={z_col}")
                    print(f"Data types: x={plot_df[x].dtype}, y={plot_df[y].dtype}")
                    
                    # Ensure x and y are converted to string to avoid issues with numeric indices
                    plot_df[x] = plot_df[x].astype(str)
                    plot_df[y] = plot_df[y].astype(str)
                    
                    try:
                        if z_col and z_col in plot_df.columns:
                            print(f"Using z-values from column: {z_col}, dtype={plot_df[z_col].dtype}")
                            
                            # Ensure z-column is numeric
                            if not pd.api.types.is_numeric_dtype(plot_df[z_col]):
                                try:
                                    plot_df[z_col] = pd.to_numeric(plot_df[z_col], errors='coerce')
                                    print(f"Converted z column '{z_col}' to numeric")
                                except:
                                    raise ValueError(f"Z-value column '{z_col}' must be numeric for heatmap")
                            
                            # Pivot data to create heatmap with z as values
                            print(f"Creating pivot table with y={y} as index, x={x} as columns, z={z_col} as values")
                            pivot_df = plot_df.pivot_table(
                                values=z_col, 
                                index=y, 
                                columns=x, 
                                aggfunc='mean',
                                fill_value=0  # Fill NaN with zeros
                            )
                            
                            print(f"Pivot table shape: {pivot_df.shape}")
                            print(f"Pivot table index (first 5): {pivot_df.index[:5].tolist()}")
                            print(f"Pivot table columns (first 5): {pivot_df.columns[:5].tolist()}")
                            
                            # Check that pivot table is not empty
                            if pivot_df.empty:
                                raise ValueError("Pivot table is empty - cannot create heatmap")
                            
                        else:
                            print(f"Creating count-based heatmap for x={x} and y={y}")
                            
                            # Use the more robust crosstab method
                            print("Using pd.crosstab for count-based heatmap")
                            pivot_df = pd.crosstab(
                                plot_df[y], 
                                plot_df[x],
                                dropna=False
                            ).fillna(0)
                            
                            print(f"Crosstab shape: {pivot_df.shape}")
                            print(f"Crosstab index (first 5): {pivot_df.index[:5].tolist()}")
                            print(f"Crosstab columns (first 5): {pivot_df.columns[:5].tolist()}")
                            
                            if pivot_df.empty:
                                raise ValueError("Crosstab is empty - cannot create heatmap")
                        
                        # Convert to lists with explicit type conversion for z, x, and y values
                        z_values = pivot_df.values.tolist()
                        x_values = [str(val) for val in pivot_df.columns.tolist()]
                        y_values = [str(val) for val in pivot_df.index.tolist()]
                        
                        print(f"Heatmap data shapes: z={len(z_values)}x{len(z_values[0]) if z_values else 0}, x={len(x_values)}, y={len(y_values)}")
                        
                        # Create heatmap with more robust data
                        fig = go.Figure(data=go.Heatmap(
                            z=z_values,
                            x=x_values,
                            y=y_values,
                            colorscale=params.get('colorscale', 'Viridis'),
                            hoverongaps=False
                        ))
                        
                        # Enhance layout for better display
                        fig.update_layout(
                            title=title,
                            xaxis_title=x,
                            yaxis_title=y,
                            xaxis={'type': 'category'},  # Force categorical axes
                            yaxis={'type': 'category'},
                            margin={'l': 50, 'r': 50, 'b': 50, 't': 80}
                        )
                        
                        # Verify the structure before returning
                        fig_dict = fig.to_dict()
                        if not fig_dict.get('data') or not isinstance(fig_dict['data'], list) or len(fig_dict['data']) == 0:
                            raise ValueError("Generated heatmap has invalid structure")
                        
                        if not isinstance(fig_dict['data'][0].get('z', []), list):
                            raise ValueError("Generated heatmap doesn't have a valid 2D z array")
                        
                        return fig
                        
                    except Exception as e:
                        print(f"Error in heatmap creation: {str(e)}")
                        
                        # Create a fallback visualization with an error message
                        fig = go.Figure()
                        fig.add_annotation(
                            text=f"Error creating heatmap: {str(e)}",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=14, color="red")
                        )
                        
                        # Try to add a simple table as a fallback
                        try:
                            # Show raw data sample if we can't create heatmap
                            sample_data = plot_df[[x, y]].head(10).to_dict('records')
                            sample_text = "\n".join([f"{row[x]} - {row[y]}" for row in sample_data])
                            
                            fig.add_annotation(
                                text=f"Sample data (first 10 rows):<br>{sample_text}",
                                xref="paper", yref="paper",
                                x=0.5, y=0.3, showarrow=False,
                                font=dict(size=10)
                            )
                        except:
                            pass
                        
                        fig.update_layout(
                            title=f"Error in {viz_type} visualization",
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False)
                        )
                        
                        return fig
                else:
                    raise ValueError("Heatmap requires both X and Y columns")
            
            # CONTOUR PLOT
            elif viz_type == 'contour':
                if x and y:
                    # Get the value to use for the contour z values
                    z_col = params.get('z')
                    
                    print(f"Creating contour plot with x={x}, y={y}, z={z_col}")
                    
                    # For contour plots, x and y must be numeric
                    for col, label in [(x, 'X'), (y, 'Y')]:
                        if not pd.api.types.is_numeric_dtype(plot_df[col]):
                            try:
                                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                                print(f"Converted {label} column '{col}' to numeric for contour plot")
                            except:
                                raise ValueError(f"{label} column '{col}' must be numeric for contour plot")
                    
                    # Filter out NaN values
                    plot_df = plot_df.dropna(subset=[x, y])
                    
                    if len(plot_df) < 10:
                        raise ValueError("Not enough valid data points for contour plot after filtering")
                    
                    # Use direct Graph Objects approach for more control
                    fig = go.Figure()
                    
                    # Different types of contour plots based on whether z is provided
                    if z_col and z_col in plot_df.columns:
                        # Make sure z column is numeric
                        if not pd.api.types.is_numeric_dtype(plot_df[z_col]):
                            try:
                                plot_df[z_col] = pd.to_numeric(plot_df[z_col], errors='coerce')
                                print(f"Converted Z column '{z_col}' to numeric for contour plot")
                            except:
                                raise ValueError(f"Z column '{z_col}' must be numeric for contour plot")
                        
                        # Remove NaN in z column
                        plot_df = plot_df.dropna(subset=[z_col])
                        
                        if len(plot_df) < 10:
                            raise ValueError("Not enough valid data points with Z values for contour plot")
                        
                        # Use griddata to interpolate scattered points to a regular grid
                        try:
                            from scipy.interpolate import griddata
                            
                            # Create a regular grid
                            x_arr = plot_df[x].values
                            y_arr = plot_df[y].values
                            z_arr = plot_df[z_col].values
                            
                            # Create grid for interpolation
                            x_min, x_max = x_arr.min(), x_arr.max()
                            y_min, y_max = y_arr.min(), y_arr.max()
                            x_grid = np.linspace(x_min, x_max, 100)
                            y_grid = np.linspace(y_min, y_max, 100)
                            X, Y = np.meshgrid(x_grid, y_grid)
                            
                            # Interpolate Z values on the grid
                            Z = griddata((x_arr, y_arr), z_arr, (X, Y), method='cubic', fill_value=np.nan)
                            
                            # Add contour trace with gridded data
                            fig.add_trace(go.Contour(
                                z=Z,
                                x=x_grid,
                                y=y_grid,
                                colorscale=params.get('colorscale', 'Viridis'),
                                contours=dict(
                                    showlabels=True,
                                    labelfont=dict(
                                        family='Raleway',
                                        size=12,
                                        color='white'
                                    )
                                ),
                                colorbar=dict(
                                    title=z_col,
                                    titleside='right'
                                )
                            ))
                        except Exception as interp_error:
                            print(f"Interpolation error: {str(interp_error)}, falling back to Plotly Express")
                            # Fallback to Plotly Express method - FIX PARAMETER NAME HERE
                            fig = px.density_contour(
                                plot_df,
                                x=x,
                                y=y,
                                z=z_col,
                                color_continuous_scale=params.get('colorscale', 'Viridis')
                            )
                    else:
                        # Create a density contour plot (no z required)
                        print("Creating density contour plot based on point density")
                        
                        # Add scatter points to show data
                        fig.add_trace(go.Scatter(
                            x=plot_df[x],
                            y=plot_df[y],
                            mode='markers',
                            marker=dict(
                                size=5,
                                opacity=0.5,
                                color='rgba(100,100,100,0.5)'
                            ),
                            name='Data Points'
                        ))
                        
                        # Add the contour trace
                        fig.add_trace(go.Histogram2dContour(
                            x=plot_df[x],
                            y=plot_df[y],
                            colorscale=params.get('colorscale', 'Viridis'),
                            contours=dict(
                                showlabels=True,
                                labelfont=dict(
                                    family='Raleway',
                                    size=12,
                                    color='white'
                                )
                            ),
                            histfunc='count',
                            name='Density'
                        ))
                    
                    # Enhance layout
                    fig.update_layout(
                        title=title,
                        xaxis_title=x,
                        yaxis_title=y
                    )
                    
                    # Validate the figure before returning
                    fig_dict = fig.to_dict()
                    
                    # Ensure traces are actually included
                    if not fig_dict.get('data') or len(fig_dict['data']) == 0:
                        print("Warning: Generated contour plot has no data traces - creating fallback")
                        # Create an explicit fallback contour if needed
                        fig = go.Figure(go.Contour(
                            z=[[1, 2, 3, 4, 5], 
                               [2, 3, 4, 5, 6], 
                               [3, 4, 5, 6, 7], 
                               [4, 5, 6, 7, 8], 
                               [5, 6, 7, 8, 9]],
                            colorscale='Viridis',
                            contours=dict(showlabels=True),
                            name='Fallback Contour'
                        ))
                        fig.update_layout(
                            title=f"Error: Could not create {title} with your data",
                            annotations=[
                                dict(
                                    text="No valid contour could be created with the selected data.<br>Try different columns with more numeric values.",
                                    xref="paper", yref="paper",
                                    x=0.5, y=0.5, showarrow=False,
                                    font=dict(color="red", size=14)
                                )
                            ]
                        )
                    
                    return fig
                else:
                    raise ValueError("Contour plot requires both X and Y columns")
                    
            # DENSITY CONTOUR PLOT
            elif viz_type == 'density_contour':
                if x and y:
                    # Need numeric columns
                    for col, label in [(x, 'X'), (y, 'Y')]:
                        if not pd.api.types.is_numeric_dtype(plot_df[col]):
                            try:
                                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                                print(f"Converted {label} column '{col}' to numeric for density contour plot")
                            except:
                                raise ValueError(f"{label} column '{col}' must be numeric for density contour plot")
                    
                    # Filter out NaN values
                    plot_df = plot_df.dropna(subset=[x, y])
                    
                    if len(plot_df) < 10:
                        raise ValueError("Not enough valid data points for density contour plot after filtering")
                    
                    # Create a density contour plot
                    fig = px.density_contour(
                        plot_df,
                        x=x,
                        y=y,
                        color=color if color in plot_df.columns else None,
                        color_continuous_scale=params.get('colorscale', 'Viridis'),  # FIXED PARAMETER NAME
                        marginal_x="histogram",
                        marginal_y="histogram"
                    )
                    
                    # Improve the contour appearance
                    fig.update_traces(
                        contours_coloring="fill", 
                        colorscale=params.get('colorscale', 'Viridis'),  # This is correct as-is for update_traces
                        selector=dict(type='contour')
                    )
                    
                    fig.update_layout(
                        title=title,
                        xaxis_title=x,
                        yaxis_title=y
                    )
                    
                    return fig
                else:
                    raise ValueError("Density contour plot requires both X and Y columns")
            
            # 3D SCATTER PLOT
            elif viz_type == 'scatter_3d':
                if x and y:
                    # Need a z column for 3D plot
                    z = params.get('z')
                    if not z or z not in plot_df.columns:
                        raise ValueError("3D scatter plot requires X, Y, and Z columns")
                    
                    # Make sure all columns are numeric
                    for col, col_name in [(x, 'X'), (y, 'Y'), (z, 'Z')]:
                        if not pd.api.types.is_numeric_dtype(plot_df[col]):
                            try:
                                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                                print(f"Converted {col_name} column '{col}' to numeric for 3D scatter plot")
                            except:
                                raise ValueError(f"{col_name} column '{col}' must be numeric for 3D scatter plot")
                    
                    # Drop NaNs in z column as well
                    plot_df = plot_df.dropna(subset=[z])
                    
                    if len(plot_df) < 3:
                        raise ValueError("Not enough valid data points for 3D scatter plot after filtering")
                    
                    # Create 3D scatter plot
                    fig = px.scatter_3d(
                        plot_df,
                        x=x,
                        y=y,
                        z=z,
                        color=color if color in plot_df.columns else None,
                        title=title,
                        opacity=0.7
                    )
                    
                    # Improve the appearance
                    fig.update_layout(
                        scene = dict(
                            xaxis_title=x,
                            yaxis_title=y,
                            zaxis_title=z
                        ),
                        margin=dict(l=0, r=0, b=0, t=30)
                    )
                    
                    return fig
                else:
                    raise ValueError("3D scatter plot requires X, Y, and Z columns")
                
            # 3D SURFACE PLOT
            elif viz_type == 'surface_3d':
                if x and y:
                    # Need a z column for surface plot
                    z = params.get('z')
                    if not z or z not in plot_df.columns:
                        raise ValueError("3D surface plot requires X, Y, and Z columns")
                    
                    # Make sure all columns are numeric
                    for col, col_name in [(x, 'X'), (y, 'Y'), (z, 'Z')]:
                        if not pd.api.types.is_numeric_dtype(plot_df[col]):
                            try:
                                plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                                print(f"Converted {col_name} column '{col}' to numeric for 3D surface plot")
                            except:
                                raise ValueError(f"{col_name} column '{col}' must be numeric for 3D surface plot")
                    
                    # Drop NaNs in z column as well
                    plot_df = plot_df.dropna(subset=[z])
                    
                    if len(plot_df) < 10:
                        raise ValueError("Not enough valid data points for 3D surface plot after filtering")
                    
                    try:
                        # For surface plots, data must be on a grid
                        from scipy.interpolate import griddata
                        
                        # Extract arrays
                        x_arr = plot_df[x].values
                        y_arr = plot_df[y].values
                        z_arr = plot_df[z].values
                        
                        # Create a regular grid
                        x_grid = np.linspace(x_arr.min(), x_arr.max(), 50)
                        y_grid = np.linspace(y_arr.min(), y_arr.max(), 50)
                        X, Y = np.meshgrid(x_grid, y_grid)
                        
                        # Interpolate Z values
                        Z = griddata((x_arr, y_arr), z_arr, (X, Y), method='cubic', fill_value=np.nan)
                        
                        # Create surface plot with interpolated grid data
                        fig = go.Figure(data=[go.Surface(z=Z, x=x_grid, y=y_grid, colorscale='Viridis')])
                        
                        fig.update_layout(
                            title=title,
                            scene = dict(
                                xaxis_title=x,
                                yaxis_title=y,
                                zaxis_title=z
                            ),
                            margin=dict(l=0, r=0, b=0, t=30)
                        )
                        
                        return fig
                    except Exception as e:
                        print(f"Error creating surface plot: {str(e)}")
                        raise ValueError(f"Could not create surface plot: {str(e)}")
                else:
                    raise ValueError("3D surface plot requires X, Y, and Z columns")
                
            # PARALLEL COORDINATES
            elif viz_type == 'parallel_coordinates':
                # Get columns to include
                columns = params.get('columns', [])
                if not columns:
                    # If no columns specified, use all numeric columns
                    columns = plot_df.select_dtypes(include=['number']).columns.tolist()
                    
                if len(columns) < 2:
                    raise ValueError("Parallel coordinates plot requires at least 2 numeric columns")
                
                # Ensure columns exist in DataFrame
                columns = [col for col in columns if col in plot_df.columns]
                
                # Check if we have enough columns after filtering
                if len(columns) < 2:
                    raise ValueError("Not enough valid columns for parallel coordinates plot")
                
                # Create a color column if specified
                color_col = color if color in plot_df.columns else None
                
                try:
                    fig = px.parallel_coordinates(
                        plot_df, 
                        dimensions=columns,
                        color=color_col,
                        title=title
                    )
                    
                    fig.update_layout(
                        font=dict(size=10),
                        margin=dict(l=80, r=80, t=60, b=30)
                    )
                    
                    return fig
                except Exception as e:
                    print(f"Error creating parallel coordinates plot: {str(e)}")
                    raise ValueError(f"Failed to create parallel coordinates plot: {str(e)}")
                
            # TREEMAP
            elif viz_type == 'treemap':
                # Need path columns and value column
                path_cols = params.get('path', [])
                value_col = params.get('values')
                
                if not path_cols or not path_cols[0] in plot_df.columns:
                    raise ValueError("Treemap requires at least one path column")
                    
                if not value_col or value_col not in plot_df.columns:
                    raise ValueError("Treemap requires a values column for sizes")
                
                # Ensure value column is numeric
                if not pd.api.types.is_numeric_dtype(plot_df[value_col]):
                    try:
                        plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors='coerce')
                        print(f"Converted values column '{value_col}' to numeric for treemap")
                    except:
                        raise ValueError(f"Values column '{value_col}' must be numeric for treemap")
                
                # Filter to only existing path columns
                path_cols = [col for col in path_cols if col in plot_df.columns]
                
                if not path_cols:
                    raise ValueError("No valid path columns found for treemap")
                
                # Remove NaN values from value column and path columns
                plot_df = plot_df.dropna(subset=[value_col] + path_cols)
                
                if plot_df.empty:
                    raise ValueError("No valid data points after removing NaN values")
                
                try:
                    fig = px.treemap(
                        plot_df,
                        path=path_cols,
                        values=value_col,
                        color=color if color in plot_df.columns else None,
                        title=title
                    )
                    
                    fig.update_layout(
                        margin=dict(l=0, r=0, b=0, t=30)
                    )
                    
                    return fig
                except Exception as e:
                    print(f"Error creating treemap: {str(e)}")
                    raise ValueError(f"Failed to create treemap: {str(e)}")
                
            # SUNBURST CHART
            elif viz_type == 'sunburst':
                # Need path columns and value column
                path_cols = params.get('path', [])
                value_col = params.get('values')
                
                if not path_cols or not path_cols[0] in plot_df.columns:
                    raise ValueError("Sunburst chart requires at least one path column")
                    
                if not value_col or value_col not in plot_df.columns:
                    raise ValueError("Sunburst chart requires a values column for sizes")
                
                # Ensure value column is numeric
                if not pd.api.types.is_numeric_dtype(plot_df[value_col]):
                    try:
                        plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors='coerce')
                        print(f"Converted values column '{value_col}' to numeric for sunburst chart")
                    except:
                        raise ValueError(f"Values column '{value_col}' must be numeric for sunburst chart")
                
                # Filter to only existing path columns
                path_cols = [col for col in path_cols if col in plot_df.columns]
                
                if not path_cols:
                    raise ValueError("No valid path columns found for sunburst chart")
                
                # Remove NaN values
                plot_df = plot_df.dropna(subset=[value_col] + path_cols)
                
                if plot_df.empty:
                    raise ValueError("No valid data points after removing NaN values")
                
                try:
                    fig = px.sunburst(
                        plot_df,
                        path=path_cols,
                        values=value_col,
                        color=color if color in plot_df.columns else None,
                        title=title
                    )
                    
                    fig.update_layout(
                        margin=dict(l=0, r=0, b=0, t=30)
                    )
                    
                    return fig
                except Exception as e:
                    print(f"Error creating sunburst chart: {str(e)}")
                    raise ValueError(f"Failed to create sunburst chart: {str(e)}")

            # RADAR/POLAR CHART
            elif viz_type == 'radar':
                # Need categories (theta) and values (r)
                theta_col = params.get('theta')
                r_col = params.get('r')
                
                if not theta_col or theta_col not in plot_df.columns:
                    raise ValueError("Radar chart requires a theta (categories) column")
                    
                if not r_col or r_col not in plot_df.columns:
                    raise ValueError("Radar chart requires an r (values) column")
                
                # Ensure r column is numeric
                if not pd.api.types.is_numeric_dtype(plot_df[r_col]):
                    try:
                        plot_df[r_col] = pd.to_numeric(plot_df[r_col], errors='coerce')
                        print(f"Converted r column '{r_col}' to numeric for radar chart")
                    except:
                        raise ValueError(f"Values column '{r_col}' must be numeric for radar chart")
                
                # Remove NaN values
                plot_df = plot_df.dropna(subset=[theta_col, r_col])
                
                if plot_df.empty:
                    raise ValueError("No valid data points after removing NaN values")
                
                try:
                    # Create the radar chart with plotly.graph_objects for more control
                    fig = go.Figure()
                    
                    if color and color in plot_df.columns:
                        # Group by color column
                        for group_name, group_df in plot_df.groupby(color):
                            fig.add_trace(go.Scatterpolar(
                                r=group_df[r_col],
                                theta=group_df[theta_col],
                                fill='toself',
                                name=str(group_name)
                            ))
                    else:
                        # Single trace radar chart
                        fig.add_trace(go.Scatterpolar(
                            r=plot_df[r_col],
                            theta=plot_df[theta_col],
                            fill='toself'
                        ))
                    
                    fig.update_layout(
                        title=title,
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, plot_df[r_col].max() * 1.1]
                            )
                        ),
                        showlegend=True if color else False
                    )
                    
                    return fig
                except Exception as e:
                    print(f"Error creating radar chart: {str(e)}")
                    raise ValueError(f"Failed to create radar chart: {str(e)}")
            
            else:
                # Default to scatter plot for unimplemented types
                print(f"Warning: Advanced plot type '{viz_type}' not specifically implemented. Using scatter plot as fallback.")
                fig = px.scatter(plot_df, x=x, y=y, color=color, title=f"{title} (Fallback)")
            
            return fig
            
        except Exception as e:
            print(f"Error creating {viz_type} plot: {str(e)}")
            # Create a fallback error figure
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
            
            return fig
        elif viz_type == 'distribution':
            # Get the column to plot
            column = params.get('column') or params.get('x') or params.get('y')
            
            if not column or column not in df.columns:
                raise ValueError("Distribution plot requires a valid column")
                
            # Ensure the column is numeric
            if not pd.api.types.is_numeric_dtype(df[column]):
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                except:
                    raise ValueError(f"Column {column} must be numeric for distribution plot")
            
            # Create figure with histogram and KDE
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=df[column].dropna(),
                name='Histogram',
                opacity=0.75,
                nbinsx=30
            ))
            
            # Add KDE (use a smoothed version of the histogram)
            try:
                from scipy import stats
                
                kde_x = np.linspace(df[column].min(), df[column].max(), 1000)
                kde_y = stats.gaussian_kde(df[column].dropna())(kde_x)
                
                # Scale KDE to match histogram height
                scale_factor = (df[column].count() / 30) / kde_y.max() * 0.8
                
                fig.add_trace(go.Scatter(
                    x=kde_x,
                    y=kde_y * scale_factor,
                    mode='lines',
                    name='KDE',
                    line=dict(color='red', width=2)
                ))
            except Exception as e:
                print(f"Warning: Couldn't generate KDE for distribution plot: {e}")
            
            fig.update_layout(
                title=title,
                xaxis_title=column,
                yaxis_title='Frequency',
                bargap=0.05
            )
            
            return fig
        
        elif viz_type == 'qq_plot':
            # Get the column to plot
            column = params.get('column')
            
            if not column or column not in df.columns:
                raise ValueError("Q-Q plot requires a valid column")
                
            # Ensure the column is numeric
            if not pd.api.types.is_numeric_dtype(df[column]):
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                except:
                    raise ValueError(f"Column {column} must be numeric for Q-Q plot")
            
            # Drop NaNs
            data = df[column].dropna()
            
            if len(data) < 3:
                raise ValueError("Not enough valid data points for Q-Q plot")
            
            try:
                # Create QQ plot using scipy and plotly
                from scipy import stats
                
                # Calculate quantiles
                quantiles = np.linspace(0.01, 0.99, min(100, len(data)))
                sample_quantiles = np.quantile(data, quantiles)
                theoretical_quantiles = stats.norm.ppf(quantiles)
                
                # Create scatter plot of quantiles
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    marker=dict(color='blue', size=8),
                    name=column
                ))
                
                # Add reference line
                min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Reference Line'
                ))
                
                fig.update_layout(
                    title=title,
                    xaxis_title='Theoretical Quantiles',
                    yaxis_title='Sample Quantiles',
                    showlegend=True
                )
                
                return fig
            except Exception as e:
                print(f"Error creating Q-Q plot: {str(e)}")
                raise ValueError(f"Failed to create Q-Q plot: {str(e)}")
        
        elif viz_type == 'residual_plot':
            # Need x (predictor) and y (response) columns
            x_col = params.get('x')
            y_col = params.get('y')
            
            if not x_col or x_col not in df.columns:
                raise ValueError("Residual plot requires a predictor (x) column")
                
            if not y_col or y_col not in df.columns:
                raise ValueError("Residual plot requires a response (y) column")
            
            # Ensure columns are numeric
            for col, label in [(x_col, 'Predictor'), (y_col, 'Response')]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        raise ValueError(f"{label} column '{col}' must be numeric for residual plot")
            
            # Drop NaN values
            plot_df = df.dropna(subset=[x_col, y_col])
            
            if len(plot_df) < 3:
                raise ValueError("Not enough valid data points for residual plot")
            
            try:
                # Fit a linear regression model
                from scipy import stats
                
                x_values = plot_df[x_col].values
                y_values = plot_df[y_col].values
                
                # Add a constant to x for the intercept
                X = np.column_stack((np.ones(len(x_values)), x_values))
                
                # Fit the model
                beta, residuals, rank, s = np.linalg.lstsq(X, y_values, rcond=None)
                
                # Calculate predicted values
                y_pred = beta[0] + beta[1] * x_values
                
                # Calculate residuals
                residuals = y_values - y_pred
                
                # Create residual plot
                fig = go.Figure()
                
                # Add residuals scatter plot
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=residuals,
                    mode='markers',
                    marker=dict(color='blue', size=8),
                    name='Residuals'
                ))
                
                # Add zero line
                fig.add_trace(go.Scatter(
                    x=[min(x_values), max(x_values)],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Zero Line'
                ))
                
                fig.update_layout(
                    title=title,
                    xaxis_title=x_col,
                    yaxis_title=f'Residuals ({y_col} - predicted {y_col})',
                    showlegend=True
                )
                
                return fig
            except Exception as e:
                print(f"Error creating residual plot: {str(e)}")
                raise ValueError(f"Failed to create residual plot: {str(e)}")
        
        elif viz_type == 'pair_plot':
            # Get columns to include in pair plot
            columns = params.get('columns', [])
            
            if not columns:
                # If no columns specified, use all numeric columns (up to a limit)
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                columns = numeric_cols[:min(5, len(numeric_cols))]  # Limit to 5 columns
            
            # Validate columns exist in dataframe
            columns = [col for col in columns if col in df.columns]
            
            if len(columns) < 2:
                raise ValueError("Pair plot requires at least 2 valid numeric columns")
            
            # Get optional color column
            color_col = params.get('color')
            if color_col and color_col not in df.columns:
                print(f"Warning: Color column '{color_col}' not found in dataset")
                color_col = None
            
            try:
                # Create a smaller dataframe with just the needed columns
                plot_df = df[columns + ([color_col] if color_col else [])]
                plot_df = plot_df.dropna()
                
                if plot_df.empty:
                    raise ValueError("No valid data points after removing NaN values")
                
                # Create pair plot using plotly
                fig = px.scatter_matrix(
                    plot_df,
                    dimensions=columns,
                    color=color_col,
                    title=title
                )
                
                # Improve the appearance
                fig.update_traces(
                    diagonal_visible=False,
                    showupperhalf=False,
                    marker=dict(size=4)
                )
                
                fig.update_layout(
                    height=600,
                    width=700,
                    title=dict(
                        text=title,
                        x=0.5
                    )
                )
                
                return fig
            except Exception as e:
                print(f"Error creating pair plot: {str(e)}")
                raise ValueError(f"Failed to create pair plot: {str(e)}")
        
        elif viz_type == 'cluster_map':
            # Get columns to include
            columns = params.get('columns', [])
            
            if not columns:
                # If no columns specified, use all numeric columns
                columns = df.select_dtypes(include=['number']).columns.tolist()
            
            # Validate columns exist in dataframe
            columns = [col for col in columns if col in df.columns]
            
            if len(columns) < 2:
                raise ValueError("Cluster map requires at least 2 valid numeric columns")
            
            try:
                # Create a dataframe with just the needed columns
                plot_df = df[columns].dropna()
                
                if plot_df.empty:
                    raise ValueError("No valid data points after removing NaN values")
                
                # Compute the correlation matrix
                corr_matrix = plot_df.corr()
                
                # Compute linkage for hierarchical clustering
                from scipy.cluster.hierarchy import linkage
                
                # Convert correlation to distance (1 - corr)
                distance_matrix = 1 - np.abs(corr_matrix)
                
                # Perform clustering (using condensed distance matrix)
                z = linkage(distance_matrix.values.flatten(), method='ward')
                
                # Get order of indices for dendrogram
                from scipy.cluster.hierarchy import dendrogram
                dendro = dendrogram(z, no_plot=True)
                
                # Get ordered indices
                ordered_indices = dendro['leaves']
                
                # Reorder correlation matrix
                ordered_corr = corr_matrix.iloc[ordered_indices, ordered_indices]
                
                # Create heatmap
                fig = px.imshow(
                    ordered_corr,
                    text_auto=params.get('show_values', True),
                    color_continuous_scale=params.get('colorscale', 'RdBu_r'),
                    title=title
                )
                
                fig.update_layout(
                    height=600,
                    width=700
                )
                
                return fig
            except Exception as e:
                print(f"Error creating cluster map: {str(e)}")
                raise ValueError(f"Failed to create cluster map: {str(e)}")
        
        else:
            raise ValueError(f"Statistical visualization type {viz_type} is not implemented")
    
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
                'x': {'type': 'column', 'label': 'X-Axis', 'required': True},
                'y': {'type': 'column', 'label': 'Y-Axis', 'required': True},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True},
                'use_index': {'type': 'boolean', 'label': 'Use Row Index for Single Column Plot', 'default': False}
            }
        elif viz_type == 'line':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'X-Axis', 'required': True},
                'y': {'type': 'column', 'label': 'Y-Axis', 'required': True},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True},
                'use_index': {'type': 'boolean', 'label': 'Use Row Index for Single Column Plot', 'default': False}
            }
        elif viz_type == 'bar':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'Categories (X-Axis)', 'required': True},
                'y': {'type': 'column', 'label': 'Values (Y-Axis)', 'required': True},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True}
            }
        elif viz_type == 'histogram':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'Values', 'required': True},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True},
                'bin_size': {'type': 'number', 'label': 'Bin Size', 'optional': True}
            }
        elif viz_type == 'box':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'Group By (Optional)', 'optional': True},
                'y': {'type': 'column', 'label': 'Values', 'required': True},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True}
            }
        elif viz_type == 'pie':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'Categories', 'required': True},
                'y': {'type': 'column', 'label': 'Values (must be numeric)', 'required': True}
            }
        
        # Advanced viz options
        elif viz_type == 'scatter_3d':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'X-Axis', 'required': True},
                'y': {'type': 'column', 'label': 'Y-Axis', 'required': True},
                'z': {'type': 'column', 'label': 'Z-Axis', 'required': True},
                'color': {'type': 'column', 'label': 'Color By', 'optional': True}
            }
        elif viz_type == 'contour':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'X-Axis (Numeric)', 'required': True},
                'y': {'type': 'column', 'label': 'Y-Axis (Numeric)', 'required': True},
                'z': {'type': 'column', 'label': 'Z Values (Optional)', 'optional': True},
                'colorscale': {'type': 'select', 'label': 'Color Scale', 'options': ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Jet']}
            }
        elif viz_type == 'density_contour':
            return {
                **common_options,
                'x': {'type': 'column', 'label': 'X-Axis (Numeric)', 'required': True},
                'y': {'type': 'column', 'label': 'Y-Axis (Numeric)', 'required': True},
                'color': {'type': 'column', 'label': 'Color By (Optional)', 'optional': True},
                'colorscale': {'type': 'select', 'label': 'Color Scale', 'options': ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Jet']}
            }
        
        # Statistical viz options
        elif viz_type == 'correlation':
            return {
                **common_options,
                'columns': {'type': 'multicolumn', 'label': 'Columns to Correlate', 'optional': True},
                'show_values': {'type': 'boolean', 'label': 'Show Values', 'default': True},
                'colorscale': {'type': 'select', 'label': 'Color Scale', 'options': ['RdBu_r', 'Viridis', 'Plasma', 'Cividis']}
            }
        elif viz_type == 'distribution':
            return {
                **common_options,
                'column': {'type': 'column', 'label': 'Values', 'required': True},
                'bin_size': {'type': 'number', 'label': 'Bin Size', 'optional': True}
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
