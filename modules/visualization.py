import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from modules.data_loader import DataLoader

class Visualizer:
    """
    Creates various types of visualizations for data analysis.
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.supported_visualizations = [
            'scatter', 'line', 'bar', 'histogram', 'box', 'violin', 
            'heatmap', 'correlation', 'pie', 'sunburst', 'treemap',
            'parallel_coordinates', 'parallel_categories', 'scatter_3d',
            'distribution', 'time_series', 'pca', 'tsne', 'umap'
        ]
    
    def create_visualization(self, file_path, viz_type, params=None):
        """
        Create a visualization based on the given parameters
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        viz_type : str
            Type of visualization to create
        params : dict, optional
            Parameters for the visualization
            
        Returns:
        --------
        dict
            Visualization data in Plotly format
        """
        if viz_type not in self.supported_visualizations:
            raise ValueError(f"Unsupported visualization type: {viz_type}. Supported types: {self.supported_visualizations}")
        
        # Load data
        df = self.data_loader.load_data(file_path)
        
        # Use provided parameters or empty dict
        params = params or {}
        
        # Create visualization based on type
        if viz_type == 'scatter':
            return self._create_scatter(df, params)
        elif viz_type == 'line':
            return self._create_line(df, params)
        elif viz_type == 'bar':
            return self._create_bar(df, params)
        elif viz_type == 'histogram':
            return self._create_histogram(df, params)
        elif viz_type == 'box':
            return self._create_box(df, params)
        elif viz_type == 'violin':
            return self._create_violin(df, params)
        elif viz_type == 'heatmap':
            return self._create_heatmap(df, params)
        elif viz_type == 'correlation':
            return self._create_correlation(df, params)
        elif viz_type == 'pie':
            return self._create_pie(df, params)
        elif viz_type == 'distribution':
            return self._create_distribution(df, params)
        elif viz_type == 'pca':
            return self._create_pca(df, params)
        elif viz_type == 'scatter_3d':
            return self._create_scatter_3d(df, params)
        # Add other visualization types as needed
    
    def _create_scatter(self, df, params):
        x = params.get('x')
        y = params.get('y')
        color = params.get('color')
        size = params.get('size')
        title = params.get('title', 'Scatter Plot')
        
        if not x or not y:
            raise ValueError("Both 'x' and 'y' must be specified for scatter plot")
        
        fig = px.scatter(
            df, x=x, y=y, 
            color=color, size=size,
            title=title,
            labels={
                x: params.get('x_label', x),
                y: params.get('y_label', y)
            },
            template='plotly_white'
        )
        
        return {'figure': fig.to_dict(), 'type': 'scatter'}
    
    def _create_line(self, df, params):
        x = params.get('x')
        y = params.get('y')
        color = params.get('color')
        title = params.get('title', 'Line Plot')
        
        if not x or not y:
            raise ValueError("Both 'x' and 'y' must be specified for line plot")
        
        fig = px.line(
            df, x=x, y=y, 
            color=color,
            title=title,
            labels={
                x: params.get('x_label', x),
                y: params.get('y_label', y)
            },
            template='plotly_white'
        )
        
        return {'figure': fig.to_dict(), 'type': 'line'}
    
    def _create_bar(self, df, params):
        x = params.get('x')
        y = params.get('y')
        color = params.get('color')
        title = params.get('title', 'Bar Chart')
        
        if not x or not y:
            raise ValueError("Both 'x' and 'y' must be specified for bar chart")
        
        fig = px.bar(
            df, x=x, y=y, 
            color=color,
            title=title,
            labels={
                x: params.get('x_label', x),
                y: params.get('y_label', y)
            },
            template='plotly_white'
        )
        
        return {'figure': fig.to_dict(), 'type': 'bar'}
    
    def _create_histogram(self, df, params):
        x = params.get('x')
        color = params.get('color')
        nbins = params.get('nbins', 30)
        title = params.get('title', 'Histogram')
        
        if not x:
            raise ValueError("'x' must be specified for histogram")
        
        fig = px.histogram(
            df, x=x, 
            color=color,
            nbins=nbins,
            title=title,
            labels={x: params.get('x_label', x)},
            template='plotly_white'
        )
        
        return {'figure': fig.to_dict(), 'type': 'histogram'}
    
    def _create_box(self, df, params):
        x = params.get('x')
        y = params.get('y')
        color = params.get('color')
        title = params.get('title', 'Box Plot')
        
        if not y:
            raise ValueError("'y' must be specified for box plot")
        
        fig = px.box(
            df, x=x, y=y,
            color=color,
            title=title,
            labels={
                x: params.get('x_label', x) if x else "",
                y: params.get('y_label', y)
            },
            template='plotly_white'
        )
        
        return {'figure': fig.to_dict(), 'type': 'box'}
    
    def _create_violin(self, df, params):
        x = params.get('x')
        y = params.get('y')
        color = params.get('color')
        title = params.get('title', 'Violin Plot')
        
        if not y:
            raise ValueError("'y' must be specified for violin plot")
        
        fig = px.violin(
            df, x=x, y=y,
            color=color,
            title=title,
            labels={
                x: params.get('x_label', x) if x else "",
                y: params.get('y_label', y)
            },
            template='plotly_white'
        )
        
        return {'figure': fig.to_dict(), 'type': 'violin'}
    
    def _create_heatmap(self, df, params):
        title = params.get('title', 'Heatmap')
        
        # Filter only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for heatmap")
        
        fig = px.imshow(
            numeric_df,
            title=title,
            color_continuous_scale=params.get('color_scale', 'Viridis'),
            template='plotly_white'
        )
        
        return {'figure': fig.to_dict(), 'type': 'heatmap'}
    
    def _create_correlation(self, df, params):
        title = params.get('title', 'Correlation Matrix')
        
        # Filter only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation matrix")
        
        # Compute correlation matrix
        corr = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr,
            title=title,
            color_continuous_scale=params.get('color_scale', 'RdBu_r'),
            template='plotly_white',
            text_auto=params.get('show_values', True)
        )
        
        return {'figure': fig.to_dict(), 'type': 'correlation'}
    
    def _create_pie(self, df, params):
        values = params.get('values')
        names = params.get('names')
        title = params.get('title', 'Pie Chart')
        
        if not values or not names:
            raise ValueError("Both 'values' and 'names' must be specified for pie chart")
        
        fig = px.pie(
            df, values=values, names=names,
            title=title,
            template='plotly_white'
        )
        
        return {'figure': fig.to_dict(), 'type': 'pie'}
    
    def _create_distribution(self, df, params):
        columns = params.get('columns', df.select_dtypes(include=['number']).columns.tolist())
        title = params.get('title', 'Distribution Plot')
        
        if not columns:
            raise ValueError("No numeric columns found for distribution plot")
        
        rows = (len(columns) + 1) // 2
        fig = make_subplots(rows=rows, cols=2, subplot_titles=columns)
        
        for i, col in enumerate(columns):
            row = i // 2 + 1
            col_pos = i % 2 + 1
            
            # Add histogram
            fig.add_trace(
                go.Histogram(x=df[col], name=col),
                row=row, col=col_pos
            )
            
        fig.update_layout(
            title=title,
            showlegend=False,
            template='plotly_white',
            height=300*rows
        )
        
        return {'figure': fig.to_dict(), 'type': 'distribution'}
    
    def _create_pca(self, df, params):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        features = params.get('features')
        n_components = params.get('n_components', 2)
        color = params.get('color')
        title = params.get('title', 'PCA Plot')
        
        if not features:
            # Use all numeric columns if not specified
            features = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(features) < 2:
            raise ValueError("At least 2 features are required for PCA")
        
        # Prepare data
        X = df[features].copy()
        X = X.dropna()  # Remove rows with NaN values
        
        # Standardize features
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_scaled)
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame(
            data=components,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Add color column if specified
        if color and color in df.columns:
            pca_df[color] = df.loc[X.index, color].values
        
        # Create plot
        if n_components == 2:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                color=color if color and color in df.columns else None,
                title=f"{title} - Explained Variance: {pca.explained_variance_ratio_.sum():.2f}",
                labels={
                    'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2f})',
                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2f})'
                },
                template='plotly_white'
            )
        elif n_components == 3:
            fig = px.scatter_3d(
                pca_df, x='PC1', y='PC2', z='PC3',
                color=color if color and color in df.columns else None,
                title=f"{title} - Explained Variance: {pca.explained_variance_ratio_.sum():.2f}",
                labels={
                    'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2f})',
                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2f})',
                    'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2f})'
                },
                template='plotly_white'
            )
        
        # Add loading vectors if 2D
        if n_components == 2 and params.get('show_loadings', True):
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            # Create loading annotations
            for i, feature in enumerate(features):
                fig.add_shape(
                    type='line',
                    x0=0, y0=0,
                    x1=loadings[i, 0],
                    y1=loadings[i, 1],
                    line=dict(color='red', width=1, dash='dot')
                )
                
                fig.add_annotation(
                    x=loadings[i, 0],
                    y=loadings[i, 1],
                    ax=0, ay=0,
                    xanchor="center",
                    yanchor="bottom",
                    text=feature,
                    arrowhead=2
                )
        
        result = {
            'figure': fig.to_dict(),
            'type': 'pca',
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'total_explained_variance': pca.explained_variance_ratio_.sum()
        }
        
        return result
    
    def _create_scatter_3d(self, df, params):
        x = params.get('x')
        y = params.get('y')
        z = params.get('z')
        color = params.get('color')
        size = params.get('size')
        title = params.get('title', '3D Scatter Plot')
        
        if not x or not y or not z:
            raise ValueError("'x', 'y', and 'z' must be specified for 3D scatter plot")
        
        fig = px.scatter_3d(
            df, x=x, y=y, z=z, 
            color=color, size=size,
            title=title,
            labels={
                x: params.get('x_label', x),
                y: params.get('y_label', y),
                z: params.get('z_label', z)
            },
            template='plotly_white'
        )
        
        return {'figure': fig.to_dict(), 'type': 'scatter_3d'}
