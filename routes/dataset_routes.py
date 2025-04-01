# ...existing code...

from modules.visualization import Visualizer

# ...existing code...

@app.route('/experiment/<int:experiment_id>/dataset/<int:dataset_id>/visualize')
@login_required
def dataset_visualize(experiment_id, dataset_id):
    # ...existing code for retrieving experiment and dataset...
    
    # Create visualization data for template
    visualizer = Visualizer()
    
    # Get visualization types categorized
    viz_types = visualizer.get_visualization_types()
    
    # Create icons and descriptions for each visualization type
    viz_icons = {
        # Basic visualizations
        'scatter': {'icon': 'chart-scatter', 'color': 'blue'},
        'line': {'icon': 'chart-line', 'color': 'green'},
        'bar': {'icon': 'chart-bar', 'color': 'yellow'},
        'histogram': {'icon': 'bars', 'color': 'red'},
        'box': {'icon': 'box', 'color': 'purple'},
        'pie': {'icon': 'chart-pie', 'color': 'indigo'},
        
        # Advanced visualizations
        'violin': {'icon': 'wave-square', 'color': 'amber'},
        'heatmap': {'icon': 'th', 'color': 'red'},
        'scatter_3d': {'icon': 'cube', 'color': 'blue'},
        'surface_3d': {'icon': 'mountain', 'color': 'emerald'},
        'contour': {'icon': 'project-diagram', 'color': 'cyan'},
        'density_contour': {'icon': 'braille', 'color': 'teal'},
        'parallel_coordinates': {'icon': 'stream', 'color': 'purple'},
        'scatter_matrix': {'icon': 'th', 'color': 'indigo'},
        'treemap': {'icon': 'th-large', 'color': 'amber'},
        'sunburst': {'icon': 'sun', 'color': 'orange'},
        'radar': {'icon': 'circle-notch', 'color': 'green'},
        
        # Statistical visualizations
        'correlation': {'icon': 'table', 'color': 'blue'},
        'distribution': {'icon': 'chart-area', 'color': 'green'},
        'qq_plot': {'icon': 'check-double', 'color': 'purple'},
        'residual_plot': {'icon': 'chart-line', 'color': 'red'},
        'pair_plot': {'icon': 'th', 'color': 'yellow'},
        'time_series_decomposition': {'icon': 'chart-line', 'color': 'blue'},
        'cluster_map': {'icon': 'braille', 'color': 'amber'},
    }
    
    # Create descriptions for each visualization type
    viz_descriptions = {
        # Basic visualizations
        'scatter': 'Plot points to show relationships between variables',
        'line': 'Connect points with lines to show trends over a continuous variable',
        'bar': 'Compare values across different categories',
        'histogram': 'Show the distribution of a numeric variable',
        'box': 'Display the distribution and identify outliers',
        'pie': 'Show proportions of a whole as slices of a circle',
        
        # Advanced visualizations
        'violin': 'Show distribution density along with quartiles',
        'heatmap': 'Visualize matrix data with colors representing values',
        'scatter_3d': 'Plot points in three-dimensional space',
        'surface_3d': 'Create a 3D surface from coordinated data points',
        'contour': 'Show isolines of 3D data on a 2D surface',
        'density_contour': 'Show the density of points using contours',
        'parallel_coordinates': 'Compare multiple variables across many observations',
        'scatter_matrix': 'Create a matrix of scatterplots for multiple variables',
        'treemap': 'Display hierarchical data as nested rectangles',
        'sunburst': 'Show hierarchical data using concentric rings',
        'radar': 'Compare multiple variables in a circular layout',
        
        # Statistical visualizations
        'correlation': 'Show correlation coefficients between variables',
        'distribution': 'Visualize data distribution with histogram and KDE',
        'qq_plot': 'Check if data follows a theoretical distribution',
        'residual_plot': 'Analyze the residuals from a regression model',
        'pair_plot': 'Create a matrix of plots showing pairwise relationships',
        'time_series_decomposition': 'Break down a time series into components',
        'cluster_map': 'Clustered heatmap showing correlations or other metrics',
    }
    
    # Create options for form controls
    viz_options = {}
    for category in viz_types:
        for viz_type in viz_types[category]:
            viz_options[viz_type] = visualizer.get_visualization_options(viz_type)
    
    return render_template(
        'dataset_visualize.html',
        experiment=experiment,
        dataset=dataset,
        data_info=data_info,  # Assuming this is defined earlier in the route
        viz_types=viz_types,
        viz_icons=viz_icons,
        viz_descriptions=viz_descriptions,
        viz_options=viz_options
    )

# ...existing code...