// Visualization type definitions

const vizTypes = {
    basic: {
        scatter: 'Scatter Plot',
        line: 'Line Chart',
        bar: 'Bar Chart',
        histogram: 'Histogram',
        box: 'Box Plot',
        pie: 'Pie Chart'
    },
    advanced: {
        violin: 'Violin Plot',
        heatmap: 'Heatmap',
        scatter_3d: '3D Scatter Plot',
        surface_3d: '3D Surface Plot',
        contour: 'Contour Plot',
        density_contour: 'Density Contour',
        parallel_coordinates: 'Parallel Coordinates',
        treemap: 'Treemap',
        sunburst: 'Sunburst Chart',
        radar: 'Radar Chart'
    },
    statistical: {
        correlation: 'Correlation Matrix',
        distribution: 'Distribution Plot',
        qq_plot: 'Q-Q Plot',
        residual_plot: 'Residual Plot',
        pair_plot: 'Pair Plot',
        cluster_map: 'Cluster Map'
    }
};

const vizIcons = {
    scatter: {icon: 'chart-scatter', color: 'blue'},
    line: {icon: 'chart-line', color: 'green'},
    bar: {icon: 'chart-bar', color: 'yellow'},
    histogram: {icon: 'bars', color: 'red'},
    box: {icon: 'box', color: 'purple'},
    pie: {icon: 'chart-pie', color: 'indigo'},
    
    // Advanced visualizations
    violin: {icon: 'wave-square', color: 'amber'},
    heatmap: {icon: 'th', color: 'red'},
    scatter_3d: {icon: 'cube', color: 'blue'},
    surface_3d: {icon: 'mountain', color: 'emerald'},
    contour: {icon: 'project-diagram', color: 'cyan'},
    density_contour: {icon: 'braille', color: 'teal'},
    parallel_coordinates: {icon: 'stream', color: 'purple'},
    treemap: {icon: 'th-large', color: 'amber'},
    sunburst: {icon: 'sun', color: 'orange'},
    radar: {icon: 'circle-notch', color: 'green'},
    
    // Statistical visualizations
    correlation: {icon: 'table', color: 'blue'},
    distribution: {icon: 'chart-area', color: 'green'},
    qq_plot: {icon: 'check-double', color: 'purple'},
    residual_plot: {icon: 'chart-line', color: 'red'},
    pair_plot: {icon: 'th', color: 'yellow'},
    cluster_map: {icon: 'braille', color: 'amber'}
};

const vizDescriptions = {
    // Basic visualizations
    scatter: 'Plot points to show relationships between variables',
    line: 'Connect points with lines to show trends over a continuous variable',
    bar: 'Compare values across different categories',
    histogram: 'Show the distribution of a numeric variable',
    box: 'Display the distribution and identify outliers',
    pie: 'Show proportions of a whole as slices of a circle',
    
    // Advanced visualizations
    violin: 'Show distribution density along with quartiles',
    heatmap: 'Visualize matrix data with colors representing values',
    scatter_3d: 'Plot points in three-dimensional space',
    surface_3d: 'Create a 3D surface from coordinated data points',
    contour: 'Show isolines of 3D data on a 2D surface',
    density_contour: 'Show the density of points using contours',
    parallel_coordinates: 'Compare multiple variables across many observations',
    treemap: 'Display hierarchical data as nested rectangles',
    sunburst: 'Show hierarchical data using concentric rings',
    radar: 'Compare multiple variables in a circular layout',
    
    // Statistical visualizations
    correlation: 'Show correlation coefficients between variables',
    distribution: 'Visualize data distribution with histogram and KDE',
    qq_plot: 'Check if data follows a theoretical distribution',
    residual_plot: 'Analyze the residuals from a regression model',
    pair_plot: 'Create a matrix of plots showing pairwise relationships',
    cluster_map: 'Clustered heatmap showing correlations or other metrics'
};

// Function to populate the HTML elements with visualization cards
function populateVisualizations() {
    // Create cards for each visualization type
    const basicContainer = document.getElementById('basic-viz');
    const advancedContainer = document.getElementById('advanced-viz');
    const statisticalContainer = document.getElementById('statistical-viz');
    
    // Clear containers
    basicContainer.innerHTML = '';
    advancedContainer.innerHTML = '';
    statisticalContainer.innerHTML = '';
    
    // Add visualization cards to each category
    populateCategory(basicContainer, vizTypes.basic);
    populateCategory(advancedContainer, vizTypes.advanced);
    populateCategory(statisticalContainer, vizTypes.statistical);
}

function populateCategory(container, vizTypeList) {
    for (const [type, name] of Object.entries(vizTypeList)) {
        const icon = vizIcons[type] || {icon: 'chart-bar', color: 'indigo'};
        const description = vizDescriptions[type] || `Visualize your data using a ${name}`;
        
        const card = document.createElement('div');
        card.className = 'viz-type-card border rounded-lg p-4 hover:bg-indigo-50 cursor-pointer transition duration-150';
        card.setAttribute('data-type', type);
        
        card.innerHTML = `
            <div class="flex items-center">
                <div class="p-3 rounded-md bg-${icon.color}-100 text-${icon.color}-800">
                    <i class="fas fa-${icon.icon}"></i>
                </div>
                <div class="ml-4">
                    <h3 class="text-md font-medium">${name}</h3>
                    <p class="text-sm text-gray-500">${description}</p>
                </div>
            </div>
        `;
        
        container.appendChild(card);
    }
}
