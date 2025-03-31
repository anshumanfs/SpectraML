/**
 * Visualization utilities for SpectraML
 */

/**
 * Parse plot data ensuring it's in the correct format
 * @param {Object|string} plotData - Plot data that might be a string or object
 * @returns {Object} Parsed plot data object
 */
function parsePlotData(plotData) {
    // If it's a string, parse it
    if (typeof plotData === 'string') {
        try {
            return JSON.parse(plotData);
        } catch (err) {
            console.error('Error parsing plot data string', err);
            throw new Error('Invalid plot data format (parsing error)');
        }
    }
    
    // Otherwise, it should already be an object
    if (typeof plotData !== 'object') {
        throw new Error('Invalid plot data format (not an object)');
    }
    
    // Check if it's a contour plot with potential empty data array
    const parsedData = typeof plotData === 'string' ? JSON.parse(plotData) : plotData;
    
    // Special handling for empty data arrays in contour plots
    if (parsedData && Array.isArray(parsedData.data) && parsedData.data.length === 0) {
        // Check if this is supposed to be a contour plot based on layout
        if (parsedData.layout && (
            parsedData.layout.title?.text?.toLowerCase().includes('contour') ||
            parsedData.layout.contour ||
            parsedData._requestParams?.viz_type?.includes('contour')
        )) {
            console.log("Detected empty contour plot data array - creating fallback contour");
            // Create a minimal valid contour data structure
            parsedData.data = [{
                type: 'contour',
                z: [[0, 0], [0, 0]],  // Minimal 2D array
                x: [0, 1],
                y: [0, 1],
                contours: {
                    coloring: 'heatmap'
                },
                colorscale: 'Viridis',
                showscale: false
            }];
        }
    }
    
    return parsedData;
}

/**
 * Validate a plot data structure to make sure it can be rendered
 * @param {Object} plotData - Parsed plot data
 * @returns {boolean} True if valid
 */
function validatePlotData(plotData) {
    // Must have data array
    if (!plotData || !plotData.data || !Array.isArray(plotData.data)) {
        console.error("Plot data missing required 'data' array property");
        return false;
    }
    
    // Must have at least one trace
    if (plotData.data.length === 0) {
        // Special case for contour plots - auto-generate minimal data
        if (plotData.layout && (
            plotData.layout.title?.text?.toLowerCase().includes('contour') ||
            plotData._requestParams?.viz_type?.includes('contour')
        )) {
            console.log("Empty data array for contour plot - will create minimal trace");
            plotData.data = [{
                type: 'contour',
                z: [[0, 0], [0, 0]],
                colorscale: 'Viridis'
            }];
            return true;
        }
        
        console.error("Plot data has empty 'data' array");
        return false;
    }
    
    // Analyze traces
    let hasValidTrace = false;
    plotData.data.forEach((trace, index) => {
        if (!trace || typeof trace !== 'object') {
            console.warn(`Trace ${index} is not a valid object`);
            return;
        }
        
        // Check trace type exists
        if (!trace.type) {
            console.warn(`Trace ${index} missing required 'type' property`);
            return;
        }
        
        // Box plots are valid with just y values
        if (trace.type === 'box') {
            if (Array.isArray(trace.y) && trace.y.length > 0) {
                console.log(`Valid box plot trace found at index ${index} with ${trace.y.length} data points`);
                hasValidTrace = true;
            } else if (trace.y === undefined && trace.name) {
                // Some box plots might use statistical values directly
                console.log(`Box plot trace at index ${index} has no y array but has name: ${trace.name}. This may still work.`);
                hasValidTrace = true;
            } else {
                console.warn(`Box plot trace at index ${index} has invalid or missing y data`);
            }
            return;
        }
        
        // Handle 3D plots specially
        if (trace.type === 'scatter3d') {
            console.log(`Found 3D scatter trace at index ${index}`);
            
            // Check if it has valid coordinates
            const hasX = Array.isArray(trace.x) && trace.x.length > 0;
            const hasY = Array.isArray(trace.y) && trace.y.length > 0;
            const hasZ = Array.isArray(trace.z) && trace.z.length > 0;
            
            if (hasX && hasY && hasZ) {
                console.log(`Valid 3D scatter with ${trace.x.length} points`);
                hasValidTrace = true;
            } else {
                console.warn(`3D scatter at index ${index} missing required coordinates`);
                if (!hasX) console.warn('Missing x coordinates');
                if (!hasY) console.warn('Missing y coordinates');
                if (!hasZ) console.warn('Missing z coordinates');
            }
        }
        // For other trace types, check appropriate data properties
        else if (trace.type === 'pie') {
            if (Array.isArray(trace.values) && trace.values.length > 0) {
                hasValidTrace = true;
            } else {
                console.warn(`Pie chart trace at index ${index} has invalid or missing values data`);
            }
        } else if (trace.type === 'heatmap' || trace.type === 'contour' || trace.type === 'contourcarpet') {
            // More lenient validation for contour plots
            if (trace.type === 'contour') {
                // Check if z data is present
                if (Array.isArray(trace.z) && trace.z.length > 0) {
                    console.log(`Valid contour plot found with z data length: ${trace.z.length}`);
                    hasValidTrace = true;
                } 
                // If no z data, check if it has valid x/y data for density contours
                else if ((Array.isArray(trace.x) && trace.x.length > 0) && 
                         (Array.isArray(trace.y) && trace.y.length > 0)) {
                    console.log(`Valid density contour found with x/y data`);
                    hasValidTrace = true;
                }
                // For completely empty contour plots, attempt to create minimal data
                else {
                    console.warn(`Empty contour plot trace detected at index ${index} - will try fallback rendering`);
                    // Create minimal data for an empty plot
                    trace.z = [[0, 0], [0, 0]];
                    trace.x = [0, 1];
                    trace.y = [0, 1];
                    trace.showscale = false;
                    trace.contours = trace.contours || {coloring: 'heatmap'};
                    hasValidTrace = true;
                }
            }
            // Regular validation for heatmaps
            else if (trace.type === 'heatmap') {
                if (Array.isArray(trace.z) && trace.z.length > 0) {
                    // For heatmaps, we need to ensure z is a 2D array with data
                    if (Array.isArray(trace.z[0])) {
                        const zRowLength = trace.z[0].length;
                        console.log(`Valid heatmap found with dimensions: ${trace.z.length}x${zRowLength}`);
                        
                        // Ensure x and y arrays are the right length if provided
                        if (trace.x && trace.x.length !== zRowLength) {
                            console.warn(`Heatmap x-axis length (${trace.x.length}) doesn't match z data width (${zRowLength})`);
                        }
                        
                        if (trace.y && trace.y.length !== trace.z.length) {
                            console.warn(`Heatmap y-axis length (${trace.y.length}) doesn't match z data height (${trace.z.length})`);
                        }
                        
                        hasValidTrace = true;
                    } else {
                        console.warn(`Heatmap trace at index ${index} has invalid z data (not a 2D array)`);
                        console.log("Z data structure:", trace.z);
                    }
                } else {
                    console.warn(`${trace.type} trace at index ${index} has invalid or missing z data`);
                    console.log("Z data is:", trace.z);
                }
            } 
            // Handle other contour types
            else {
                if (Array.isArray(trace.z) && trace.z.length > 0) {
                    console.log(`Valid ${trace.type} found with z data`);
                    hasValidTrace = true;
                } else {
                    console.warn(`${trace.type} trace at index ${index} has invalid or missing z data`);
                }
            }
        } else {
            const hasX = Array.isArray(trace.x) && trace.x.length > 0;
            const hasY = Array.isArray(trace.y) && trace.y.length > 0;
            
            if (hasX || hasY) {
                hasValidTrace = true;
            } else {
                console.warn(`Trace ${index} (${trace.type}) has invalid or missing x/y data`);
            }
        }
    });
    
    return hasValidTrace;
}

/**
 * Safely render a plot with error handling and fixes for common issues
 * @param {string} containerId - DOM ID of container element
 * @param {Object|string} plotData - Plot data to render
 * @returns {Promise} Promise that resolves when plot is rendered
 */
function renderPlot(containerId, plotData) {
    return new Promise((resolve, reject) => {
        try {
            // First save original request parameters if they exist
            const originalParams = {};
            if (typeof plotData === 'string') {
                try {
                    // Extract viz_type from the title if available
                    const parsedData = JSON.parse(plotData);
                    if (parsedData.layout && parsedData.layout.title && parsedData.layout.title.text) {
                        const titleText = parsedData.layout.title.text;
                        // Check if title contains viz type
                        if (titleText.toLowerCase().includes('contour')) {
                            originalParams.viz_type = 'contour';
                        }
                    }
                } catch (e) {
                    console.warn("Failed to extract visualization type from title");
                }
            }
            
            // Parse the data and attach original params
            const parsedData = parsePlotData(plotData);
            parsedData._requestParams = originalParams;
            
            // Log the data structure for debugging
            console.log("Plot data structure:", parsedData);
            
            // Special handling for different plot types
            if (parsedData.data && parsedData.data.length > 0) {
                const trace = parsedData.data[0];
                
                // Special handling for heatmaps
                if (trace.type === 'heatmap') {
                    console.log("Heatmap detected - applying special handling");
                    
                    // Fix missing or empty x/y arrays by creating default ones
                    if (!trace.x || !Array.isArray(trace.x) || trace.x.length === 0) {
                        if (trace.z && trace.z[0]) {
                            console.log("Adding default x-axis values to heatmap");
                            trace.x = Array.from({length: trace.z[0].length}, (_, i) => `Column ${i+1}`);
                        }
                    }
                    
                    if (!trace.y || !Array.isArray(trace.y) || trace.y.length === 0) {
                        if (trace.z) {
                            console.log("Adding default y-axis values to heatmap");
                            trace.y = Array.from({length: trace.z.length}, (_, i) => `Row ${i+1}`);
                        }
                    }
                    
                    // Ensure all z values are valid numbers
                    if (trace.z) {
                        trace.z = trace.z.map(row => row.map(v => isNaN(v) ? 0 : v));
                    }
                }
                
                // Special handling for 3D plots
                else if (trace.type === 'scatter3d') {
                    console.log("3D scatter plot detected - applying special handling");
                    
                    // Ensure scene is properly configured
                    if (!parsedData.layout) parsedData.layout = {};
                    if (!parsedData.layout.scene) {
                        parsedData.layout.scene = {
                            aspectmode: 'cube',
                            camera: {
                                eye: {x: 1.5, y: 1.5, z: 1.5}
                            }
                        };
                    }
                    
                    // Adjust height for better 3D visualization
                    parsedData.layout.height = parsedData.layout.height || 650;
                    
                    console.log("Enhanced 3D scatter plot layout:", parsedData.layout);
                }
                
                // Special handling for contour plots - more comprehensive enhancement
                else if (trace.type === 'contour' || trace.type === 'contourcarpet') {
                    console.log("Contour plot detected - applying enhanced handling");
                    
                    // Ensure layout has proper configuration for contours
                    if (!parsedData.layout) parsedData.layout = {};
                    
                    // Set more reasonable contour defaults
                    parsedData.layout.title = parsedData.layout.title || {text: 'Contour Plot'};
                    
                    // Add contour-specific layout options if not present
                    if (!parsedData.layout.contour) {
                        parsedData.layout.contour = {
                            showlabels: true,
                            labelfont: {
                                family: 'Raleway',
                                size: 12,
                                color: 'white',
                            }
                        };
                    }
                    
                    // Apply special handling for contour traces
                    if (trace.contours && !trace.contours.coloring) {
                        trace.contours.coloring = 'heatmap';
                    }
                    
                    // Force a more compatible color scale if not specified
                    if (!trace.colorscale) {
                        trace.colorscale = 'Viridis';
                    }
                    
                    console.log("Enhanced contour plot layout and trace options");
                }
            }
            
            // Validate data structure
            if (!validatePlotData(parsedData)) {
                // Create a simple fallback plot instead of rejecting outright
                console.warn('Invalid plot data structure - creating fallback visualization');
                
                // Prepare a fallback message plot
                parsedData.data = [{
                    type: 'scatter',
                    mode: 'text',
                    x: [0],
                    y: [0],
                    text: ['No valid data available for this visualization'],
                    textfont: {
                        color: 'red',
                        size: 14
                    }
                }];
                
                parsedData.layout = {
                    title: {
                        text: 'Visualization Error - No Valid Data'
                    },
                    xaxis: {
                        visible: false
                    },
                    yaxis: {
                        visible: false
                    },
                    annotations: [{
                        x: 0,
                        y: -0.2,
                        xref: 'paper',
                        yref: 'paper',
                        text: 'Try different parameters or check your data',
                        showarrow: false,
                        font: {
                            size: 12,
                            color: '#666'
                        }
                    }]
                };
            }
            
            // Set default layout if none provided
            const layout = parsedData.layout || {
                title: {text: 'Visualization'},
                margin: {l: 50, r: 50, t: 50, b: 50},
                autosize: true
            };
            
            // Configure Plotly options
            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            };
            
            // Render the plot
            Plotly.newPlot(containerId, parsedData.data, layout, config)
                .then(() => {
                    console.log(`Plot rendered successfully in ${containerId}`);
                    resolve();
                })
                .catch(err => {
                    console.error('Error rendering plot with Plotly', err);
                    reject(err);
                });
        } catch (err) {
            console.error('Error preparing plot data', err);
            reject(err);
        }
    });
}

/**
 * Show an error message in place of a plot
 * @param {string} containerId - DOM ID of container element
 * @param {string} message - Error message to display
 */
function showPlotError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="flex flex-col justify-center items-center h-full text-red-500">
                <i class="fas fa-exclamation-circle text-3xl mb-2"></i>
                <span>Error: ${message}</span>
                <p class="mt-2 text-sm text-gray-500">Please try different parameters or contact support.</p>
            </div>
        `;
    }
}

/**
 * Show a warning message in place of a plot
 * @param {string} containerId - DOM ID of container element
 * @param {string} message - Warning message to display
 */
function showPlotWarning(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="flex flex-col items-center justify-center h-full text-amber-500">
                <i class="fas fa-exclamation-triangle text-3xl mb-2"></i>
                <span>${message}</span>
                <p class="mt-2 text-gray-500">Try selecting different columns or check if your data contains valid values.</p>
            </div>
        `;
    }
}

/**
 * Display debug information for a plot
 * @param {Object|string} plotData - The raw plot data
 * @param {Object} params - Visualization parameters used
 */
function showPlotDebugInfo(plotData, params) {
    try {
        const rawData = typeof plotData === 'string' 
            ? JSON.parse(plotData) 
            : plotData;
        
        const debugModal = document.createElement('div');
        debugModal.className = 'fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50';
        debugModal.innerHTML = `
            <div class="bg-white rounded-lg w-3/4 max-h-3/4 overflow-auto p-4">
                <div class="flex justify-between items-center border-b pb-2 mb-4">
                    <h3 class="text-lg font-medium">Plot Debug Information</h3>
                    <button class="text-gray-500 hover:text-gray-700" onclick="this.parentNode.parentNode.parentNode.remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="mb-4">
                    <h4 class="font-medium text-gray-700 mb-2">Visualization Parameters:</h4>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        ${Object.entries(params).map(([key, value]) => 
                            `<p>${key}: <span class="font-mono text-sm">${value !== undefined ? value : 'None'}</span></p>`
                        ).join('')}
                    </div>
                </div>
                <h4 class="font-medium text-gray-700 mb-2">Raw Plot Data:</h4>
                <pre class="text-xs overflow-auto p-2 bg-gray-100 rounded max-h-96">${JSON.stringify(rawData, null, 2)}</pre>
            </div>
        `;
        document.body.appendChild(debugModal);
    } catch (err) {
        console.error('Error displaying debug info:', err);
        alert('Error parsing plot data for debug display');
    }
}
