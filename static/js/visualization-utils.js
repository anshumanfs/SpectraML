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
    
    return plotData;
}

/**
 * Validate a plot data structure to make sure it can be rendered
 * @param {Object} plotData - Parsed plot data
 * @returns {boolean} True if valid
 */
function validatePlotData(plotData) {
    // Must have data array
    if (!plotData || !plotData.data || !Array.isArray(plotData.data)) {
        return false;
    }
    
    // Must have at least one trace
    if (plotData.data.length === 0) {
        return false;
    }
    
    // Each trace should have a type
    const validTraces = plotData.data.filter(trace => 
        trace && typeof trace === 'object' && trace.type
    );
    
    return validTraces.length > 0;
}

/**
 * Safely render a plot with error handling
 * @param {string} containerId - DOM ID of container element
 * @param {Object|string} plotData - Plot data to render
 * @returns {Promise} Promise that resolves when plot is rendered
 */
function renderPlot(containerId, plotData) {
    return new Promise((resolve, reject) => {
        try {
            // Parse the data
            const parsedData = parsePlotData(plotData);
            
            // Validate data structure
            if (!validatePlotData(parsedData)) {
                reject(new Error('Invalid plot data structure'));
                return;
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
