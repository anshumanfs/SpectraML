/**
 * Utility functions for feature engineering operations
 */

/**
 * Get a human-readable description of a feature engineering operation
 * @param {Object} operation - The operation object
 * @returns {String} Human-readable description
 */
function getOperationDescription(operation) {
    const params = operation.params || {};
    let description = '';
    
    switch (operation.type) {
        case 'filter_rows':
            if (params.operation === 'is_null') {
                description = `Filter rows where '${params.column}' is null`;
            } else if (params.operation === 'is_not_null') {
                description = `Filter rows where '${params.column}' is not null`;
            } else {
                description = `Filter rows where '${params.column}' ${params.operation} '${params.value}'`;
            }
            break;
            
        case 'drop_columns':
            const columns = Array.isArray(params.columns) ? params.columns.join(', ') : params.columns;
            description = `Drop columns: ${columns}`;
            break;
            
        case 'impute_missing':
            const impute_columns = Array.isArray(params.columns) ? params.columns.join(', ') : params.columns;
            let strategyDesc = params.strategy;
            if (params.strategy === 'constant' && params.constant_value) {
                strategyDesc = `constant (${params.constant_value})`;
            }
            description = `Impute missing values in ${impute_columns} using ${strategyDesc}`;
            break;
            
        case 'scale_normalize':
            const scale_columns = Array.isArray(params.columns) ? params.columns.join(', ') : params.columns;
            description = `Scale ${scale_columns} using ${params.method} scaling`;
            break;
            
        case 'one_hot_encode':
            const encode_columns = Array.isArray(params.columns) ? params.columns.join(', ') : params.columns;
            description = `One-hot encode ${encode_columns}`;
            break;
            
        case 'bin_numeric':
            description = `Bin '${params.column}' into ${params.num_bins} ${params.strategy} bins`;
            break;
            
        case 'log_transform':
            const log_columns = Array.isArray(params.columns) ? params.columns.join(', ') : params.columns;
            description = `Apply log transform (base ${params.base}) to ${log_columns}`;
            break;
            
        case 'pca':
            const pca_columns = Array.isArray(params.columns) ? params.columns.join(', ') : params.columns;
            description = `Apply PCA to ${pca_columns} (${params.n_components} components)`;
            break;
            
        case 'create_datetime_features':
            const components = Array.isArray(params.components) ? params.components.join(', ') : params.components;
            description = `Extract ${components} from '${params.column}'`;
            break;
            
        case 'text_extraction':
            const features = Array.isArray(params.extract) ? params.extract.join(', ') : params.extract;
            description = `Extract ${features} from '${params.column}'`;
            break;
            
        case 'polynomial_features':
            const poly_columns = Array.isArray(params.columns) ? params.columns.join(', ') : params.columns;
            description = `Create polynomial features (degree ${params.degree}) from ${poly_columns}`;
            break;
            
        default:
            description = `${operation.name} operation`;
    }
    
    return description;
}

/**
 * Format parameter value for display
 * @param {*} value - The parameter value
 * @param {String} type - The parameter type
 * @returns {String} Formatted value
 */
function formatParameterValue(value, type) {
    if (value === null || value === undefined) {
        return 'None';
    }
    
    if (Array.isArray(value)) {
        if (value.length === 0) {
            return '[]';
        } else if (value.length <= 3) {
            return value.join(', ');
        } else {
            return `${value.slice(0, 3).join(', ')}... (${value.length} total)`;
        }
    }
    
    if (type === 'column' || type === 'text') {
        return value;
    }
    
    if (type === 'number') {
        return Number(value).toString();
    }
    
    return String(value);
}

/**
 * Format a result summary from feature engineering operation
 * @param {Object} result - The operation result
 * @returns {String} Formatted summary
 */
function formatOperationResult(result) {
    let summary = [];
    
    // Check for common result properties
    if (result.rows_before !== undefined && result.rows_after !== undefined) {
        const rowDiff = result.rows_before - result.rows_after;
        if (rowDiff > 0) {
            summary.push(`Removed ${rowDiff} rows`);
        } else if (rowDiff < 0) {
            summary.push(`Added ${Math.abs(rowDiff)} rows`);
        } else {
            summary.push('No change in row count');
        }
    }
    
    if (result.columns_before !== undefined && result.columns_after !== undefined) {
        const colDiff = result.columns_after - result.columns_before;
        if (colDiff > 0) {
            summary.push(`Added ${colDiff} columns`);
        } else if (colDiff < 0) {
            summary.push(`Removed ${Math.abs(colDiff)} columns`);
        } else {
            summary.push('No change in column count');
        }
    }
    
    // Operation-specific summaries
    switch (result.type) {
        case 'filter_rows':
            if (result.rows_removed !== undefined) {
                summary = [`Removed ${result.rows_removed} rows`];
            }
            break;
            
        case 'drop_columns':
            if (result.columns_dropped !== undefined) {
                const cols = Array.isArray(result.columns_dropped) ? result.columns_dropped : [result.columns_dropped];
                summary = [`Dropped ${cols.length} columns`];
            }
            break;
            
        case 'impute_missing':
            if (result.columns_imputed !== undefined) {
                const cols = Array.isArray(result.columns_imputed) ? result.columns_imputed : [result.columns_imputed];
                summary = [`Imputed missing values in ${cols.length} columns`];
            }
            break;
            
        case 'one_hot_encode':
            if (result.columns_encoded !== undefined && result.new_columns_count !== undefined) {
                const cols = Array.isArray(result.columns_encoded) ? result.columns_encoded : [result.columns_encoded];
                summary = [`Encoded ${cols.length} columns, created ${result.new_columns_count} new columns`];
            }
            break;
            
        case 'create_datetime_features':
            if (result.components_added !== undefined) {
                const comps = Array.isArray(result.components_added) ? result.components_added : [result.components_added];
                summary = [`Created ${comps.length} datetime features`];
            }
            break;
            
        case 'pca':
            if (result.explained_variance_ratio !== undefined) {
                const totalVar = result.explained_variance_ratio.reduce((a, b) => a + b, 0);
                summary = [`Created ${result.n_components} components explaining ${(totalVar * 100).toFixed(2)}% of variance`];
            }
            break;
    }
    
    return summary.join(', ');
}

/**
 * Create a visualization of a feature engineering operation
 * @param {String} elementId - ID of the container element
 * @param {Object} operation - The operation object
 * @param {Object} result - The operation result
 */
function visualizeOperation(elementId, operation, result) {
    const container = document.getElementById(elementId);
    if (!container) return;
    
    // Clear previous content
    container.innerHTML = '';
    
    // Different visualizations based on operation type
    switch (operation.type) {
        case 'filter_rows':
            // Simple bar chart showing rows before/after
            const filterViz = document.createElement('div');
            filterViz.className = 'h-32';
            container.appendChild(filterViz);
            
            const rowsBefore = result.rows_before || 0;
            const rowsAfter = result.rows_after || 0;
            
            const data = [{
                x: ['Before', 'After'],
                y: [rowsBefore, rowsAfter],
                type: 'bar',
                marker: {
                    color: ['rgb(158,202,225)', 'rgb(107,174,214)']
                }
            }];
            
            const layout = {
                title: 'Rows Before/After Filtering',
                xaxis: { title: 'Stage' },
                yaxis: { title: 'Row Count' },
                autosize: true,
                margin: { l: 50, r: 30, b: 40, t: 40 }
            };
            
            Plotly.newPlot(filterViz, data, layout, {responsive: true, displayModeBar: false});
            break;
            
        case 'pca':
            // Bar chart of explained variance by component
            if (result.explained_variance_ratio) {
                const pcaViz = document.createElement('div');
                pcaViz.className = 'h-32';
                container.appendChild(pcaViz);
                
                const components = Array.from({length: result.explained_variance_ratio.length}, (_, i) => `PC${i+1}`);
                
                const data = [{
                    x: components,
                    y: result.explained_variance_ratio.map(v => v * 100), // Convert to percentage
                    type: 'bar',
                    marker: {
                        color: 'rgb(107,174,214)'
                    }
                }];
                
                const layout = {
                    title: 'Explained Variance by Component',
                    xaxis: { title: 'Component' },
                    yaxis: { title: 'Variance Explained (%)' },
                    autosize: true,
                    margin: { l: 50, r: 30, b: 40, t: 40 }
                };
                
                Plotly.newPlot(pcaViz, data, layout, {responsive: true, displayModeBar: false});
            }
            break;
            
        // Add more visualizations for other operation types
            
        default:
            // Default visualization showing column count change
            const defaultViz = document.createElement('div');
            defaultViz.className = 'h-32';
            container.appendChild(defaultViz);
            
            const colsBefore = result.columns_before || 0;
            const colsAfter = result.columns_after || 0;
            
            const defaultData = [{
                x: ['Before', 'After'],
                y: [colsBefore, colsAfter],
                type: 'bar',
                marker: {
                    color: ['rgb(253,208,162)', 'rgb(253,174,107)']
                }
            }];
            
            const defaultLayout = {
                title: 'Columns Before/After Operation',
                xaxis: { title: 'Stage' },
                yaxis: { title: 'Column Count' },
                autosize: true,
                margin: { l: 50, r: 30, b: 40, t: 40 }
            };
            
            Plotly.newPlot(defaultViz, defaultData, defaultLayout, {responsive: true, displayModeBar: false});
            break;
    }
}
