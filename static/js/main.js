document.addEventListener('DOMContentLoaded', function() {
    // Initialize CodeMirror editor
    const codeEditor = CodeMirror.fromTextArea(document.getElementById('code-editor'), {
        mode: 'javascript',  // Using JavaScript mode for now (we'd create a custom PQL mode in a full implementation)
        theme: 'dracula',
        lineNumbers: true,
        // autoCloseBrackets: true,
        autoCloseBrackets: {
            pairs: '()[]{}""\'\'``',
            closeBefore: ')]}\'"`,;:>',
            triples: '',
            explode: '[]{}',
        },
        matchBrackets: true,
        indentUnit: 4,
        tabSize: 4,
        indentWithTabs: false,
        extraKeys: {
            "Tab": function(cm) {
                cm.replaceSelection("    ", "end");
            },
            "Ctrl-Space": "autocomplete"
        },
        hintOptions: {
            hint: pqlHint,
            completeSingle: false,
            alignWithWord: true
        }
    });

    // Setup autocompletion events
    codeEditor.on("keyup", function(cm, event) {
        // Don't activate for backspace, enter, etc.
        const ignoreKeys = [8, 9, 13, 27, 37, 38, 39, 40];
        
        if (!cm.state.completionActive && 
            !ignoreKeys.includes(event.keyCode) && 
            event.key.length === 1) {
            CodeMirror.commands.autocomplete(cm, null, {completeSingle: false});
        }
    });

    // Handle custom snippet insertion
    codeEditor.on("hint", function(cm, data) {
        // When a completion is selected
        if (data && data.snippet) {
            // Cancel default insertion
            data.cancel();
            // Insert our custom snippet
            insertSnippet(cm, data.snippet.text);
        }
    });

    // UI Elements
    const runButton = document.getElementById('run-button');
    const parseButton = document.getElementById('parse-button');
    const clearButton = document.getElementById('clear-button');
    const clearConsoleButton = document.getElementById('clear-console-button');
    const consoleOutput = document.getElementById('console-output');
    const outputContent = document.getElementById('output-content');
    const parseTreeContent = document.getElementById('parse-tree-content');
    const visualizationContent = document.getElementById('visualization-content');
    const samplesMenu = document.getElementById('samples-menu');

    // Load sample code snippets
    loadSampleCodeSnippets();

    // Event listeners
    runButton.addEventListener('click', () => executeCode());
    parseButton.addEventListener('click', () => parseCode());
    clearButton.addEventListener('click', () => clearEditor());
    clearConsoleButton.addEventListener('click', () => clearConsole());

    // Define PQL autocomplete data
    const pqlKeywords = [
        'var', 'follows', 'query', 'load_data', 'cluster', 'find_associations', 
        'classify', 'data', 'fitted_to', 'P', 'E', 'correlation', 'outliers'
    ];

    const pqlDistributions = [
        'Normal', 'LogNormal', 'Poisson', 'Bernoulli', 'EmpiricalDistribution', 
        'Gamma', 'Beta', 'Multinomial'
    ];

    const pqlOperators = [
        'and', 'or', 'not'
    ];

    // Define snippets with placeholders
    const pqlSnippets = {
        'load_data': {
            text: 'load_data(/* "filename.csv" */, name: /* dataset_name */);',
            displayText: 'load_data("filename.csv", name: dataset_name)'
        },
        'var': {
            text: 'var /* variable_name */ follows /* distribution */;',
            displayText: 'var variable_name follows distribution'
        },
        'query P': {
            text: 'query P(/* condition */);',
            displayText: 'query P(condition)'
        },
        'query E': {
            text: 'query E(/* variable */);',
            displayText: 'query E(variable)'
        },
        'cluster': {
            text: 'cluster(/* dataset */, dimensions: [/* dim1 */, /* dim2 */], k: /* number */);',
            displayText: 'cluster(dataset, dimensions: [...], k: n)'
        },
        'find_associations': {
            text: 'find_associations(/* dataset */, min_support: /* 0.3 */, min_confidence: /* 0.8 */);',
            displayText: 'find_associations(dataset, min_support: 0.3, min_confidence: 0.8)'
        },
        'classify': {
            text: 'classify(/* dataset */, target: /* column_name */, features: [/* feature1 */, /* feature2 */]);',
            displayText: 'classify(dataset, target: column, features: [...])'
        }
    };

    // Custom PQL hint function
    function pqlHint(editor) {
        const cursor = editor.getCursor();
        const line = editor.getLine(cursor.line);
        const token = editor.getTokenAt(cursor);
        const startPos = token.type !== null && !/\s/.test(token.string) ? token : { start: cursor.ch, string: '', end: cursor.ch };
        const currentWord = startPos.string;
        
        // Get context before cursor for better suggestions
        const contextBefore = line.slice(0, cursor.ch);
        
        // Prepare list of suggestions
        let list = [];
        
        // Add different types of suggestions based on context
        if (contextBefore.match(/var\s+\w+\s+follows\s+$/)) {
            // After "follows", suggest distributions
            list = pqlDistributions.map(item => ({ text: item, displayText: item }));
        } else if (contextBefore.match(/query\s+$/)) {
            // After "query", suggest query types
            list = [
                { text: 'P(', displayText: 'P()' },
                { text: 'E(', displayText: 'E()' },
                { text: 'correlation(', displayText: 'correlation()' },
                { text: 'outliers(', displayText: 'outliers()' }
            ];
        } else {
            // Default suggestions
            // Add keywords
            pqlKeywords.forEach(keyword => {
                if (keyword.startsWith(currentWord)) {
                    // Check if it's a snippet
                    if (pqlSnippets[keyword]) {
                        list.push(pqlSnippets[keyword]);
                    } else {
                        list.push({ text: keyword, displayText: keyword });
                    }
                }
            });
            
            // Add distributions
            pqlDistributions.forEach(dist => {
                if (dist.startsWith(currentWord)) {
                    list.push({ text: dist, displayText: dist });
                }
            });
            
            // Add operators
            pqlOperators.forEach(op => {
                if (op.startsWith(currentWord)) {
                    list.push({ text: op, displayText: op });
                }
            });
            
            // Add special snippets based on current word
            if ('query'.startsWith(currentWord)) {
                list.push(pqlSnippets['query P']);
                list.push(pqlSnippets['query E']);
            }
        }
        
        return {
            list: list,
            from: CodeMirror.Pos(cursor.line, startPos.start),
            to: CodeMirror.Pos(cursor.line, startPos.end)
        };
    }

    // Function to handle snippet placeholders
    function insertSnippet(editor, text) {
        const placeholderRegex = /\${(\d+):([^}]*)}/g;
        let match;
        let result = text;
        let cursorPos = null;
        let placeholders = [];
        
        // Replace placeholders with their default values and collect positions
        while ((match = placeholderRegex.exec(text)) !== null) {
            const id = parseInt(match[1]);
            const defaultValue = match[2];
            const placeholder = { id, text: defaultValue };
            const startPos = match.index - (text.length - result.length);
            const endPos = startPos + match[0].length;
            
            // Store position information
            placeholder.from = startPos;
            placeholder.to = endPos;
            placeholders.push(placeholder);
            
            // Replace in the result string
            result = result.replace(match[0], defaultValue);
        }
        
        // Insert the text
        const cursor = editor.getCursor();
        editor.replaceRange(result, cursor, cursor);
        
        // If there are placeholders, select the first one
        if (placeholders.length > 0) {
            placeholders.sort((a, b) => a.id - b.id);
            const first = placeholders[0];
            const from = editor.posFromIndex(cursor.ch + first.from);
            const to = editor.posFromIndex(cursor.ch + first.from + first.text.length);
            editor.setSelection(from, to);
        }
    }

    // CSV Upload functionality
    const uploadForm = document.getElementById('upload-form');
    const csvFileInput = document.getElementById('csv-file');
    const uploadSubmitButton = document.getElementById('upload-submit');
    const uploadProgressBar = document.querySelector('.upload-progress .progress-bar');
    const uploadProgressContainer = document.querySelector('.upload-progress');
    const uploadSuccessAlert = document.querySelector('.upload-success');
    const uploadErrorAlert = document.querySelector('.upload-error');
    const csvFilesList = document.getElementById('csv-files-list');
    const uploadModal = document.getElementById('uploadModal');

    // Refresh CSV files list when the modal is opened
    const uploadModalInstance = new bootstrap.Modal(uploadModal);
    uploadModal.addEventListener('shown.bs.modal', function () {
        refreshCsvFilesList();
    });

    // Handle file upload
    uploadSubmitButton.addEventListener('click', function() {
        const file = csvFileInput.files[0];
        if (!file) {
            showUploadError('Please select a CSV file to upload.');
            return;
        }
        
        if (!file.name.endsWith('.csv')) {
            showUploadError('Only CSV files are allowed.');
            return;
        }
        
        uploadFile(file);
    });

    // Function to refresh the list of CSV files
    function refreshCsvFilesList() {
        fetch('/list-csv-files')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (data.files.length === 0) {
                        csvFilesList.innerHTML = '<div class="text-muted">No CSV files available.</div>';
                    } else {
                        csvFilesList.innerHTML = data.files.map(file => 
                            `<button type="button" class="list-group-item list-group-item-action csv-file-item" data-filename="${file}">
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-file-text me-2">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                    <polyline points="14 2 14 8 20 8"></polyline>
                                    <line x1="16" y1="13" x2="8" y2="13"></line>
                                    <line x1="16" y1="17" x2="8" y2="17"></line>
                                    <polyline points="10 9 9 9 8 9"></polyline>
                                </svg>
                                ${file}
                            </button>`
                        ).join('');
                        
                        // Add click event for file items
                        document.querySelectorAll('.csv-file-item').forEach(item => {
                            item.addEventListener('click', function() {
                                const filename = this.dataset.filename;
                                insertFileReferenceToEditor(filename);
                            });
                        });
                    }
                } else {
                    csvFilesList.innerHTML = `<div class="text-danger">Error: ${data.message}</div>`;
                }
            })
            .catch(error => {
                csvFilesList.innerHTML = `<div class="text-danger">Error loading file list: ${error.message}</div>`;
            });
    }

    // Insert file reference to the code editor
    function insertFileReferenceToEditor(filename) {
        // Get the current cursor position
        const cursor = codeEditor.getCursor();
        
        // Insert the file reference at the cursor position
        codeEditor.replaceRange(`load_data("${filename}", name: dataset_name);`, cursor);
        
        // Close the modal and focus back on the editor
        uploadModalInstance.hide();
        codeEditor.focus();
        
        // Log to console
        logToConsole('info', `Added reference to ${filename}`);
    }

    // Upload file using fetch API
    function uploadFile(file) {
        // Reset alerts
        resetUploadAlerts();
        
        // Show progress
        uploadProgressContainer.classList.remove('d-none');
        uploadProgressBar.style.width = '0%';
        
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/upload-csv', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // Set progress to 100% when complete
            uploadProgressBar.style.width = '100%';
            return response.json();
        })
        .then(data => {
            if (data.success) {
                showUploadSuccess(`File ${data.filename} uploaded successfully.`);
                csvFileInput.value = ''; // Clear the file input
                refreshCsvFilesList(); // Refresh the file list
                logToConsole('success', `Uploaded file: ${data.filename}`);
            } else {
                showUploadError(data.message);
                logToConsole('error', `Upload failed: ${data.message}`);
            }
        })
        .catch(error => {
            showUploadError(`Upload failed: ${error.message}`);
            logToConsole('error', `Upload error: ${error.message}`);
        })
        .finally(() => {
            // Hide progress after a short delay
            setTimeout(() => {
                uploadProgressContainer.classList.add('d-none');
            }, 1000);
        });
    }

    // Helper functions for showing upload status
    function resetUploadAlerts() {
        uploadSuccessAlert.classList.add('d-none');
        uploadErrorAlert.classList.add('d-none');
        uploadSuccessAlert.textContent = '';
        uploadErrorAlert.textContent = '';
    }

    function showUploadSuccess(message) {
        uploadSuccessAlert.textContent = message;
        uploadSuccessAlert.classList.remove('d-none');
        uploadErrorAlert.classList.add('d-none');
    }

    function showUploadError(message) {
        uploadErrorAlert.textContent = message;
        uploadErrorAlert.classList.remove('d-none');
        uploadSuccessAlert.classList.add('d-none');
    }

    // Function to load sample code snippets
    function loadSampleCodeSnippets() {
        fetch('/samples')
            .then(response => response.json())
            .then(data => {
                // Clear loading placeholder
                samplesMenu.innerHTML = '';
                
                // Add sample items
                data.samples.forEach((sample, index) => {
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.className = 'dropdown-item';
                    a.href = '#';
                    a.textContent = sample.name;
                    a.dataset.sampleCode = sample.code;
                    
                    a.addEventListener('click', (e) => {
                        e.preventDefault();
                        codeEditor.setValue(sample.code);
                        logToConsole('info', `Loaded sample: ${sample.name}`);
                    });
                    
                    li.appendChild(a);
                    samplesMenu.appendChild(li);
                });
            })
            .catch(error => {
                logToConsole('error', `Failed to load samples: ${error.message}`);
                samplesMenu.innerHTML = '<li><a class="dropdown-item" href="#">Failed to load samples</a></li>';
            });
    }

    // Function to execute PQL code
    function executeCode() {
        const code = codeEditor.getValue();
        if (!code.trim()) {
            logToConsole('error', 'No code to execute.');
            return;
        }
        
        logToConsole('info', 'Executing PQL code...');
        
        // First check syntax
        parseCode(true).then(parseResult => {
            if (!parseResult.success) {
                logToConsole('error', 'Cannot execute code with syntax errors. Please fix them first.');
                return;
            }
            
            // If syntax is correct, execute the code
            fetch('/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ code })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    logToConsole('success', 'Code executed successfully.');
                    displayExecutionResults(data);
                } else {
                    logToConsole('error', `Execution failed: ${data.error}`);
                    outputContent.innerHTML = `
                        <div class="alert alert-danger">
                            <h5>Execution Error</h5>
                            <p>${data.error}</p>
                        </div>
                    `;
                }
            })
            .catch(error => {
                logToConsole('error', `Execution request failed: ${error.message}`);
            });
        });
    }

    // Function to parse PQL code (check syntax)
    function parseCode(silent = false) {
        const code = codeEditor.getValue();
        if (!code.trim()) {
            if (!silent) {
                logToConsole('error', 'No code to parse.');
            }
            return Promise.resolve({ success: false });
        }
        
        if (!silent) {
            logToConsole('info', 'Parsing PQL code...');
        }
        
        return fetch('/parse', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ code })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (!silent) {
                    logToConsole('success', 'Syntax is valid.');
                    // Display parse tree
                    visualizeParseTree(code);
                }
            } else {
                if (!silent) {
                    logToConsole('error', 'Syntax errors found:');
                    data.errors.forEach(error => {
                        logToConsole('error', `Line ${error.line}, Col ${error.column}: ${error.message}`);
                    });
                    
                    // Display errors in parse tree view
                    parseTreeContent.innerHTML = `
                        <div class="alert alert-danger">
                            <h5>Syntax Errors</h5>
                            <ul>
                                ${data.errors.map(error => 
                                    `<li>Line ${error.line}, Col ${error.column}: ${error.message}</li>`
                                ).join('')}
                            </ul>
                        </div>
                    `;
                }
                
                // Highlight errors in editor
                highlightErrors(data.errors);
            }
            return data;
        })
        .catch(error => {
            if (!silent) {
                logToConsole('error', `Parse request failed: ${error.message}`);
            }
            return { success: false };
        });
    }

    // Function to visualize parse tree
    function visualizeParseTree(code) {
        fetch('/visualize_parse_tree', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ code })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Display the parse tree using D3.js
                parseTreeContent.innerHTML = '<div id="tree-diagram" class="tree-container"></div>';
                renderParseTree(data.tree_json);
                
                // Also show the raw parse tree string
                const treeString = document.createElement('div');
                treeString.className = 'mt-3 p-3 bg-dark';
                treeString.style.fontSize = '12px';
                treeString.style.overflowX = 'auto';
                treeString.textContent = data.parse_tree;
                parseTreeContent.appendChild(treeString);
            } else {
                parseTreeContent.innerHTML = `
                    <div class="alert alert-danger">
                        <h5>Failed to Visualize Parse Tree</h5>
                        <p>${data.error || 'Unknown error'}</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            logToConsole('error', `Parse tree visualization failed: ${error.message}`);
        });
    }

    // Function to display execution results
    function displayExecutionResults(data) {
        // Update output tab
        let outputHtml = '';
        if (data.results && data.results.length > 0) {
            outputHtml = '<div class="results-container">';
            data.results.forEach((result, index) => {
                outputHtml += `<div class="result-item p-3 mb-3 bg-dark">`;
                outputHtml += `<h5>${result.type}</h5>`;
                
                // Format result based on type
                switch (result.type) {
                    case 'load_data':
                        outputHtml += `
                            <p>Loaded dataset: ${result.dataset} (${result.rows} rows)</p>
                            <p>Columns: ${result.columns.join(', ')}</p>
                            <h6>Preview:</h6>
                            <div style="overflow-x: auto;">
                                <table class="result-table">
                                    <thead>
                                        <tr>
                                            ${Object.keys(result.preview[0] || {}).map(col => 
                                                `<th>${col}</th>`
                                            ).join('')}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${result.preview.map(row => 
                                            `<tr>${Object.values(row).map(val => 
                                                `<td>${val}</td>`
                                            ).join('')}</tr>`
                                        ).join('')}
                                    </tbody>
                                </table>
                            </div>
                        `;
                        break;
                        
                    case 'variable_declaration':
                        outputHtml += `
                            <p>Defined variable: ${result.variable}</p>
                            <p>Distribution: ${result.distribution}</p>
                            <p>Generated ${result.samples} samples</p>
                            <p>Mean: ${result.mean.toFixed(4)}, Std: ${result.std.toFixed(4)}</p>
                        `;
                        break;
                        
                    case 'variable_assignment':
                        outputHtml += `
                            <p>Assigned ${result.variable} = ${result.value}</p>
                        `;
                        break;
                        
                    case 'query_result':
                        outputHtml += `<p>Query type: ${result.query_type}</p>`;
                        if (result.query_type === 'probability') {
                            if (result.condition) {
                                outputHtml += `<p>P(${result.variable}) = ${result.result.toFixed(4)}</p>`;
                            } else {
                                outputHtml += `<p>P(${result.variable}) = ${result.result.toFixed(4)}</p>`;
                            }
                        } else if (result.query_type === 'expectation') {
                            if (result.condition) {
                                outputHtml += `<p>E(${result.variable} | ${result.condition}) = ${result.result.toFixed(4)}</p>`;
                            } else {
                                outputHtml += `<p>E(${result.variable}) = ${result.result.toFixed(4)}</p>`;
                            }
                        } else if (result.query_type === 'correlation') {
                            outputHtml += `<p>Correlation(${result.variables[0]}, ${result.variables[1]}) = ${result.result.toFixed(4)}</p>`;
                        } else if (result.query_type === 'outliers') {
                            outputHtml += `
                                <p>Found ${result.outlier_count} outliers in ${result.variable}</p>
                                <p>Outlier indices: ${result.outlier_indices.join(', ')}</p>
                            `;
                        }
                        break;
                        
                    case 'clustering_result':
                        outputHtml += `
                            <p>Clustered dataset: ${result.dataset}</p>
                            <p>Dimensions: ${result.dimensions.join(', ')}</p>
                            <p>K = ${result.k}, Silhouette Score = ${result.silhouette_score.toFixed(4)}</p>
                            <p>Cluster sizes: ${result.cluster_sizes.join(', ')}</p>
                        `;
                        break;
                        
                    case 'association_result':
                        outputHtml += `
                            <p>Found association rules in: ${result.dataset}</p>
                            <p>Min support: ${result.min_support}, Min confidence: ${result.min_confidence}</p>
                            <p>Found ${result.rules_count} rules</p>
                            <div style="overflow-x: auto;">
                                <table class="result-table">
                                    <thead>
                                        <tr>
                                            <th>Antecedent</th>
                                            <th>Consequent</th>
                                            <th>Support</th>
                                            <th>Confidence</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${result.rules.map(rule => 
                                            `<tr>
                                                <td>${rule.antecedent}</td>
                                                <td>${rule.consequent}</td>
                                                <td>${rule.support.toFixed(4)}</td>
                                                <td>${rule.confidence.toFixed(4)}</td>
                                            </tr>`
                                        ).join('')}
                                    </tbody>
                                </table>
                            </div>
                        `;
                        break;
                        
                    case 'classification_result':
                        outputHtml += `
                            <p>Classification on dataset: ${result.dataset}</p>
                            <p>Target variable: ${result.target}</p>
                            <p>Classifier: ${result.classifier}</p>
                            <p>Accuracy: ${result.accuracy.toFixed(4)}</p>
                            <h6>Feature Importance:</h6>
                            <div style="overflow-x: auto;">
                                <table class="result-table">
                                    <thead>
                                        <tr>
                                            <th>Feature</th>
                                            <th>Importance</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${result.feature_importance.map(fi => 
                                            `<tr>
                                                <td>${fi.feature}</td>
                                                <td>${fi.importance.toFixed(4)}</td>
                                            </tr>`
                                        ).join('')}
                                    </tbody>
                                </table>
                            </div>
                            <h6>Confusion Matrix:</h6>
                            <table class="result-table">
                                <thead>
                                    <tr>
                                        <th>Actual / Predicted</th>
                                        ${Object.values(result.label_mapping).map(label => `<th>${label}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody>
                                    ${result.confusion_matrix.map((row, i) =>
                                        `<tr>
                                            <th>${result.label_mapping[i]}</th>
                                            ${row.map(val => `<td>${val}</td>`).join('')}
                                        </tr>`
                                    ).join('')}
                                </tbody>
                            </table>
                            
                        `;
                        break;
                        
                    case 'error':
                        outputHtml += `
                            <div class="alert alert-danger">
                                <p>${result.message}</p>
                            </div>
                        `;
                        break;
                        
                    default:
                        outputHtml += `<pre>${JSON.stringify(result, null, 2)}</pre>`;
                }
                
                outputHtml += `</div>`;
            });
            outputHtml += '</div>';
        } else {
            outputHtml = `
                <div class="alert alert-warning">
                    <p>No results returned.</p>
                </div>
            `;
        }
        
        outputContent.innerHTML = outputHtml;
        
        // Update visualization tab
        let visualizationHtml = '<div class="visualization-container">';
        let hasVisualizations = false;
        
        if (data.results && data.results.length > 0) {
            data.results.forEach(result => {
                if (result.visualization) {
                    hasVisualizations = true;
                    visualizationHtml += `
                        <div class="visualization-item">
                            <div class="visualization-title">${result.type} - ${result.variable || result.query_type || ''}</div>
                            <img class="visualization-image" src="data:image/png;base64,${result.visualization}" alt="Visualization">
                        </div>
                    `;
                }
            });
        }
        
        visualizationHtml += '</div>';
        
        if (hasVisualizations) {
            visualizationContent.innerHTML = visualizationHtml;
        } else {
            visualizationContent.innerHTML = `
                <div class="alert alert-secondary">
                    <p>No visualizations available for this query.</p>
                </div>
            `;
        }
    }

    // Function to highlight syntax errors in editor
    function highlightErrors(errors) {
        // Clear any existing error markers
        codeEditor.doc.getAllMarks().forEach(mark => mark.clear());
        
        errors.forEach(error => {
            const line = error.line - 1;  // CodeMirror lines are 0-indexed
            const startCol = error.column;
            
            // Get the line text
            const lineText = codeEditor.doc.getLine(line);
            
            // Find appropriate end column (end of current token or end of line)
            let endCol = startCol + 1;
            while (endCol < lineText.length && /\w/.test(lineText[endCol])) {
                endCol++;
            }
            
            // Mark the error in the editor
            const from = { line, ch: startCol };
            const to = { line, ch: endCol };
            
            codeEditor.doc.markText(from, to, {
                className: 'syntax-error',
                title: error.message
            });
        });
    }

    // Function to log messages to the console
    function logToConsole(type, message) {
        const line = document.createElement('div');
        line.className = `${type}-line`;
        
        const timestamp = new Date().toLocaleTimeString();
        line.textContent = `[${timestamp}] ${message}`;
        
        consoleOutput.appendChild(line);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    // Function to clear editor
    function clearEditor() {
        codeEditor.setValue('');
        logToConsole('info', 'Editor cleared.');
    }

    // Function to clear console
    function clearConsole() {
        consoleOutput.innerHTML = '';
        logToConsole('info', 'Console cleared.');
    }

    // Initialize with a syntax check
    setTimeout(() => {
        parseCode(true);
    }, 500);
});
