const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const protobuf = require('protobufjs');
const diff = require('diff');
const ort = require('onnxruntime-node');

const ONNX_DATA_TYPE_MAP = {
    1: 'Float',
    2: 'UInt8',
    3: 'Int8',
    4: 'UIInt16',
    5: 'Int16',
    6: 'Int32',
    7: 'Int64',
    8: 'String',
    9: 'Bool',
    10: 'Float16',
    11: 'Double',
    12: 'UIInt32',
    13: 'uInt64',
    14: 'Complex64',
    15: 'COMPLEX128',
    16: 'BFLOAT16'
};

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
    console.log('ONNX Model Comparator is activating...');
    
    try {
        let disposable = vscode.commands.registerCommand('onnx-model-comparator.compareModels', async function () {
            console.log('Compare Models command triggered');
            // Ask user to select first ONNX model
            const firstModelUri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { 'ONNX Models': ['onnx'] },
                title: 'Select First ONNX Model'
            });
            
            if (!firstModelUri || firstModelUri.length === 0) {
                return;
            }
            
            // Ask user to select second ONNX model
            const secondModelUri = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { 'ONNX Models': ['onnx'] },
                title: 'Select Second ONNX Model'
            });
            
            if (!secondModelUri || secondModelUri.length === 0) {
                return;
            }
            
            try {
                // Load both models
                const model1Path = firstModelUri[0].fsPath;
                const model2Path = secondModelUri[0].fsPath;
                
                const model1Buffer = fs.readFileSync(model1Path);
                const model2Buffer = fs.readFileSync(model2Path);
                
                // Parse ONNX models
                const protoPath = path.join(__dirname, 'onnx.proto');
                const model1 = await parseOnnxModel(model1Buffer, protoPath);
                const model2 = await parseOnnxModel(model2Buffer, protoPath);

                // Extract inputs, outputs, initializers, nodes, and value_info
                const inputs1 = getInputs(model1);
                const inputs2 = getInputs(model2);
                const outputs1 = getOutputs(model1);
                const outputs2 = getOutputs(model2);
                const inits1 = getInitializers(model1);
                const inits2 = getInitializers(model2);
                const nodes1 = model1.graph.node;
                const nodes2 = model2.graph.node;
                const valueInfo1 = getValueInfo(model1); // Add this
                const valueInfo2 = getValueInfo(model2); // Add this

                // Build maps for all inputs, outputs, initializers, value_info by name
                const allInputs1 = {};
                const allInputs2 = {};
                inputs1.forEach(input => allInputs1[input.name] = input);
                inputs2.forEach(input => allInputs2[input.name] = input);

                const allOutputs1 = {};
                const allOutputs2 = {};
                outputs1.forEach(output => allOutputs1[output.name] = output);
                outputs2.forEach(output => allOutputs2[output.name] = output);

                const allInits1 = {};
                const allInits2 = {};
                inits1.forEach(init => allInits1[init.name] = init);
                inits2.forEach(init => allInits2[init.name] = init);

                const allValueInfo1 = {}; // Add this block
                const allValueInfo2 = {};
                valueInfo1.forEach(vi => allValueInfo1[vi.name] = vi);
                valueInfo2.forEach(vi => allValueInfo2[vi.name] = vi);

                // Generate comparison report
                const comparisonResult = compareOnnxModels(model1, model2);
                
                // Create and show webview with comparison results
                const panel = vscode.window.createWebviewPanel(
                    'onnxComparison',
                    'ONNX Model Comparison',
                    vscode.ViewColumn.One,
                    {
                        enableScripts: true
                    }
                );
                
                // Pass all necessary data to the webview content function
                panel.webview.html = getComparisonWebviewContent(
                    path.basename(model1Path),
                    path.basename(model2Path),
                    comparisonResult,
                    allInputs1, allInputs2,
                    allOutputs1, allOutputs2,
                    allInits1, allInits2,
                    allValueInfo1, allValueInfo2, // Pass value info maps
                    nodes1, nodes2
                );
                
            } catch (error) {
                vscode.window.showErrorMessage(`Error comparing ONNX models: ${error.message}`);
            }
        });

        context.subscriptions.push(disposable);
        console.log('ONNX Model Comparator activated successfully');
    } catch (error) {
        console.error('Error during extension activation:', error);
        throw error; // Re-throw to ensure VS Code sees the activation failure
    }
}

async function parseOnnxModel(buffer, protoPath) {
    const root = await protobuf.load(protoPath);
    const ModelProto = root.lookupType('onnx.ModelProto');
    return ModelProto.decode(buffer);
}

function getInputs(model) {
    return model.graph.input.map(input => ({
        name: input.name,
        shape: input.type.tensorType.shape.dim.map(d => d.dimValue),
        type: input.type.tensorType.elemType
    }));
}

function getOutputs(model) {
    return model.graph.output.map(output => ({
        name: output.name,
        shape: output.type.tensorType.shape.dim.map(d => d.dimValue),
        type: output.type.tensorType.elemType
    }));
}

function getInitializers(model) {
    return model.graph.initializer.map(init => ({
        name: init.name,
        dims: init.dims,
        dataType: init.dataType
    }));
}

function getValueInfo(model) {
    if (!model.graph.valueInfo) return [];
    return model.graph.valueInfo.map(vi => ({
        name: vi.name,
        shape: vi.type?.tensorType?.shape?.dim?.map(d => d.dimValue ?? d.dimParam ?? '?') || [], // Handle different dim types
        type: vi.type?.tensorType?.elemType
    }));
}

function compareArrays(arr1, arr2, key = 'name') {
    const names1 = arr1.map(x => x[key]);
    const names2 = arr2.map(x => x[key]);
    return {
        onlyIn1: arr1.filter(x => !names2.includes(x[key])),
        onlyIn2: arr2.filter(x => !names1.includes(x[key])),
        inBoth: arr1.filter(x => names2.includes(x[key]))
    };
}

/**
 * Compares two ONNX models and returns their differences
 * @param {onnx.ModelProto} model1 
 * @param {onnx.ModelProto} model2 
 */
function compareOnnxModels(model1, model2) {
    const result = {
        modelInfo: compareModelInfo(model1, model2),
        graphs: compareGraphs(model1.graph, model2.graph),
        metadata: compareMetadata(model1, model2)
    };
    
    return result;
}

/**
 * Compares basic model information
 */
function compareModelInfo(model1, model2) {
    // Helper to safely convert potential Long to number for comparison
    const safeToNumber = (val) => {
        if (val && typeof val === 'object' && val.low !== undefined && val.high !== undefined) {
            // Handle protobufjs Long objects
            return val.toNumber ? val.toNumber() : (val.low + (val.high * Math.pow(2, 32)));
        }
        // Handle cases where it might already be a number or string representation
        if (typeof val === 'string') {
            const num = parseInt(val, 10);
            if (!isNaN(num)) return num;
        }
        return val; // Return original value if not a Long object or parsable string
    };

    const irVersion1 = safeToNumber(model1.irVersion);
    const irVersion2 = safeToNumber(model2.irVersion);
    const modelVersion1 = safeToNumber(model1.modelVersion);
    const modelVersion2 = safeToNumber(model2.modelVersion);

    return {
        irVersion: {
            model1: irVersion1, // Store converted value
            model2: irVersion2, // Store converted value
            isDifferent: irVersion1 !== irVersion2 // Compare converted values
        },
        producerName: {
            model1: model1.producerName,
            model2: model1.producerName,
            isDifferent: model1.producerName !== model2.producerName
        },
        producerVersion: {
            model1: model1.producerVersion,
            model2: model2.producerVersion,
            isDifferent: model1.producerVersion !== model2.producerVersion
        },
        domain: {
            model1: model1.domain,
            model2: model2.domain,
            isDifferent: model1.domain !== model2.domain
        },
        modelVersion: {
            model1: modelVersion1, // Store converted value
            model2: modelVersion2, // Store converted value
            isDifferent: modelVersion1 !== modelVersion2 // Compare converted values
        }
        // Add other model properties if needed
    };
}

/**
 * Compares graphs from both models
 */
function compareGraphs(graph1, graph2) {
    if (!graph1 || !graph2) {
        return { error: "One or both models don't have a graph" };
    }

    return {
        // Compare nodes (operators)
        nodes: compareNodes(graph1.node, graph2.node),
        
        // Compare inputs
        inputs: compareIOs(graph1.input, graph2.input),
        
        // Compare outputs
        outputs: compareIOs(graph1.output, graph2.output),
        
        // Compare initializers (weights and constants)
        initializers: compareInitializers(graph1.initializer, graph2.initializer)
    };
}

/**
 * Compares nodes (operators) between two graphs
 */
function compareNodes(nodes1, nodes2) {
    if (!nodes1 || !nodes2) {
        return { error: "One or both graphs don't have nodes" };
    }

    const nodeMap1 = {};
    const nodeMap2 = {};
    
    // Map nodes by name for easier comparison
    nodes1.forEach(node => nodeMap1[node.name || node.output[0]] = node);
    nodes2.forEach(node => nodeMap2[node.name || node.output[0]] = node);
    
    const allNodeNames = new Set([
        ...Object.keys(nodeMap1),
        ...Object.keys(nodeMap2)
    ]);
    
    const nodeDifferences = {};
    
    allNodeNames.forEach(nodeName => {
        const node1 = nodeMap1[nodeName];
        const node2 = nodeMap2[nodeName];
        
        if (!node1) {
            nodeDifferences[nodeName] = { status: 'added', details: node2 };
        } else if (!node2) {
            nodeDifferences[nodeName] = { status: 'removed', details: node1 };
        } else if (node1.opType !== node2.opType) {
            nodeDifferences[nodeName] = {
                status: 'modified',
                opType: { model1: node1.opType, model2: node2.opType },
                inputs: compareArrays(node1.input, node2.input),
                outputs: compareArrays(node1.output, node2.output),
                attributes: compareAttributes(node1.attribute, node2.attribute)
            };
        } else {
            // Check for changes in inputs, outputs, or attributes
            const inputDiff = compareArrays(node1.input, node2.input);
            const outputDiff = compareArrays(node1.output, node2.output);
            const attributeDiff = compareAttributes(node1.attribute, node2.attribute);
            
            if (inputDiff.isDifferent || outputDiff.isDifferent || attributeDiff.some(attr => attr.isDifferent)) {
                nodeDifferences[nodeName] = {
                    status: 'modified',
                    opType: { model1: node1.opType, model2: node2.opType },
                    inputs: inputDiff,
                    outputs: outputDiff,
                    attributes: attributeDiff
                };
            }
        }
    });
    
    return {
        model1Count: nodes1.length,
        model2Count: nodes2.length,
        differences: nodeDifferences
    };
}

/**
 * Compares inputs or outputs between two graphs
 */
function compareIOs(ios1, ios2) {
    if (!ios1 || !ios2) {
        return { error: "One or both graphs don't have the specified IOs" };
    }

    const ioMap1 = {};
    const ioMap2 = {};
    
    // Map IOs by name for easier comparison
    ios1.forEach(io => ioMap1[io.name] = io);
    ios2.forEach(io => ioMap2[io.name] = io);
    
    const allIONames = new Set([
        ...Object.keys(ioMap1),
        ...Object.keys(ioMap2)
    ]);
    
    const ioDifferences = {};
    
    allIONames.forEach(ioName => {
        const io1 = ioMap1[ioName];
        const io2 = ioMap2[ioName];
        
        if (!io1) {
            ioDifferences[ioName] = { status: 'added' };
        } else if (!io2) {
            ioDifferences[ioName] = { status: 'removed' };
        } else if (JSON.stringify(io1.type) !== JSON.stringify(io2.type)) {
            ioDifferences[ioName] = {
                status: 'modified',
                type: {
                    model1: io1.type,
                    model2: io2.type
                }
            };
        }
    });
    
    return {
        model1Count: ios1.length,
        model2Count: ios2.length,
        differences: ioDifferences
    };
}

/**
 * Compares initializers (weights and constants) between two graphs
 */
function compareInitializers(inits1, inits2) {
    if (!inits1 || !inits2) {
        return { error: "One or both graphs don't have initializers" };
    }
    
    const initMap1 = {};
    const initMap2 = {};
    
    // Map initializers by name for easier comparison
    inits1.forEach(init => initMap1[init.name] = init);
    inits2.forEach(init => initMap2[init.name] = init);
    
    const allInitNames = new Set([
        ...Object.keys(initMap1),
        ...Object.keys(initMap2)
    ]);
    
    const initDifferences = {};
    
    allInitNames.forEach(initName => {
        const init1 = initMap1[initName];
        const init2 = initMap2[initName];
        
        if (!init1) {
            initDifferences[initName] = { status: 'added' };
        } else if (!init2) {
            initDifferences[initName] = { status: 'removed' };
        } else {
            // Compare tensor data types
            const dataTypeDiff = init1.dataType !== init2.dataType;
            
            // Compare dimensions
            const dimsDiff = !compareArrays(init1.dims, init2.dims).areEqual;
            
            // Basic check for tensor content difference (this can be improved)
            const contentDiff = init1.rawData && init2.rawData && 
                    init1.rawData.length !== init2.rawData.length;
            
            if (dataTypeDiff || dimsDiff || contentDiff) {
                initDifferences[initName] = {
                    status: 'modified',
                    dataType: {
                        model1: init1.dataType,
                        model2: init2.dataType,
                        isDifferent: dataTypeDiff
                    },
                    dims: {
                        model1: init1.dims,
                        model2: init2.dims,
                        isDifferent: dimsDiff
                    },
                    contentDifferent: contentDiff
                };
            }
        }
    });
    
    return {
        model1Count: inits1.length,
        model2Count: inits2.length,
        differences: initDifferences
    };
}

/**
 * Compares attributes between two nodes
 */
function compareAttributes(attrs1, attrs2) {
    if (!attrs1 && !attrs2) return [];
    if (!attrs1) attrs1 = [];
    if (!attrs2) attrs2 = [];
    
    const attrMap1 = {};
    const attrMap2 = {};
    
    // Map attributes by name for easier comparison
    attrs1.forEach(attr => attrMap1[attr.name] = attr);
    attrs2.forEach(attr => attrMap2[attr.name] = attr);
    
    const allAttrNames = new Set([
        ...Object.keys(attrMap1),
        ...Object.keys(attrMap2)
    ]);
    
    const results = [];
    
    allAttrNames.forEach(attrName => {
        const attr1 = attrMap1[attrName];
        const attr2 = attrMap2[attrName];
        
        if (!attr1) {
            results.push({
                name: attrName,
                status: 'added',
                isDifferent: true
            });
        } else if (!attr2) {
            results.push({
                name: attrName,
                status: 'removed',
                isDifferent: true
            });
        } else {
            // Compare attribute values (simplified, can be expanded)
            const isDifferent = attr1.type !== attr2.type || 
                                JSON.stringify(attr1.f) !== JSON.stringify(attr2.f) ||
                                JSON.stringify(attr1.i) !== JSON.stringify(attr2.i) ||
                                JSON.stringify(attr1.s) !== JSON.stringify(attr2.s) ||
                                JSON.stringify(attr1.t) !== JSON.stringify(attr2.t) ||
                                JSON.stringify(attr1.floats) !== JSON.stringify(attr2.floats) ||
                                JSON.stringify(attr1.ints) !== JSON.stringify(attr2.ints) ||
                                JSON.stringify(attr1.strings) !== JSON.stringify(attr2.strings);
            
            results.push({
                name: attrName,
                status: isDifferent ? 'modified' : 'unchanged',
                isDifferent: isDifferent
            });
        }
    });
    
    return results;
}

/**
 * Compare metadata properties between models
 */
function compareMetadata(model1, model2) {
    // This function can be expanded as needed
    return {
        // Add metadata comparison logic here
    };
}

/**
 * Helper function to compare arrays
 */
function compareArrays(array1, array2) {
    if (!array1) array1 = [];
    if (!array2) array2 = [];
    
    const maxLength = Math.max(array1.length, array2.length);
    let isDifferent = array1.length !== array2.length;
    
    const details = [];
    
    for (let i = 0; i < maxLength; i++) {
        const value1 = i < array1.length ? array1[i] : null;
        const value2 = i < array2.length ? array2[i] : null;
        
        if ((value1 !== value2) || (value1 === null || value2 === null)) {
            isDifferent = true;
            details.push({
                index: i,
                model1: value1,
                model2: value2
            });
        }
    }
    
    return {
        areEqual: !isDifferent,
        isDifferent: isDifferent,
        details: isDifferent ? details : []
    };
}

/**
 * Generate HTML content for the comparison webview
 */
function getComparisonWebviewContent(
    model1Name, model2Name, comparisonResult,
    allInputs1, allInputs2,
    allOutputs1, allOutputs2,
    allInits1, allInits2,
    allValueInfo1, allValueInfo2, // Add value info maps
    nodes1, nodes2
) {
    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ONNX Model Comparison</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    padding: 0 20px;
                    color: var(--vscode-foreground);
                    background-color: var(--vscode-editor-background);
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px 0;
                    border-bottom: 1px solid var(--vscode-panel-border);
                }
                h1 {
                    font-size: 18px;
                    margin: 10px 0;
                }
                h2 {
                    font-size: 16px;
                    margin: 15px 0 10px 0;
                }
                h3 {
                    font-size: 14px;
                    margin: 10px 0 5px 0;
                }
                .section {
                    margin: 20px 0;
                    padding: 10px;
                    border: 1px solid var(--vscode-panel-border);
                    border-radius: 5px;
                }
                .diff-added {
                    background-color: rgba(0, 255, 0, 0.1);
                    color: var(--vscode-gitDecoration-addedResourceForeground, green);
                }
                .diff-removed {
                    background-color: rgba(255, 0, 0, 0.1);
                    color: var(--vscode-gitDecoration-deletedResourceForeground, red);
                }
                .diff-modified {
                    background-color: rgba(255, 255, 0, 0.1);
                    color: var(--vscode-gitDecoration-modifiedResourceForeground, orange);
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }
                th, td {
                    border: 1px solid var(--vscode-panel-border);
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: var(--vscode-editor-inactiveSelectionBackground);
                }
                .summary {
                    font-weight: bold;
                    margin: 5px 0;
                }
                .hidden {
                    display: none;
                }
                .toggle-btn {
                    cursor: pointer;
                    color: var(--vscode-textLink-foreground);
                    margin-left: 5px;
                }
                details {
                    margin: 5px 0;
                }
                summary {
                    cursor: pointer;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ONNX Model Comparison</h1>
            </div>
            
            <div class="section">
                <h2>Models</h2>
                <table>
                    <tr>
                        <th>Model 1</th>
                        <th>Model 2</th>
                    </tr>
                    <tr>
                        <td>${model1Name}</td>
                        <td>${model2Name}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Model Information</h2>
                <table>
                    <tr>
                        <th>Property</th>
                        <th>Model 1</th>
                        <th>Model 2</th>
                        <th>Status</th>
                    </tr>
                    ${generateModelInfoRows(comparisonResult.modelInfo)}
                </table>
            </div>
            
            <div class="section">
                <h2>Graph Comparison</h2>

                <h3>Inputs</h3>
                <div class="summary">
                    Model 1: ${comparisonResult.graphs.inputs.model1Count} inputs |
                    Model 2: ${comparisonResult.graphs.inputs.model2Count} inputs
                </div>
                ${generateIODiffContent(comparisonResult.graphs.inputs.differences, 'input', allInputs1, allInputs2, allInits1, allInits2, allValueInfo1, allValueInfo2)}

                <h3>Outputs</h3>
                <div class="summary">
                    Model 1: ${comparisonResult.graphs.outputs.model1Count} outputs |
                    Model 2: ${comparisonResult.graphs.outputs.model2Count} outputs
                </div>
                ${generateIODiffContent(comparisonResult.graphs.outputs.differences, 'output', allOutputs1, allOutputs2, allInits1, allInits2, allValueInfo1, allValueInfo2)}

                <h3>Nodes (Operators)</h3>
                <input type="text" id="nodeSearchInput" placeholder="Search nodes by name..." onkeyup="filterNodes()" style="margin-bottom: 10px; padding: 5px; width: 98%;">
                <div class="summary">
                    Model 1: ${comparisonResult.graphs.nodes.model1Count} nodes |
                    Model 2: ${comparisonResult.graphs.nodes.model2Count} nodes
                </div>
                ${generateNodesDiffContent( // Update this call
                    comparisonResult.graphs.nodes.differences,
                    allInputs1, allInputs2,
                    allOutputs1, allOutputs2,
                    allInits1, allInits2,
                    allValueInfo1, allValueInfo2, // Pass value info maps
                    nodes1, nodes2
                )}

                <h3>Initializers (Weights and Constants)</h3>
                <div class="summary">
                    Model 1: ${comparisonResult.graphs.initializers.model1Count} initializers |
                    Model 2: ${comparisonResult.graphs.initializers.model2Count} initializers
                </div>
                ${generateInitializersDiffContent(comparisonResult.graphs.initializers.differences, allInits1, allInits2)}
            </div>

            <script>
                function filterNodes() {
                    const input = document.getElementById('nodeSearchInput');
                    const filter = input.value.toUpperCase();
                    const table = document.querySelector('#nodesTable'); // Add an ID to the nodes table
                    if (!table) return; // Exit if table not found
                    const tr = table.getElementsByTagName('tr');

                    // Loop through all table rows (start from 1 to skip header)
                    for (let i = 1; i < tr.length; i++) {
                        const td = tr[i].getElementsByTagName('td')[0]; // Get the first cell (Node Name)
                        if (td) {
                            const txtValue = td.textContent || td.innerText;
                            if (txtValue.toUpperCase().indexOf(filter) > -1) {
                                tr[i].style.display = ""; // Show row
                            } else {
                                tr[i].style.display = "none"; // Hide row
                            }
                        }
                    }
                }

                // Add any other client-side JavaScript functionality here
                function toggleSection(sectionId) {
                    const section = document.getElementById(sectionId);
                    if (section) {
                        section.classList.toggle('hidden');
                    }
                }
            </script>
        </body>
        </html>
    `;
}

/**
 * Generate HTML rows for model info comparison
 */
function generateModelInfoRows(modelInfo) {
    let rows = '';
    
    for (const [property, comparison] of Object.entries(modelInfo)) {
        const status = comparison.isDifferent ? 'diff-modified' : '';
        rows += `
            <tr class="${status}">
                <td>${property}</td>
                <td>${comparison.model1 || 'N/A'}</td>
                <td>${comparison.model2 || 'N/A'}</td>
                <td>${comparison.isDifferent ? 'Different' : 'Same'}</td>
            </tr>
        `;
    }
    
    return rows;
}

/**
 * Generate HTML content for nodes differences
 */
function generateNodesDiffContent(
    differences,
    allInputs1 = {},
    allInputs2 = {},
    allOutputs1 = {},
    allOutputs2 = {},
    allInits1 = {},
    allInits2 = {},
    allValueInfo1 = {},
    allValueInfo2 = {},
    nodes1 = [],
    nodes2 = []
) {
    // Use the order from nodes1, then add any nodes only in nodes2
    const nodeNames = [
        ...nodes1.map(n => n.name || n.output[0]),
        ...nodes2.map(n => n.name || n.output[0]).filter(n => !nodes1.some(m => (m.name || m.output[0]) === n))
    ];

    if (nodeNames.length === 0) {
        return '<p>No differences found in nodes.</p>';
    }

    // Helper to get shape string
    const getShapeString = (name, allInputs, allOutputs, allInits, allValueInfo) => {
        let shape = null;
        if (allInputs[name]?.shape) shape = allInputs[name].shape;
        else if (allOutputs[name]?.shape) shape = allOutputs[name].shape;
        else if (allValueInfo[name]?.shape) shape = allValueInfo[name].shape;
        else if (allInits[name]?.dims) shape = allInits[name].dims;

        if (!shape || !Array.isArray(shape)) {
            return '—'; // Return '—' if shape is null, undefined, or not an array
        }

        // Map dimension values to strings, handling numbers, Longs, strings, null/undefined
        const shapeStr = `[${shape.map(d => {
            if (typeof d === 'number') {
                return d; // Keep numbers as is
            } else if (d && typeof d === 'object' && d.low !== undefined && d.high !== undefined) {
                // Handle protobufjs Long objects (simple conversion for display)
                // Note: This might lose precision for very large numbers, but okay for display
                return d.toNumber ? d.toNumber() : (d.low + (d.high * Math.pow(2, 32)));
            } else if (typeof d === 'string') {
                return d; // Keep symbolic dimensions (strings) as is
            } else {
                return '?'; // Replace null, undefined, or other types with '?'
            }
        }).join(', ')}]`;

        return shapeStr;
    };

    // Helper to format node attributes
    const formatAttributes = (node) => {
        if (!node || !node.attribute || node.attribute.length === 0) return 'None';
        let attrs = '';
        node.attribute.forEach(attr => {
            let value = 'N/A';
            // Check for array types first
            if (attr.ints && attr.ints.length > 0) {
                value = `[${attr.ints.join(', ')}]`;
            } else if (attr.floats && attr.floats.length > 0) {
                value = `[${attr.floats.join(', ')}]`;
            } else if (attr.strings && attr.strings.length > 0) {
                value = `[${attr.strings.map(s => `"${s}"`).join(', ')}]`;
            }
            // Then check for single values
            else if (attr.f !== undefined && attr.f !== null) {
                value = attr.f;
            } else if (attr.i !== undefined && attr.i !== null) {
                value = attr.i;
            } else if (attr.s !== undefined && attr.s !== null) {
                value = `"${attr.s}"`;
            }
            // Add more types if needed (tensors, graphs)
            attrs += `&nbsp;&nbsp;&nbsp;&nbsp;${attr.name}: ${value}<br>`;
        });
        return attrs;
    };

    // Helper to compare and format differences
    const formatDifferences = (node1, node2, allInputs1, allOutputs1, allInits1, allValueInfo1, allInputs2, allOutputs2, allInits2, allValueInfo2) => {
        if (!node1) return 'Added in Model 2';
        if (!node2) return 'Removed from Model 2';

        let diffText = '';

        // Separate inputs and compare
        const inputs1 = node1.input || [];
        const inputs2 = node2.input || [];
        const dataInputs1 = inputs1.filter(name => !allInits1[name]);
        const paramInputs1 = inputs1.filter(name => allInits1[name]);
        const dataInputs2 = inputs2.filter(name => !allInits2[name]);
        const paramInputs2 = inputs2.filter(name => allInits2[name]);

        // Compare Data Input Shapes
        const maxDataInputs = Math.max(dataInputs1.length, dataInputs2.length);
        for (let i = 0; i < maxDataInputs; i++) {
            const shape1 = getShapeString(dataInputs1[i], allInputs1, allOutputs1, allInits1, allValueInfo1);
            const shape2 = getShapeString(dataInputs2[i], allInputs2, allOutputs2, allInits2, allValueInfo2);
            if (shape1 !== shape2) {
                diffText += `Data Input ${i} Shape: ${shape1} -> ${shape2}<br>`;
            }
        }

        // Compare Parameter Input Shapes
        const maxParamInputs = Math.max(paramInputs1.length, paramInputs2.length);
         for (let i = 0; i < maxParamInputs; i++) {
            // Use name for clarity if available
            const name1 = paramInputs1[i] || 'N/A';
            const name2 = paramInputs2[i] || 'N/A';
            const shape1 = getShapeString(name1, allInputs1, allOutputs1, allInits1, allValueInfo1);
            const shape2 = getShapeString(name2, allInputs2, allOutputs2, allInits2, allValueInfo2);
            if (shape1 !== shape2 || name1 !== name2) {
                 // Show name if it differs, otherwise just shapes
                 const diffLabel = name1 === name2 ? `Param Input ${i} (${name1}) Shape` : `Param Input ${i}`;
                 diffText += `${diffLabel}: ${shape1} -> ${shape2}<br>`;
            }
        }


        // Compare Output Shapes
        const outputs1 = node1.output || [];
        const outputs2 = node2.output || [];
        const maxOutputs = Math.max(outputs1.length, outputs2.length);
        for (let i = 0; i < maxOutputs; i++) {
            const shape1 = getShapeString(outputs1[i], allInputs1, allOutputs1, allInits1, allValueInfo1);
            const shape2 = getShapeString(outputs2[i], allInputs2, allOutputs2, allInits2, allValueInfo2);
            if (shape1 !== shape2) {
                diffText += `Output ${i} Shape: ${shape1} -> ${shape2}<br>`;
            }
        }

        // Compare Attributes
        const attrs1 = node1.attribute || [];
        const attrs2 = node2.attribute || [];
        const attrMap1 = {};
        attrs1.forEach(a => attrMap1[a.name] = a);
        const attrMap2 = {};
        attrs2.forEach(a => attrMap2[a.name] = a);
        const allAttrNames = new Set([...Object.keys(attrMap1), ...Object.keys(attrMap2)]);

        allAttrNames.forEach(name => {
            const attr1 = attrMap1[name];
            const attr2 = attrMap2[name];
            const val1Str = JSON.stringify(attr1); // Simple comparison
            const val2Str = JSON.stringify(attr2);
            if (val1Str !== val2Str) {
                 // Extract simple values for display
                 let v1 = 'N/A';
                 if (attr1) {
                     if (attr1.ints?.length) v1 = `[${attr1.ints.join(',')}]`;
                     else if (attr1.floats?.length) v1 = `[${attr1.floats.join(',')}]`;
                     else v1 = attr1.i ?? attr1.f ?? attr1.s ?? v1;
                 }
                 let v2 = 'N/A';
                  if (attr2) {
                     if (attr2.ints?.length) v2 = `[${attr2.ints.join(',')}]`;
                     else if (attr2.floats?.length) v2 = `[${attr2.floats.join(',')}]`;
                     else v2 = attr2.i ?? attr2.f ?? attr2.s ?? v2;
                 }
                 diffText += `${name}: ${v1} -> ${v2}<br>`;
            }
        });

        return diffText || 'No difference';
    };


    let content = '<details open><summary>Show Node Differences (' + nodeNames.length + ')</summary><table id="nodesTable">'; // Add id="nodesTable" here
    content += `
        <tr>
            <th>Node Name</th>
            <th>Model 1</th>
            <th>Model 2</th>
            <th>Difference</th>
        </tr>
    `;

    nodeNames.forEach(nodeName => {
        const diffData = differences[nodeName] || {};
        let node1 = diffData.details && diffData.details.model1 ? diffData.details.model1 : null;
        let node2 = diffData.details && diffData.details.model2 ? diffData.details.model2 : null;

        // Fallback: try to get node1/node2 from diff if not present in details
        if (!node1 && diffData.model1) node1 = diffData.model1;
        if (!node2 && diffData.model2) node2 = diffData.model2;

        // If still not found, try to get from nodes1/nodes2 directly
        if (!node1) node1 = nodes1.find(n => (n.name || n.output[0]) === nodeName);
        if (!node2) node2 = nodes2.find(n => (n.name || n.output[0]) === nodeName);

        let model1Content = '—';
        let model2Content = '—';
        let statusClass = ''; // CSS class for the row

        // Generate content for Model 1
        if (node1) {
            // ... (generate model1Content as before) ...
            const inputs = node1.input || [];
            const dataInputs = inputs.filter(name => !allInits1[name]); // Inputs not in initializers
            const paramInputs = inputs.filter(name => allInits1[name]); // Inputs found in initializers
            const dataInputShapes = dataInputs.map(n => getShapeString(n, allInputs1, allOutputs1, allInits1, allValueInfo1)).join(', ') || '—';
            const paramInputShapes = paramInputs.map(n => `${n}: ${getShapeString(n, allInputs1, allOutputs1, allInits1, allValueInfo1)}`).join('<br>') || 'None';
            const outputShapes = (node1.output || []).map(n => getShapeString(n, allInputs1, allOutputs1, allInits1, allValueInfo1)).join(', ') || '—';
            const attributes = formatAttributes(node1);
            model1Content = `
                <strong>Input Shape:</strong> ${dataInputShapes}<br>
                <strong>Node Attributes:</strong><br>${attributes}
                <strong>Parameter Shapes:</strong><br>&nbsp;&nbsp;&nbsp;&nbsp;${paramInputShapes}<br>
                <strong>Output Shape:</strong> ${outputShapes}
            `;
        }

        // Generate content for Model 2
        if (node2) {
            // ... (generate model2Content as before) ...
            const inputs = node2.input || [];
            const dataInputs = inputs.filter(name => !allInits2[name]);
            const paramInputs = inputs.filter(name => allInits2[name]);
            const dataInputShapes = dataInputs.map(n => getShapeString(n, allInputs2, allOutputs2, allInits2, allValueInfo2)).join(', ') || '—';
            const paramInputShapes = paramInputs.map(n => `${n}: ${getShapeString(n, allInputs2, allOutputs2, allInits2, allValueInfo2)}`).join('<br>') || 'None';
            const outputShapes = (node2.output || []).map(n => getShapeString(n, allInputs2, allOutputs2, allInits2, allValueInfo2)).join(', ') || '—';
            const attributes = formatAttributes(node2);
            model2Content = `
                <strong>Input Shape:</strong> ${dataInputShapes}<br>
                <strong>Node Attributes:</strong><br>${attributes}
                <strong>Parameter Shapes:</strong><br>&nbsp;&nbsp;&nbsp;&nbsp;${paramInputShapes}<br>
                <strong>Output Shape:</strong> ${outputShapes}
            `;
        }

        const differenceDetails = formatDifferences(node1, node2, allInputs1, allOutputs1, allInits1, allValueInfo1, allInputs2, allOutputs2, allInits2, allValueInfo2);

        // Determine status class based on node existence and differences
        if (!node1) {
            statusClass = 'diff-added'; // Node added in Model 2
        } else if (!node2) {
            statusClass = 'diff-removed'; // Node removed from Model 2
        } else if (differenceDetails !== 'No difference') {
            statusClass = 'diff-modified'; // Node exists in both but is modified
        }
        // No class needed if node is unchanged

        content += `
            <tr class="${statusClass}">
                <td>${nodeName}</td>
                <td>${model1Content}</td>
                <td>${model2Content}</td>
                <td>${differenceDetails}</td>
            </tr>
        `;
    });

    content += '</table></details>';
    return content;
}

/**
 * Generate HTML content for I/O differences
 */
function generateIODiffContent(
    differences, ioType,
    allIO1 = {}, allIO2 = {}, // Pass the relevant maps (inputs or outputs)
    allInits1 = {}, allInits2 = {}, // Needed for getShapeString
    allValueInfo1 = {}, allValueInfo2 = {} // Needed for getShapeString
) {
    const keys = Object.keys(differences);

    if (keys.length === 0) {
        return `<p>No differences found in ${ioType}s.</p>`;
    }

    // Re-use or adapt the shape helper from generateNodesDiffContent
    const getShapeString = (name, allIO, allInits, allValueInfo) => {
        let shape = null;
        // Check the primary IO map first
        if (allIO[name]?.shape) shape = allIO[name].shape;
        // Fallback checks (might be relevant if an output is also an initializer or in value_info)
        else if (allValueInfo[name]?.shape) shape = allValueInfo[name].shape;
        else if (allInits[name]?.dims) shape = allInits[name].dims;

        if (!shape || !Array.isArray(shape)) {
            return '—';
        }

        const shapeStr = `[${shape.map(d => {
            if (typeof d === 'number') return d;
            if (d && typeof d === 'object' && d.low !== undefined && d.high !== undefined) {
                return d.toNumber ? d.toNumber() : (d.low + (d.high * Math.pow(2, 32)));
            }
            if (typeof d === 'string') return d;
            return '?';
        }).join(', ')}]`;
        return shapeStr;
    };

    // Use the mapping for data type display
    const getTypeName = t => t ? (ONNX_DATA_TYPE_MAP[t] || t) : '—';

    let content = `<details open><summary>Show ${ioType.charAt(0).toUpperCase() + ioType.slice(1)} Differences (${keys.length})</summary><table>`;
    content += `
        <tr>
            <th>Name</th>
            <th>Model 1</th>
            <th>Model 2</th>
            <th>Difference</th>
        </tr>
    `;

    keys.forEach(ioName => {
        const diff = differences[ioName];
        let statusClass = '';
        let model1Details = '—', model2Details = '—';
        let differenceText = '';

        const io1 = allIO1[ioName];
        const io2 = allIO2[ioName];

        const type1 = io1 ? getTypeName(io1.type) : 'N/A';
        const type2 = io2 ? getTypeName(io2.type) : 'N/A';
        const shape1 = getShapeString(ioName, allIO1, allInits1, allValueInfo1);
        const shape2 = getShapeString(ioName, allIO2, allInits2, allValueInfo2);


        if (io1) {
            model1Details = `<strong>Type:</strong> ${type1}<br><strong>Shape:</strong> ${shape1}`;
        }
        if (io2) {
            model2Details = `<strong>Type:</strong> ${type2}<br><strong>Shape:</strong> ${shape2}`;
        }

        switch (diff.status) {
            case 'added':
                statusClass = 'diff-added';
                differenceText = `Added in Model 2`;
                model1Details = '—'; // Doesn't exist in Model 1
                 break;
            case 'removed':
                statusClass = 'diff-removed';
                differenceText = `Removed from Model 2`;
                model2Details = '—'; // Doesn't exist in Model 2
                break;
            case 'modified':
                statusClass = 'diff-modified';
                if (type1 !== type2) differenceText += `Type: ${type1} -> ${type2}<br>`;
                if (shape1 !== shape2) differenceText += `Shape: ${shape1} -> ${shape2}<br>`;
                if (!differenceText) differenceText = 'Modified (Unknown Detail)'; // Fallback
                break;
            default: // Unchanged (shouldn't appear if only differences are passed, but good practice)
                 model1Details = `<strong>Type:</strong> ${type1}<br><strong>Shape:</strong> ${shape1}`;
                 model2Details = `<strong>Type:</strong> ${type2}<br><strong>Shape:</strong> ${shape2}`;
                 differenceText = 'Same';
                 break;
        }

        content += `
            <tr class="${statusClass}">
                <td>${ioName}</td>
                <td>${model1Details}</td>
                <td>${model2Details}</td>
                <td>${differenceText || diff.status}</td>
            </tr>
        `;
    });

    content += '</table></details>';
    return content;
}

/**
 * Generate HTML content for initializers differences
 */
function generateInitializersDiffContent(differences, allInits1 = {}, allInits2 = {}) {
    const keys = Object.keys(differences);
    if (keys.length === 0) {
        return '<p>No differences found in initializers.</p>';
    }

    let content = '<details open><summary>Show Initializer Differences (' + keys.length + ')</summary><table>';
    content += `
        <tr>
            <th>Name</th>
            <th>Model 1</th>
            <th>Model 2</th>
            <th>Status</th>
        </tr>
    `;

    keys.forEach(initName => {
        const diff = differences[initName];
        let statusClass = '';
        let model1Dims = '', model2Dims = '', model1Type = '', model2Type = '';

        // Try to get dims/type from both models if available
        const init1 = allInits1[initName];
        const init2 = allInits2[initName];

        // Use the mapping for data type display
        const getTypeName = t => t ? (ONNX_DATA_TYPE_MAP[t] || t) : '—';

        switch (diff.status) {
            case 'added':
                statusClass = 'diff-added';
                model1Dims = init1 ? `[${(init1.dims || []).join(', ')}]` : '—';
                model1Type = init1 ? getTypeName(init1.dataType) : '—';
                model2Dims = init2 ? `[${(init2.dims || []).join(', ')}]` : '—';
                model2Type = init2 ? getTypeName(init2.dataType) : '—';
                break;
            case 'removed':
                statusClass = 'diff-removed';
                model1Dims = init1 ? `[${(init1.dims || []).join(', ')}]` : '—';
                model1Type = init1 ? getTypeName(init1.dataType) : '—';
                model2Dims = init2 ? `[${(init2.dims || []).join(', ')}]` : '—';
                model2Type = init2 ? getTypeName(init2.dataType) : '—';
                break;
            case 'modified':
                statusClass = 'diff-modified';
                model1Dims = `[${(diff.dims?.model1 || []).join(', ')}]`;
                model2Dims = `[${(diff.dims?.model2 || []).join(', ')}]`;
                model1Type = getTypeName(diff.dataType?.model1);
                model2Type = getTypeName(diff.dataType?.model2);
                break;
        }

        content += `
            <tr class="${statusClass}">
                <td>${initName}</td>
                <td>
                    <strong>DataType:</strong> ${model1Type}<br>
                    <strong>Dims:</strong> ${model1Dims}
                </td>
                <td>
                    <strong>DataType:</strong> ${model2Type}<br>
                    <strong>Dims:</strong> ${model2Dims}
                </td>
                <td>${diff.status.charAt(0).toUpperCase() + diff.status.slice(1)}</td>
            </tr>
        `;
    });

    content += '</table></details>';
    return content;
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
}