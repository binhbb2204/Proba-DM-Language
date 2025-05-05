// Render a parse tree using D3.js
// Enhanced tree visualization with view mode toggle
function renderParseTree(treeData, containerId = "tree-diagram", viewMode = "fit") {
    if (!treeData) return;
    
    // Clear previous tree
    d3.select(`#${containerId}`).html("");
    
    // Add view mode toggle controls
    const controlsDiv = d3.select(`#${containerId}`)
        .append("div")
        .attr("class", "tree-controls mb-3");
    
    controlsDiv.append("span")
        .attr("class", "me-2")
        .text("View Mode:");
    
    controlsDiv.append("div")
        .attr("class", "btn-group btn-group-sm")
        .html(`
            <button class="btn ${viewMode === 'fit' ? 'btn-primary' : 'btn-outline-primary'}" 
                    id="${containerId}-fit-btn">Fit to Screen</button>
            <button class="btn ${viewMode === 'auto' ? 'btn-primary' : 'btn-outline-primary'}" 
                    id="${containerId}-auto-btn">Auto Scale</button>
        `);
    
    // Add zoom controls
    controlsDiv.append("div")
        .attr("class", "btn-group btn-group-sm ms-3")
        .html(`
            <button class="btn btn-outline-secondary" id="${containerId}-zoom-in">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" 
                    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    <line x1="11" y1="8" x2="11" y2="14"></line>
                    <line x1="8" y1="11" x2="14" y2="11"></line>
                </svg>
            </button>
            <button class="btn btn-outline-secondary" id="${containerId}-zoom-out">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" 
                    stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    <line x1="8" y1="11" x2="14" y2="11"></line>
                </svg>
            </button>
            <button class="btn btn-outline-secondary" id="${containerId}-reset">Reset</button>
        `);
    
    // Set up dimensions and margins based on view mode
    let margin, width, height, nodeDistance, levelDistance;
    
    if (viewMode === "fit") {
        // Fit to screen mode - use container dimensions
        margin = { top: 20, right: 90, bottom: 30, left: 90 };
        width = 960 - margin.left - margin.right;
        height = 500 - margin.top - margin.bottom;
        nodeDistance = null; // Let d3.tree handle spacing
        levelDistance = null;
    } else {
        // Auto scale mode - give more space between nodes
        margin = { top: 20, right: 120, bottom: 30, left: 120 };
        
        // Count nodes to determine width and height
        let nodeCount = countNodes(treeData);
        let maxDepth = getTreeDepth(treeData);
        
        // Calculate dimensions based on tree size
        width = Math.max(960, nodeCount * 80) - margin.left - margin.right;
        height = Math.max(500, maxDepth * 100) - margin.top - margin.bottom;
        
        // Set explicit node spacing
        nodeDistance = 30; // Vertical space between siblings
        levelDistance = 180; // Horizontal space between parent and child
    }
    
    // Create SVG container with zoom capability
    const svgContainer = d3.select(`#${containerId}`)
        .append("div")
        .attr("class", "tree-container")
        .style("overflow", "auto")
        .style("width", "100%")
        .style("height", `${height + margin.top + margin.bottom}px`);
        
    const svg = svgContainer
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .attr("class", "tree-svg");
        
    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`)
        .attr("class", "tree-group");
    
    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 3])
        .on("zoom", (event) => {
            g.attr("transform", event.transform);
        });
    
    svg.call(zoom);
    
    // Create a tree layout
    const treeLayout = d3.tree().size([height, width]);
    
    // For auto scale mode, set fixed node size instead of overall dimensions
    if (viewMode === "auto" && nodeDistance && levelDistance) {
        treeLayout.nodeSize([nodeDistance, levelDistance]);
        treeLayout.separation((a, b) => {
            return a.parent === b.parent ? 1.5 : 2;
        });
    }
    
    // Convert the nested data to a hierarchical structure
    const root = d3.hierarchy(treeData);
    
    // Assign positions to nodes
    treeLayout(root);
    
    // Create links between nodes
    const links = g.selectAll(".parse-tree-link")
        .data(root.links())
        .enter()
        .append("path")
        .attr("class", "parse-tree-link")
        .attr("d", d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x));
    
    // Create nodes
    const nodes = g.selectAll(".parse-tree-node")
        .data(root.descendants())
        .enter()
        .append("g")
        .attr("class", "parse-tree-node")
        .attr("transform", d => `translate(${d.y},${d.x})`)
        .on("click", function(event, d) {
            // Toggle node children on click
            if (d.children) {
                d._children = d.children;
                d.children = null;
            } else {
                d.children = d._children;
                d._children = null;
            }
            renderParseTree(treeData, containerId, viewMode);  // Re-render the tree
        });
    
    // Add circles to nodes
    nodes.append("circle")
        .attr("r", 5)
        .attr("fill", d => d.data.type === 'terminal' ? "#f8f9fa" : "#339af0");
    
    // Add node labels
    nodes.append("text")
        .attr("dy", "0.32em")
        .attr("x", d => d.children ? -8 : 8)
        .attr("text-anchor", d => d.children ? "end" : "start")
        .text(d => d.data.name)
        .style("font-size", d => d.data.type === 'terminal' ? "10px" : "12px")
        .style("fill", d => d.data.type === 'terminal' ? "#adb5bd" : "#e9ecef");
    
    // Initial center and fit the tree
    if (viewMode === "fit") {
        resetZoom();
    }
    
    // Add event listeners for view mode toggle buttons
    document.getElementById(`${containerId}-fit-btn`).addEventListener("click", () => {
        renderParseTree(treeData, containerId, "fit");
    });
    
    document.getElementById(`${containerId}-auto-btn`).addEventListener("click", () => {
        renderParseTree(treeData, containerId, "auto");
    });
    
    // Add event listeners for zoom controls
    document.getElementById(`${containerId}-zoom-in`).addEventListener("click", () => {
        svg.transition().call(zoom.scaleBy, 1.3);
    });
    
    document.getElementById(`${containerId}-zoom-out`).addEventListener("click", () => {
        svg.transition().call(zoom.scaleBy, 0.7);
    });
    
    document.getElementById(`${containerId}-reset`).addEventListener("click", resetZoom);
    
    function resetZoom() {
        svg.transition().call(zoom.transform, d3.zoomIdentity
            .translate(margin.left, margin.top)
            .scale(1));
    }
    
    // Helper function to count all nodes in the tree
    function countNodes(node) {
        if (!node) return 0;
        let count = 1; // Count this node
        if (node.children) {
            node.children.forEach(child => {
                count += countNodes(child);
            });
        }
        return count;
    }
    
    // Helper function to get the maximum depth of the tree
    function getTreeDepth(node, currentDepth = 0) {
        if (!node) return currentDepth;
        if (!node.children) return currentDepth + 1;
        
        let maxChildDepth = currentDepth;
        node.children.forEach(child => {
            const childDepth = getTreeDepth(child, currentDepth + 1);
            maxChildDepth = Math.max(maxChildDepth, childDepth);
        });
        
        return maxChildDepth;
    }
}

// Enhanced AST tree visualization with the same view modes
function renderASTTree(treeData, containerId = "ast-diagram", viewMode = "fit") {
    if (!treeData) return;
    
    // For the AST tree, we'll use the same functionality but with different styling
    // Create a deep clone of the tree data to avoid modifying the original
    const astData = JSON.parse(JSON.stringify(treeData));
    const simplifiedTree = simplifyForAST(astData);
    
    // Render using the enhanced tree function
    renderParseTree(simplifiedTree, containerId, viewMode);
    
    // Override styles for AST tree
    d3.select(`#${containerId}`)
        .selectAll(".parse-tree-node circle")
        .attr("fill", d => d.data.type === 'terminal' ? "#ffd43b" : "#20c997");
    
    // Function to simplify the tree to represent an AST better
    function simplifyForAST(node) {
        if (!node) return null;
        
        if (node.type === 'terminal' && node.name.match(/^[,;()[\]{}]$/)) {
            return null;  // Skip pure punctuation nodes
        }
        
        if (node.children) {
            const newChildren = [];
            for (const child of node.children) {
                const simplified = simplifyForAST(child);
                if (simplified) {
                    newChildren.push(simplified);
                }
            }
            node.children = newChildren.length > 0 ? newChildren : null;
        }
        
        return node;
    }
}


