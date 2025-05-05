// Render a parse tree using D3.js
function renderParseTree(treeData) {
    if (!treeData) return;
    
    // Set up dimensions and margins
    const margin = { top: 20, right: 90, bottom: 30, left: 90 };
    const width = 960 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;
    
    // Append SVG to the container
    const svg = d3.select("#tree-diagram")
        .append("svg")
        .attr("width", "100%")
        .attr("height", height + margin.top + margin.bottom)
        .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);
    
    // Create a tree layout
    const treeLayout = d3.tree().size([height, width]);
    
    // Convert the nested data to a hierarchical structure
    const root = d3.hierarchy(treeData);
    
    // Assign positions to nodes
    treeLayout(root);
    
    // Create links between nodes
    const links = svg.selectAll(".parse-tree-link")
        .data(root.links())
        .enter()
        .append("path")
        .attr("class", "parse-tree-link")
        .attr("d", d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x));
    
    // Create nodes
    const nodes = svg.selectAll(".parse-tree-node")
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
            renderParseTree(treeData);  // Re-render the tree
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
}

// Render an AST tree using D3.js (simpler version for now)
function renderASTTree(treeData) {
    if (!treeData) return;
    
    // For simplicity, we'll use the same tree visualization as the parse tree
    // In a real implementation, we'd generate a proper AST and have a different visualization
    
    // Set up dimensions and margins (slightly different to distinguish from parse tree)
    const margin = { top: 20, right: 120, bottom: 30, left: 60 };
    const width = 960 - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;
    
    // Append SVG to the container
    const svg = d3.select("#ast-diagram")
        .append("svg")
        .attr("width", "100%")
        .attr("height", height + margin.top + margin.bottom)
        .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);
    
    // Create a tree layout (more compact)
    const treeLayout = d3.tree().size([height, width * 0.9]);
    
    // Simplify the tree to represent an AST better
    // (in a real implementation, this would be a different data structure)
    function simplifyForAST(node) {
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
    
    // Create a deep clone of the tree data to avoid modifying the original
    const astData = JSON.parse(JSON.stringify(treeData));
    const simplifiedTree = simplifyForAST(astData);
    
    // Convert the nested data to a hierarchical structure
    const root = d3.hierarchy(simplifiedTree);
    
    // Assign positions to nodes
    treeLayout(root);
    
    // Create links between nodes
    const links = svg.selectAll(".parse-tree-link")
        .data(root.links())
        .enter()
        .append("path")
        .attr("class", "parse-tree-link")
        .attr("d", d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x));
    
    // Create nodes
    const nodes = svg.selectAll(".parse-tree-node")
        .data(root.descendants())
        .enter()
        .append("g")
        .attr("class", "parse-tree-node")
        .attr("transform", d => `translate(${d.y},${d.x})`);
    
    // Add circles to nodes
    nodes.append("circle")
        .attr("r", 5)
        .attr("fill", d => d.data.type === 'terminal' ? "#ffd43b" : "#20c997");
    
    // Add node labels
    nodes.append("text")
        .attr("dy", "0.32em")
        .attr("x", d => d.children ? -8 : 8)
        .attr("text-anchor", d => d.children ? "end" : "start")
        .text(d => d.data.name)
        .style("font-size", d => d.data.type === 'terminal' ? "10px" : "12px")
        .style("fill", d => d.data.type === 'terminal' ? "#adb5bd" : "#e9ecef");
}
