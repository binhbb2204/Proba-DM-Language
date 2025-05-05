import os
import json
import traceback
import sys
from flask import Flask, render_template, request, jsonify

# Ensure CompiledFiles is in the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CompiledFiles'))

# Import parser and executor
from pql_parser import PQLParser
from pql_executor import PQLExecutor

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "pql_development_key")

# Initialize parser and executor
# Since parser generation is now handled by grammar_compiler.py, we can reference
# the grammar file directly, as it will be compiled to the CompiledFiles directory
parser = PQLParser(grammar_path='ProbDataMine.g4')
executor = PQLExecutor()

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/parse', methods=['POST'])
def parse_pql():
    """Parse PQL code and return results"""
    try:
        code = request.json.get('code', '')
        if not code:
            return jsonify({'success': False, 'error': 'No code provided'})
            
        # Initialize parser if needed
        if not parser.initialized:
            success = parser.initialize()
            if not success:
                return jsonify({'success': False, 'error': 'Failed to initialize parser'})
        
        # Parse the code
        result = parser.parse(code)
        
        # Make sure we return a serializable result
        serializable_result = {
            'success': result.get('success', False)
        }
        
        if not serializable_result['success']:
            serializable_result['errors'] = result.get('errors', [])
        else:
            # Don't include the actual tree object, just indicate it's valid
            serializable_result['message'] = 'Syntax is valid'
            
        return jsonify(serializable_result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/visualize_parse_tree', methods=['POST'])
def visualize_parse_tree():
    """Generate a parse tree visualization"""
    try:
        code = request.json.get('code', '')
        if not code:
            return jsonify({'success': False, 'error': 'No code provided'})
            
        # Get tree visualization
        result = parser.visualize_tree(code)
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/execute', methods=['POST'])
def execute_pql():
    """Execute PQL code and return results"""
    try:
        code = request.json.get('code', '')
        if not code:
            return jsonify({'success': False, 'error': 'No code provided'})
            
        # First parse the code to check syntax
        parse_result = parser.parse(code)
        
        # Then execute if parsing was successful
        execution_result = executor.execute(code, parse_result)
        return jsonify(execution_result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/samples', methods=['GET'])
def get_samples():
    """Return sample PQL code snippets"""
    samples = [
        {
            'name': 'Basic Data Load',
            'code': 'load_data("sample.csv", name: "customer_data");\n\n// View basic statistics\nquery correlation(data.age, data.income);'
        },
        {
            'name': 'Probabilistic Variables',
            'code': '// Define probabilistic variables\nvar income follows Normal(50000, 15000);\nvar age follows Normal(35, 10);\nvar is_customer follows Bernoulli(0.7);\n\n// Query probabilities\nquery P(income > 60000);\nquery correlation(income, age);'
        },
        {
            'name': 'Clustering Example',
            'code': 'load_data("sample.csv", name: "customer_data");\n\n// Perform clustering\ncluster(customer_data, dimensions: [age, income, spending], k: 3);'
        },
        {
            'name': 'Association Rules',
            'code': 'load_data("sample.csv", name: "transaction_data");\n\n// Find association rules\nfind_associations(transaction_data, min_support: 0.1, min_confidence: 0.5);'
        },
        {
            'name': 'Classification Model',
            'code': 'load_data("sample.csv", name: "customer_data");\n\n// Build a classification model\nclassify(customer_data, target: is_customer, classifier: "random_forest");'
        }
    ]
    
    return jsonify({'samples': samples})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)