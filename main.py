import sys
import os
from grammar_compiler import generate_parser

# Generate the parser files before importing the app
if not generate_parser():
    print("Failed to generate parser files. Exiting.")
    sys.exit(1)

# Add CompiledFiles to Python path to ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CompiledFiles'))

# Now import the app
from app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)