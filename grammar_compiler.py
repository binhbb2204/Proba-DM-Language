import os
import subprocess
import sys

def generate_parser():
    """Generate ANTLR4 parser files from grammar definition"""
    # Create CompiledFiles directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CompiledFiles')
    os.makedirs(output_dir, exist_ok=True)

    # Path to grammar file
    grammar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ProbDataMine.g4')

    # Path to ANTLR4 jar
    antlr_jar = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JavaLib', 'antlr-4.13.2-complete.jar')

    # Run ANTLR to generate parser
    command = [
        'java', 
        '-jar', 
        antlr_jar, 
        '-o', 
        output_dir, 
        '-Dlanguage=Python3',
        '-visitor',
        grammar_file
    ]

    print("Generating ANTLR parser files...")
    try:
        subprocess.run(command, check=True)
        print("Parser files generated successfully in CompiledFiles folder.")
        
        # Create __init__.py in CompiledFiles directory for proper imports
        with open(os.path.join(output_dir, '__init__.py'), 'w') as f:
            pass  # Create an empty file
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating parser: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    generate_parser()