import os
import subprocess
from .constants import ANTLR_JAR, CPL_DEST, DIR, SRC

def print_usage():
    """Print usage instructions"""
    print('python run.py gen')
    print('python run.py test')

def print_break():
    """Print separator"""
    print('-----------------------------------------------')
    
def generate_antlr2python():
    """Generate Python parser from grammar"""
    print('Antlr4 is running...')
    try:
        subprocess.run(['java', '-jar', ANTLR_JAR, '-o', CPL_DEST, '-no-listener', '-Dlanguage=Python3', SRC], check=True)
        print('Generate successfully')
        return True
    except subprocess.CalledProcessError as e:
        print(f'Failed to generate parser: {str(e)}')
        return False

def ensure_parser_exists():
    """Ensure the parser files exist, generate them if needed"""
    print("Checking for existing parser...")
    if not os.path.exists(os.path.join(DIR, CPL_DEST, "ProbDataMineLexer.py")):
        print("Generating parser from grammar...")
        return generate_antlr2python()
    else:
        print("Using existing parser")
        return True