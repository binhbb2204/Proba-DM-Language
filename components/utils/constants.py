import os

# Define constants
DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Update the ANTLR JAR path according to your system
# Ensure the path is correct for your environment
ANTLR_JAR = 'C:/JavaLibrary/antlr-4.13.2-complete.jar'
CPL_DEST = 'CompiledFiles'
SRC = 'ProbDataMine.g4'
TESTS = os.path.join(DIR, './tests')