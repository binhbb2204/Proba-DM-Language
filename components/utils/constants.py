import os

# Define constants
DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ANTLR_JAR = 'C:/JavaLib/antlr-4.13.2-complete.jar'
CPL_DEST = 'CompiledFiles'
SRC = 'ProbDataMine.g4'
TESTS = os.path.join(DIR, './tests')