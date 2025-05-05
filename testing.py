import os
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from antlr4 import *
from antlr4.error.ErrorListener import ErrorListener
from CompiledFiles.ProbDataMineLexer import ProbDataMineLexer
from CompiledFiles.ProbDataMineParser import ProbDataMineParser

# Define constants
DIR = os.path.dirname(__file__)
ANTLR_JAR = 'C:/JavaLib/antlr-4.13.2-complete.jar'
CPL_Dest = 'CompiledFiles'
SRC = 'Sample.g4'

# Ensure output directory exists
os.makedirs(CPL_Dest, exist_ok=True)

# Function to generate ANTLR Python files
def generateAntlr2Python():
    print('Generating ANTLR parser...')
    subprocess.run([
        'java', '-jar', ANTLR_JAR,
        '-o', CPL_Dest,
        '-Dlanguage=Python3',
        '-no-listener',
        SRC
    ])
    print('Generated successfully!')

# Function to parse PQL input
def parse_pql(input_text):
    input_stream = InputStream(input_text)
    lexer = ProbDataMineLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = ProbDataMineParser(stream)
    parser.removeErrorListeners()
    error_collector = SyntaxErrorCollector()
    parser.addErrorListener(error_collector)

    tree = parser.program()

    if error_collector.errors:
        return "\n".join(error_collector.errors)
    return "PQL parsed successfully. Ready to evaluate."

# Custom error listener to collect syntax errors
class SyntaxErrorCollector(ErrorListener):
    def __init__(self):
        super(SyntaxErrorCollector, self).__init__()
        self.errors = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")

# GUI setup
def create_gui():
    root = tk.Tk()
    root.title("PQL Interpreter")
    root.geometry("1000x600")

    # Frames
    left_frame = ttk.Frame(root, width=500)
    right_frame = ttk.Frame(root, width=500)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Text Editor
    code_input = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, font=("Consolas", 12))
    code_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Output Box
    output_display = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=("Consolas", 12), state=tk.DISABLED)
    output_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Button
    def run_code():
        pql_code = code_input.get("1.0", tk.END).strip()
        result = parse_pql(pql_code)
        output_display.config(state=tk.NORMAL)
        output_display.delete("1.0", tk.END)
        output_display.insert(tk.END, result)
        output_display.config(state=tk.DISABLED)

    run_button = ttk.Button(left_frame, text="Run PQL", command=run_code)
    run_button.pack(pady=5)

    root.mainloop()

# Entry point
if __name__ == '__main__':
    generateAntlr2Python()
    create_gui()
