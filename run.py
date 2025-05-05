import sys
import os
import tkinter as tk

# Ensure the components directory exists and is in the path
if not os.path.exists('components'):
    os.makedirs('components', exist_ok=True)
    os.makedirs('components/ui', exist_ok=True)
    os.makedirs('components/utils', exist_ok=True)

# Import project components
from components.utils.antlr_utils import print_usage, print_break, generate_antlr2python
from components.ui.app import ProbabilisticQueryLanguageApp

def main(argv):
    """Main entry point"""
    from components.utils.constants import ANTLR_JAR
    print('ANTLR JAR file: ' + str(ANTLR_JAR))
    print('Arguments: ' + str(len(argv)))
    print_break()
    
    if len(argv) <= 1:
        print_usage()
    elif argv[1] == 'gen':
        generate_antlr2python()
    elif argv[1] == 'test':
        root = tk.Tk()
        app = ProbabilisticQueryLanguageApp(root)
        root.mainloop()
    else:
        print_usage()

if __name__ == "__main__":
    main(sys.argv)