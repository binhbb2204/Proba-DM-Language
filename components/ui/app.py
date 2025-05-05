import os
import sys
import tkinter as tk
import traceback
from tkinter import ttk, scrolledtext, filedialog, messagebox
from antlr4 import *
from antlr4.tree.Tree import TerminalNode
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from components.utils.constants import DIR, CPL_DEST, SRC
from components.utils.antlr_utils import ensure_parser_exists
from components.utils.code_utils import apply_syntax_highlighting
from components.error_handling import PQLSyntaxErrorListener
from components.interpreter import PQLInterpreterVisitor
from components.ui.output import RedirectOutput
from components.ui.visualization import Visualizer

class ProbabilisticQueryLanguageApp:
    """Main application class for PQL IDE"""
    def __init__(self, root):
        self.root = root
        self.root.title("Probabilistic Query Language for Data Mining")
        self.root.geometry("1200x800")
        
        self.create_menu()
        self.create_ui()
        self.load_sample_code()
        self.setup_syntax_highlighting()
    
    def create_menu(self):
        """Create menu bar"""
        menu_bar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New", command=self.clear_code)
        file_menu.add_command(label="Open", command=self.load_code)
        file_menu.add_command(label="Save", command=self.save_code)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        examples_menu = tk.Menu(menu_bar, tearoff=0)
        examples_menu.add_command(label="Customer Segmentation", command=lambda: self.load_example("customer"))
        examples_menu.add_command(label="Risk Assessment", command=lambda: self.load_example("risk"))
        examples_menu.add_command(label="Predictive Maintenance", command=lambda: self.load_example("maintenance"))
        menu_bar.add_cascade(label="Examples", menu=examples_menu)
        
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)
    
    def create_ui(self):
        """Create main UI components"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Editor frame
        self.editor_frame = ttk.LabelFrame(paned_window, text="PQL Editor")
        paned_window.add(self.editor_frame, weight=1)
        
        self.line_numbers = tk.Text(self.editor_frame, width=4, padx=3, takefocus=0,
                                   border=0, background="#f0f0f0", state="disabled")
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), pady=5)
        
        self.code_editor = scrolledtext.ScrolledText(self.editor_frame, wrap=tk.WORD, undo=True,
                                                   font=("Consolas", 11))
        self.code_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        button_frame = ttk.Frame(self.editor_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Run", command=self.run_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load", command=self.load_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_code).pack(side=tk.LEFT, padx=5)
        
        # Output frame
        self.output_frame = ttk.LabelFrame(paned_window, text="Results")
        paned_window.add(self.output_frame, weight=1)
        
        self.notebook = ttk.Notebook(self.output_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.output_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.output_tab, text="Output")
        
        self.output_text = scrolledtext.ScrolledText(self.output_tab, wrap=tk.WORD, bg="#f0f0f0")
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.config(state=tk.DISABLED)
        
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="Visualization")
        
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ast_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ast_tab, text="AST")
        
        self.ast_text = scrolledtext.ScrolledText(self.ast_tab, wrap=tk.WORD)
        self.ast_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_syntax_highlighting(self):
        """Configure syntax highlighting"""
        self.code_editor.tag_config("keyword", foreground="blue")
        self.code_editor.tag_config("distribution", foreground="purple")
        self.code_editor.tag_config("operator", foreground="red")
        self.code_editor.tag_config("comment", foreground="gray")
        self.code_editor.tag_config("string", foreground="green")
        
        self.code_editor.bind("<KeyRelease>", self.highlight_syntax)
    
    def highlight_syntax(self, event=None):
        """Apply syntax highlighting"""
        for tag in ["keyword", "distribution", "operator", "comment", "string"]:
            self.code_editor.tag_remove(tag, "1.0", tk.END)
        
        keywords = r'\b(var|follows|query|load_data|cluster|find_associations|classify|data|fitted_to)\b'
        distributions = r'\b(Normal|LogNormal|Poisson|Bernoulli|EmpiricalDistribution|Gamma|Beta|Multinomial)\b'
        operators = r'(>|<|>=|<=|==|!=|\band\b|\bor\b|\bnot\b|\?|:|\+|-|\*|/|\^)'
        comments = r'//.*$'
        strings = r'"[^"]*"'
        
        content = self.code_editor.get("1.0", tk.END)
        apply_syntax_highlighting(self.code_editor, content, keywords, "keyword")
        apply_syntax_highlighting(self.code_editor, content, distributions, "distribution")
        apply_syntax_highlighting(self.code_editor, content, operators, "operator")
        apply_syntax_highlighting(self.code_editor, content, comments, "comment", multiline=False)
        apply_syntax_highlighting(self.code_editor, content, strings, "string")
    
    def load_sample_code(self):
        """Load sample PQL code"""
        sample_code = """// Customer Segmentation Example
load_data("customer_database.csv");

// Define probabilistic variables
var age follows EmpiricalDistribution(data.age);
var income follows LogNormal(10, 0.5);
var purchaseFrequency follows Poisson(3.2);

// Define customer segments
var highValueCustomer = (income > 75000) and (purchaseFrequency > 5);
var churnRisk = (daysSinceLastPurchase > 60) and (purchaseFrequency < 1);

// Mine insights with probabilistic queries
query P(highValueCustomer | age > 40);
query E(lifetimeValue | churnRisk);
cluster(customers, dimensions: [age, income, purchaseFrequency], k: 3);
"""
        self.code_editor.delete("1.0", tk.END)
        self.code_editor.insert(tk.END, sample_code)
        self.highlight_syntax()
    
    def load_example(self, example_type):
        """Load example PQL code"""
        self.clear_code()
        
        if example_type == "customer":
            self.load_sample_code()
        elif example_type == "risk":
            risk_code = """// Risk Assessment Example
load_data("transactions.csv");

// Define probabilistic variables
var transactionAmount follows EmpiricalDistribution(data.amount);
var userBehavior follows fitted_to: data.frequency;
var locationDistance follows Gamma(2.0, 0.5);

// Define risk factors
var unusualAmount = (transactionAmount > 1000);
var unusualLocation = (locationDistance > 100);
var fraudRisk = unusualAmount and unusualLocation;

// Risk queries
query P(fraudRisk | transactionAmount > 1000);
query E(financialImpact | fraudRisk);
query correlation(transactionAmount, locationDistance);
"""
            self.code_editor.insert(tk.END, risk_code)
        elif example_type == "maintenance":
            maintenance_code = """// Predictive Maintenance Example
load_data("equipment_sensors.csv");

// Define probabilistic variables
var temperatureReadings follows Normal(70, 5);
var vibrationLevel follows Gamma(2.0, 1.5);
var operatingHours follows EmpiricalDistribution(data.hours);

// Define maintenance conditions
var abnormalTemperature = (temperatureReadings > 85);
var highVibration = (vibrationLevel > 3.5);
var failureRisk = abnormalTemperature and highVibration;

// Maintenance queries
query P(failureRisk | operatingHours > 5000);
query E(timeToFailure | abnormalTemperature);
find_associations(sensorReadings, min_support: 0.3, min_confidence: 0.8);
"""
            self.code_editor.insert(tk.END, maintenance_code)
            
        self.highlight_syntax()
    
    def run_code(self):
        """Run PQL code from editor"""
        code = self.code_editor.get("1.0", tk.END)
        
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        
        redirect = RedirectOutput(self.output_text)
        sys.stdout = redirect
        
        try:
            grammar_file = os.path.join(DIR, SRC)
            if not os.path.exists(grammar_file):
                print(f"Error: Grammar file {SRC} not found")
                return
            
            if not ensure_parser_exists():
                return
            
            print("Running probabilistic query language interpreter...")
            print("-" * 50)
            
            results = self.parse_and_process_code(code)
            
            if results:
                self.generate_ast_visualization(code)
                Visualizer.create_visualization(self.fig, results)
                self.canvas.draw()
            else:
                print("No results to visualize due to errors")
                
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            print(traceback.format_exc())
        finally:
            sys.stdout = sys.__stdout__
    
    def parse_and_process_code(self, code):
        """Parse and process PQL code using ANTLR"""
        try:
            from CompiledFiles.ProbDataMineLexer import ProbDataMineLexer
            from CompiledFiles.ProbDataMineParser import ProbDataMineParser
        except ImportError:
            print("Error: ANTLR-generated files not found. Please ensure parser is generated.")
            return {}
        
        try:
            input_stream = InputStream(code)
            lexer = ProbDataMineLexer(input_stream)
            stream = CommonTokenStream(lexer)
            parser = ProbDataMineParser(stream)
            
            error_listener = PQLSyntaxErrorListener()
            lexer.removeErrorListeners()
            parser.removeErrorListeners()
            lexer.addErrorListener(error_listener)
            parser.addErrorListener(error_listener)
            
            tree = parser.program()
            
            if error_listener.errors:
                for error in error_listener.errors:
                    print(error)
                print(f"Found {len(error_listener.errors)} syntax errors. Aborting execution.")
                return {}
            
            print("Parse tree structure:")
            print(self.generate_tree_representation(tree, parser.ruleNames))
            print("-" * 50)
            
            visitor = PQLInterpreterVisitor(lambda msg: print(msg))
            results = visitor.visit(tree)
            if results is None:
                print("Error: Visitor returned None instead of results dictionary")
                return {}
            
            return results
        except Exception as e:
            print(f"Error during parsing/processing: {str(e)}")
            print(traceback.format_exc())
            return {}
    
    def generate_ast_visualization(self, code):
        """Generate AST visualization"""
        try:
            from CompiledFiles.ProbDataMineLexer import ProbDataMineLexer
            from CompiledFiles.ProbDataMineParser import ProbDataMineParser
        except ImportError:
            print("Error: ANTLR-generated files not found.")
            return
        
        input_stream = InputStream(code)
        lexer = ProbDataMineLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = ProbDataMineParser(stream)
        
        tree = parser.program()
        
        self.ast_text.delete("1.0", tk.END)
        ast_text = self.generate_tree_representation(tree, parser.ruleNames)
        self.ast_text.insert(tk.END, ast_text)
    
    def generate_tree_representation(self, tree, rule_names, level=0):
        """Generate text representation of parse tree"""
        if tree is None:
            return ""
        
        indent = "  " * level
        prefix = "├─ " if level > 0 else ""
        
        if isinstance(tree, TerminalNode):
            return f"{indent}{prefix}Terminal({tree.getText()})\n"
        
        rule_name = rule_names[tree.getRuleIndex()] if tree.getRuleIndex() >= 0 else "Unknown"
        result = f"{indent}{prefix}{rule_name}\n"
        
        for i in range(tree.getChildCount()):
            child = tree.getChild(i)
            result += self.generate_tree_representation(child, rule_names, level + 1)
        
        return result
    
    def save_code(self):
        """Save code to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pql",
            filetypes=[("PQL files", "*.pql"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, "w") as f:
                f.write(self.code_editor.get("1.0", tk.END))
            messagebox.showinfo("Save", f"File saved successfully: {filename}")
    
    def load_code(self):
        """Load code from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("PQL files", "*.pql"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, "r") as f:
                code = f.read()
            self.clear_code()
            self.code_editor.insert(tk.END, code)
            self.highlight_syntax()
    
    def clear_code(self):
        """Clear code editor"""
        self.code_editor.delete("1.0", tk.END)
    
    def show_documentation(self):
        """Show PQL documentation"""
        doc_window = tk.Toplevel(self.root)
        doc_window.title("PQL Documentation")
        doc_window.geometry("800x600")
        
        doc_notebook = ttk.Notebook(doc_window)
        doc_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        overview_tab = ttk.Frame(doc_notebook)
        doc_notebook.add(overview_tab, text="Overview")
        
        overview_text = scrolledtext.ScrolledText(overview_tab, wrap=tk.WORD)
        overview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        overview_text.insert(tk.END, """# Probabilistic Query Language (PQL) for Data Mining

PQL is a domain-specific language designed for probabilistic data mining and analysis. It combines 
concepts from probability theory, statistics, and data mining to provide a powerful yet accessible 
way to extract insights from data.

## Key Features

- **Probabilistic Variables**: Define variables that follow statistical distributions
- **Probabilistic Queries**: Ask questions about probabilities and expected values
- **Data Mining Operations**: Perform clustering, association rule mining, and classification
- **Visualization**: Generate visualizations of query results

## Basic Syntax
// Load data
load_data("filename.csv");

// Define variables
var x follows Normal(0, 1);

// Make queries
query P(x > 0);
query E(x | x > 0);
""")
        overview_text.config(state=tk.DISABLED)
        
        syntax_tab = ttk.Frame(doc_notebook)
        doc_notebook.add(syntax_tab, text="Syntax")
        
        syntax_text = scrolledtext.ScrolledText(syntax_tab, wrap=tk.WORD)
        syntax_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        syntax_text.insert(tk.END, """# PQL Syntax Reference

## Data Loading
load_data("filename.csv");

## Variable Declarations

Supported distributions:
- Normal(mean, stddev)
- LogNormal(mu, sigma)
- Poisson(lambda)
- Bernoulli(p)
- EmpiricalDistribution(data_reference)
- Gamma(shape, scale)
- Beta(alpha, beta)
- Multinomial(n, p1, p2, ...)
- fitted_to: data_reference

## Queries
Probability query:
query P(condition);

Expected value query:
query E(expression);
query E(expression | condition);

Correlation query:
query correlation(expr1, expr2);

Outliers query:
query outliers(expr1, expr2, ...);

## Data Mining Operations
Clustering:
query cluster(dataset, dimensions: [dim1, dim2, ...], k: num_clusters);


Association Rules:
query find_associations(dataset, min_support: threshold, min_confidence: threshold);

Classification:
query classify(dataset, target: target_variable, features: [feature1, feature2, ...]);
""")
        syntax_text.config(state=tk.DISABLED)
        
        examples_tab = ttk.Frame(doc_notebook)
        doc_notebook.add(examples_tab, text="Examples")
        
        examples_text = scrolledtext.ScrolledText(examples_tab, wrap=tk.WORD)
        examples_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        examples_text.insert(tk.END, """# PQL Examples

## Customer Segmentation
// Load customer data
load_data("customers.csv");

// Define probabilistic variables
var age follows EmpiricalDistribution(data.age);
var income follows LogNormal(10, 0.5);
var purchaseFrequency follows Poisson(3.2);

// Define customer segments
var highValueCustomer = (income > 75000) and (purchaseFrequency > 5);
var churnRisk = (daysSinceLastPurchase > 60) and (purchaseFrequency < 1);

// Mine insights with probabilistic queries
query P(highValueCustomer | age > 40);
query E(lifetimeValue | churnRisk);
query cluster(customers, dimensions: [age, income, purchaseFrequency], k: 3);

## Fraud Detection
// Load transaction data
load_data("transactions.csv");

// Define probabilistic variables
var amount follows EmpiricalDistribution(data.transactionAmount);
var frequency follows Poisson(5.2);
var distance follows Gamma(2.0, 1.0);

// Define fraud indicators
var unusualAmount = amount > 3 * E(amount);
var unusualLocation = distance > 100;

// Query fraud probability
query P(unusualAmount and unusualLocation);
query E(financialImpact | unusualAmount and unusualLocation);
query correlation(amount, distance);
""")
        examples_text.config(state=tk.DISABLED)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Probabilistic Query Language for Data Mining

Version 1.0

A domain-specific language for probabilistic data mining and analysis.
Combines concepts from probability theory, statistics, and data mining
to provide a powerful yet accessible way to extract insights from data.

© 2025 PQL Development Team
"""
        messagebox.showinfo("About PQL", about_text)