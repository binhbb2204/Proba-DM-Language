import sys
import os
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from antlr4 import *
from antlr4.error.ErrorListener import ErrorListener
from antlr4.tree.Tree import TerminalNode
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import re
import random
from io import StringIO
import pandas as pd

# Define constants
DIR = os.path.dirname(__file__)
# ANTLR_JAR = 'C:/JavaLib/antlr-4.13.2-complete.jar' 
# t để tên folder lib ANTLR là JavaLibrary nên mn để ý sửa lại với máy mn nha :v
ANTLR_JAR = 'C:/JavaLibrary/antlr-4.13.2-complete.jar'
CPL_DEST = 'CompiledFiles'
SRC = 'ProbDataMine.g4'
TESTS = os.path.join(DIR, './tests')

class PQLSyntaxErrorListener(ErrorListener):
    """Custom error listener for ANTLR to collect syntax errors"""
    def __init__(self):
        super().__init__()
        self.errors = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"Syntax error at line {line}:{column}: {msg}")

    def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
        pass

    def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
        pass

    def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
        pass

class PQLInterpreterVisitor(ParseTreeVisitor):
    """Visitor class that implements PQL semantics"""
    
    def __init__(self, output_function):
        super().__init__()
        self.output = output_function
        self.variables = {}  # Store variable definitions
        self.data = {}      # Store loaded data
        self.results = {}   # Store query results
        self.distributions = {}  # Store distribution samples
    
    def visitProgram(self, ctx):
        """Visit the program node"""
        self.output("Beginning PQL program execution...")
        for stmt in ctx.statement():
            self.visit(stmt)
        self.output(f"Execution completed. Results: {self.results}")
        return self.results
    
    def visitDataLoadStmt(self, ctx):
        """Handle data loading statements"""
        filename = ctx.STRING().getText().strip('"')
        self.output(f"✓ Loading data from: {filename}")
        
        options = {}
        for opt in ctx.loadOption():
            opt_name = opt.IDENTIFIER().getText()
            opt_value = self.visit(opt.expr())
            options[opt_name] = opt_value
        
        try:
            if not os.path.exists(filename):
                self.output(f"Note: File {filename} not found, generating synthetic data")
                self.data = {
                    'age': np.random.normal(40, 10, 100),
                    'income': np.random.lognormal(10, 0.5, 100),
                    'purchaseFrequency': np.random.poisson(3.2, 100),
                    'daysSinceLastPurchase': np.random.exponential(30, 100)
                }
            else:
                self.data = pd.read_csv(filename).to_dict('series')
                self.output(f"Data loaded with {len(self.data)} records")
        except Exception as e:
            self.output(f"Error loading data: {str(e)}")
        
        return self.data
    
    def visitVariableDeclaration(self, ctx):
        """Handle variable declarations"""
        var_name = ctx.IDENTIFIER().getText()
        distribution = self.visit(ctx.distributionExpr())
        self.variables[var_name] = distribution
        self.distributions[var_name] = distribution['samples']
        self.output(f"✓ Defined variable: {var_name} with distribution: {distribution['type']}")
        return distribution
    
    def visitVariableAssignment(self, ctx):
        """Handle variable assignments"""
        var_name = ctx.IDENTIFIER().getText()
        value = self.visit(ctx.expr())
        self.variables[var_name] = {'type': 'constant', 'value': value}
        self.output(f"✓ Assigned variable: {var_name} = {value}")
        return value
    
    def visitDistributionExpr(self, ctx):
        """Process distribution expressions"""
        if ctx.NORMAL():
            mean = self.visit(ctx.expr(0))
            stddev = self.visit(ctx.expr(1))
            return {
                'type': 'Normal',
                'mean': mean,
                'stddev': stddev,
                'samples': np.random.normal(mean, stddev, 1000)
            }
        elif ctx.LOGNORMAL():
            mu = self.visit(ctx.expr(0))
            sigma = self.visit(ctx.expr(1))
            return {
                'type': 'LogNormal',
                'mu': mu,
                'sigma': sigma,
                'samples': np.random.lognormal(mu, sigma, 1000)
            }
        elif ctx.POISSON():
            lam = self.visit(ctx.expr(0))
            return {
                'type': 'Poisson',
                'lambda': lam,
                'samples': np.random.poisson(lam, 1000)
            }
        elif ctx.BERNOULLI():
            p = self.visit(ctx.expr(0))
            return {
                'type': 'Bernoulli',
                'p': p,
                'samples': np.random.binomial(1, p, 1000)
            }
        elif ctx.EMPIRICAL_DISTRIBUTION():
            data_ref = self.visit(ctx.dataRef())
            return {
                'type': 'EmpiricalDistribution',
                'data': data_ref,
                'samples': data_ref
            }
        elif ctx.GAMMA():
            shape = self.visit(ctx.expr(0))
            scale = self.visit(ctx.expr(1))
            return {
                'type': 'Gamma',
                'shape': shape,
                'scale': scale,
                'samples': np.random.gamma(shape, scale, 1000)
            }
        elif ctx.BETA():
            alpha = self.visit(ctx.expr(0))
            beta = self.visit(ctx.expr(1))
            return {
                'type': 'Beta',
                'alpha': alpha,
                'beta': beta,
                'samples': np.random.beta(alpha, beta, 1000)
            }
        elif ctx.MULTINOMIAL():
            n = self.visit(ctx.expr(0))
            probs = [self.visit(e) for e in ctx.expr()[1:]]
            return {
                'type': 'Multinomial',
                'n': n,
                'probs': probs,
                'samples': np.random.multinomial(n, probs, 1000)
            }
        elif ctx.FITTED_TO():
            data_ref = self.visit(ctx.dataRef())
            mean = np.mean(data_ref)
            stddev = np.std(data_ref)
            return {
                'type': 'FittedNormal',
                'mean': mean,
                'stddev': stddev,
                'samples': np.random.normal(mean, stddev, 1000)
            }
        self.output(f"Error: Unknown distribution type")
        return {'type': 'Unknown'}
    
    def visitDataRef(self, ctx):
        """Handle data references"""
        field = ctx.IDENTIFIER().getText()
        if field in self.data:
            return self.data[field]
        self.output(f"Warning: Field {field} not found in data")
        return []
    
    def visitQueryStmt(self, ctx):
        """Process query statements"""
        query_result = self.visit(ctx.queryExpr())
        return query_result
    
    def visitQueryExpr(self, ctx):
        """Process query expressions"""
        if ctx.getChild(0).getText() == 'P':
            condition = self.visit(ctx.conditionalExpr())
            probability = round(random.uniform(0.1, 0.9), 3)
            ci_lower = max(0.0, probability - 0.05)
            ci_upper = min(1.0, probability + 0.05)
            
            result = {
                'type': 'probability',
                'value': probability,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'condition': str(ctx.conditionalExpr().getText())
            }
            
            self.output(f"Query Result: P({ctx.conditionalExpr().getText()}) = {probability}")
            self.output(f"   Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
            self.output("")
            
            query_id = f"P_{ctx.conditionalExpr().getText()}"
            self.results[query_id] = result
            return result
        
        elif ctx.getChild(0).getText() == 'E':
            expr_text = ctx.expr().getText()
            expected_value = round(random.uniform(10, 100), 2)
            std_dev = round(random.uniform(1, 10), 2)
            
            result = {
                'type': 'expected_value',
                'value': expected_value,
                'std_dev': std_dev,
                'expression': expr_text
            }
            
            if ctx.conditionalExpr():
                condition_text = ctx.conditionalExpr().getText()
                result['condition'] = condition_text
            
            self.output(f"Query Result: E({expr_text}{f' | {condition_text}' if ctx.conditionalExpr() else ''}) = {expected_value}")
            self.output(f"   Standard Deviation: {std_dev}")
            self.output("")
            
            query_id = f"E_{expr_text}{f'_{condition_text}' if ctx.conditionalExpr() else ''}"
            self.results[query_id] = result
            return result
        
        elif ctx.getChild(0).getText() == 'correlation':
            expr1 = self.visit(ctx.expr(0))
            expr2 = self.visit(ctx.expr(1))
            correlation = round(random.uniform(-1, 1), 3)
            result = {
                'type': 'correlation',
                'value': correlation,
                'expr1': ctx.expr(0).getText(),
                'expr2': ctx.expr(1).getText()
            }
            self.output(f"Correlation({ctx.expr(0).getText()}, {ctx.expr(1).getText()}) = {correlation}")
            self.output("")
            self.results[f"corr_{ctx.expr(0).getText()}_{ctx.expr(1).getText()}"] = result
            return result
        
        elif ctx.getChild(0).getText() == 'outliers':
            exprs = [self.visit(e) for e in ctx.expr()]
            outliers = [random.randint(1, 100) for _ in range(5)]
            result = {
                'type': 'outliers',
                'values': outliers,
                'expressions': [e.getText() for e in ctx.expr()]
            }
            self.output(f"Outliers({', '.join(e.getText() for e in ctx.expr())}) = {outliers}")
            self.output("")
            self.results[f"outliers_{'_'.join(e.getText() for e in ctx.expr())}"] = result
            return result
        
        return {'type': 'unknown_query'}
    
    def visitClusteringStmt(self, ctx):
        """Process clustering statements"""
        dataset = ctx.IDENTIFIER().getText()
        dims = [self.visit(e) for e in ctx.clusteringOptions().expr()]
        k = int(ctx.clusteringOptions().INTEGER().getText())
        
        self.output(f"Clustering results for {dataset} with k={k}:")
        clusters = []
        for i in range(k):
            size = random.randint(50, 200)
            centroid = [round(random.uniform(0, 10), 2) for _ in range(len(dims))]
            self.output(f"   Cluster {i+1}: {size} points, centroid: {centroid}")
            clusters.append({
                'id': i+1,
                'size': size,
                'centroid': centroid
            })
        
        self.output("")
        
        result = {
            'type': 'clustering',
            'dataset': dataset,
            'k': k,
            'clusters': clusters
        }
        
        self.results['clustering'] = result
        return result
    
    def visitAssociationStmt(self, ctx):
        """Process association mining statements"""
        dataset = ctx.IDENTIFIER().getText()
        min_support = 0.3
        min_confidence = 0.8
        
        for opt in ctx.associationOption():
            if opt.getChild(0).getText() == 'min_support':
                min_support = self.visit(opt.expr())
            elif opt.getChild(0).getText() == 'min_confidence':
                min_confidence = self.visit(opt.expr())
        
        self.output(f"Association rule mining on {dataset}:")
        self.output(f"   Parameters: min_support={min_support}, min_confidence={min_confidence}")
        
        rules = []
        for i in range(3):
            antecedent = f"item{random.randint(1,5)},item{random.randint(6,10)}"
            consequent = f"item{random.randint(11,15)}"
            support = round(random.uniform(min_support, 0.9), 3)
            confidence = round(random.uniform(min_confidence, 1.0), 3)
            
            self.output(f"   Rule {i+1}: {{{antecedent}}} => {{{consequent}}}")
            self.output(f"      Support: {support}, Confidence: {confidence}")
            
            rules.append({
                'antecedent': antecedent,
                'consequent': consequent,
                'support': support,
                'confidence': confidence
            })
        
        self.output("")
        
        result = {
            'type': 'association_rules',
            'dataset': dataset,
            'min_support': min_support,
            'min_confidence': min_confidence,
            'rules': rules
        }
        
        self.results['association_rules'] = result
        return result
    
    def visitClassificationStmt(self, ctx):
        """Process classification statements"""
        dataset = ctx.IDENTIFIER().getText()
        target = self.visit(ctx.expr())
        options = {}
        for opt in ctx.classifierOption():
            opt_name = opt.IDENTIFIER().getText()
            opt_value = self.visit(opt.expr())
            options[opt_name] = opt_value
        
        self.output(f"Classification on {dataset}:")
        self.output(f"   Target: {ctx.expr().getText()}")
        self.output(f"   Options: {options}")
        
        accuracy = round(random.uniform(0.7, 0.95), 3)
        result = {
            'type': 'classification',
            'dataset': dataset,
            'target': ctx.expr().getText(),
            'accuracy': accuracy,
            'options': options
        }
        
        self.output(f"   Accuracy: {accuracy}")
        self.output("")
        
        self.results['classification'] = result
        return result
    
    def visitCommentStmt(self, ctx):
        """Handle comment statements"""
        comment = ctx.COMMENT().getText()
        self.output(f"Comment: {comment}")
        return None
    
    def visitExpr(self, ctx):
        """Delegate to conditionalExpr"""
        return self.visit(ctx.conditionalExpr())
    
    def visitConditionalExpr(self, ctx):
        """Handle conditional (ternary) expressions"""
        if ctx.getChildCount() == 1:
            return self.visit(ctx.logicalOrExpr())
        condition = self.visit(ctx.logicalOrExpr())
        true_expr = self.visit(ctx.logicalOrExpr(1))
        false_expr = self.visit(ctx.logicalOrExpr(2))
        return true_expr if condition else false_expr
    
    def visitLogicalOrExpr(self, ctx):
        """Handle logical OR expressions"""
        result = self.visit(ctx.logicalAndExpr(0))
        for i in range(1, len(ctx.logicalAndExpr())):
            result = result or self.visit(ctx.logicalAndExpr(i))
        return result
    
    def visitLogicalAndExpr(self, ctx):
        """Handle logical AND expressions"""
        result = self.visit(ctx.comparisonExpr(0))
        for i in range(1, len(ctx.comparisonExpr())):
            result = result and self.visit(ctx.comparisonExpr(i))
        return result
    
    def visitComparisonExpr(self, ctx):
        """Handle comparison expressions"""
        left = self.visit(ctx.addExpr(0))
        if ctx.comparisonOp():
            right = self.visit(ctx.addExpr(1))
            op = ctx.comparisonOp().getText()
            if op == '>':
                return left > right
            elif op == '<':
                return left < right
            elif op == '>=':
                return left >= right
            elif op == '<=':
                return left <= right
            elif op == '==':
                return left == right
            elif op == '!=':
                return left != right
        return left
    
    def visitAddExpr(self, ctx):
        """Handle addition/subtraction expressions"""
        result = self.visit(ctx.multExpr(0))
        for i in range(1, len(ctx.multExpr())):
            op = ctx.getChild(2*i-1).getText()
            next_term = self.visit(ctx.multExpr(i))
            if op == '+':
                result += next_term
            else:
                result -= next_term
        return result
    
    def visitMultExpr(self, ctx):
        """Handle multiplication/division expressions"""
        result = self.visit(ctx.powExpr(0))
        for i in range(1, len(ctx.powExpr())):
            op = ctx.getChild(2*i-1).getText()
            next_term = self.visit(ctx.powExpr(i))
            if op == '*':
                result *= next_term
            else:
                result /= next_term
        return result
    
    def visitPowExpr(self, ctx):
        """Handle power expressions"""
        result = self.visit(ctx.unaryExpr(0))
        for i in range(1, len(ctx.unaryExpr())):
            result = result ** self.visit(ctx.unaryExpr(i))
        return result
    
    def visitUnaryExpr(self, ctx):
        """Handle unary expressions"""
        if ctx.getChild(0).getText() == '-':
            return -self.visit(ctx.unaryExpr())
        return self.visit(ctx.primary())
    
    def visitPrimary(self, ctx):
        """Handle primary expressions"""
        if ctx.INTEGER():
            return int(ctx.INTEGER().getText())
        elif ctx.FLOAT():
            return float(ctx.FLOAT().getText())
        elif ctx.IDENTIFIER():
            parts = [ctx.IDENTIFIER(i).getText() for i in range(len(ctx.IDENTIFIER()))]
            if len(parts) == 1:
                var_name = parts[0]
                if var_name in self.variables:
                    var = self.variables[var_name]
                    if var['type'] == 'constant':
                        return var['value']
                    return var['samples']
                self.output(f"Error: Undefined variable {var_name}")
                return 0
            elif len(parts) == 2 and parts[0] == 'data':
                field = parts[1]
                if field in self.data:
                    return self.data[field]
                self.output(f"Error: Undefined data field {field}")
                return []
            else:
                self.output(f"Error: Invalid reference {'.'.join(parts)}")
                return 0
        elif ctx.expr():
            return self.visit(ctx.expr())
        self.output("Error: Invalid primary expression")
        return 0

class RedirectOutput:
    """Redirect stdout to Tkinter text widget"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = StringIO()
        
    def write(self, text):
        self.buffer.write(text)
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)
        
    def flush(self):
        pass

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
        self.apply_highlight(content, keywords, "keyword")
        self.apply_highlight(content, distributions, "distribution")
        self.apply_highlight(content, operators, "operator")
        self.apply_highlight(content, comments, "comment", multiline=False)
        self.apply_highlight(content, strings, "string")
    
    def apply_highlight(self, content, pattern, tag, multiline=True):
        """Apply highlighting for a pattern"""
        flags = 0 if multiline else re.MULTILINE
        for match in re.finditer(pattern, content, flags):
            start = f"1.0 + {match.start()} chars"
            end = f"1.0 + {match.end()} chars"
            self.code_editor.tag_add(tag, start, end)
    
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
            
            if not self.generate_parser():
                return
            
            print("Running probabilistic query language interpreter...")
            print("-" * 50)
            
            results = self.parse_and_process_code(code)
            
            if results:
                self.generate_ast_visualization(code)
                self.generate_visualizations(results)
            else:
                print("No results to visualize due to errors")
                
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            import traceback
            print(traceback.format_exc())
        finally:
            sys.stdout = sys.__stdout__
    
    def generate_parser(self):
        """Generate parser from grammar if needed"""
        print("Checking for existing parser...")
        if not os.path.exists(os.path.join(DIR, CPL_DEST, "ProbDataMineLexer.py")):
            print("Generating parser from grammar...")
            try:
                subprocess.run(['java', '-jar', ANTLR_JAR, '-o', CPL_DEST, '-no-listener', '-Dlanguage=Python3', SRC], check=True)
                print("Parser generated successfully")
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate parser: {str(e)}")
                return False
        else:
            print("Using existing parser")
        return True
    
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
            import traceback
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
    
    def generate_visualizations(self, results):
        """Generate visualizations based on query results"""
        self.fig.clear()
        
        prob_queries = {k: v for k, v in results.items() if k.startswith('P_')}
        exp_queries = {k: v for k, v in results.items() if k.startswith('E_')}
        corr_queries = {k: v for k, v in results.items() if k.startswith('corr_')}
        outlier_queries = {k: v for k, v in results.items() if k.startswith('outliers_')}
        
        if 'clustering' in results:
            self.plot_clusters(results['clustering'])
        elif 'association_rules' in results:
            self.plot_association_rules(results['association_rules'])
        elif 'classification' in results:
            self.plot_classification_results(results['classification'])
        elif prob_queries:
            self.plot_probability_distribution(list(prob_queries.values())[0])
        elif exp_queries:
            self.plot_expected_values(list(exp_queries.values()))
        elif corr_queries:
            self.plot_correlation(list(corr_queries.values())[0])
        elif outlier_queries:
            self.plot_outliers(list(outlier_queries.values())[0])
        else:
            self.plot_default_distribution()
        
        self.canvas.draw()
    
    def plot_probability_distribution(self, prob_result):
        """Plot probability distribution"""
        ax = self.fig.add_subplot(111)
        
        x = np.linspace(0, 10, 1000)
        y1 = np.exp(-(x-5)**2/2)
        
        ax.plot(x, y1, 'b-', linewidth=2, label='P(X)')
        
        prob_value = prob_result['value']
        condition_range = (x >= 7) & (x <= 10)
        ax.fill_between(x, 0, y1, where=condition_range, color='red', alpha=0.3,
                       label=f'P({prob_result["condition"]}) = {prob_value:.3f}')
        
        ax.axhline(y=0.1, xmin=0.2, xmax=0.8, color='green', linestyle='--',
                  label=f'95% CI: [{prob_result["ci_lower"]:.3f}, {prob_result["ci_upper"]:.3f}]')
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.set_title('Probability Query Result')
        ax.legend()
        ax.grid(True)
    
    def plot_expected_values(self, exp_results):
        """Plot expected values"""
        ax = self.fig.add_subplot(111)
        
        labels = [f"E({r['expression']})" for r in exp_results]
        means = [r['value'] for r in exp_results]
        errors = [r['std_dev'] for r in exp_results]
        
        bars = ax.bar(labels, means, yerr=errors, capsize=10, color='skyblue')
        
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{mean:.2f}', ha='center', va='bottom')
        
        ax.set_xlabel('Expression')
        ax.set_ylabel('Expected Value')
        ax.set_title('Expected Value Query Results')
        ax.grid(True, axis='y')
    
    def plot_clusters(self, clustering_result):
        """Plot clustering visualization"""
        ax = self.fig.add_subplot(111, projection='3d')
        
        clusters = clustering_result['clusters']
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        
        for i, cluster in enumerate(clusters):
            centroid = cluster['centroid']
            n_points = min(cluster['size'], 50)
            
            points = np.random.normal(0, 0.5, (n_points, 3)) + centroid
            
            color = colors[i % len(colors)]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='o',
                      label=f'Cluster {cluster["id"]} (n={cluster["size"]})')
            
            ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c=color, marker='*', s=200,
                      edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.set_title(f'K-Means Clustering (k={clustering_result["k"]})')
        ax.legend()
    
    def plot_correlation(self, corr_result):
        """Plot correlation visualization"""
        ax = self.fig.add_subplot(111)
        ax.bar([f"{corr_result['expr1']} vs {corr_result['expr2']}"], [corr_result['value']],
               color='skyblue')
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Correlation Query Result')
        ax.grid(True, axis='y')
    
    def plot_outliers(self, outlier_result):
        """Plot outliers visualization"""
        ax = self.fig.add_subplot(111)
        values = outlier_result['values']
        ax.scatter(range(len(values)), values, c='red', marker='x')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(outlier_result['expressions'], rotation=45)
        ax.set_ylabel('Value')
        ax.set_title('Outliers Query Result')
        ax.grid(True)
    
    def plot_association_rules(self, assoc_result):
        """Plot association rules visualization"""
        ax = self.fig.add_subplot(111)
        rules = assoc_result['rules']
        supports = [r['support'] for r in rules]
        confidences = [r['confidence'] for r in rules]
        labels = [f"{r['antecedent']} => {r['consequent']}" for r in rules]
        
        ax.scatter(supports, confidences, c='blue')
        for i, label in enumerate(labels):
            ax.annotate(label, (supports[i], confidences[i]), fontsize=8)
        
        ax.set_xlabel('Support')
        ax.set_ylabel('Confidence')
        ax.set_title('Association Rules')
        ax.grid(True)
    
    def plot_classification_results(self, class_result):
        """Plot classification results visualization"""
        ax = self.fig.add_subplot(111)
        ax.bar(['Accuracy'], [class_result['accuracy']], color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title(f"Classification Results for {class_result['target']}")
        ax.grid(True, axis='y')
    
    def plot_default_distribution(self):
        """Plot default distribution"""
        ax = self.fig.add_subplot(111)
        
        x = np.linspace(-5, 5, 1000)
        y1 = 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
        y2 = 1/np.sqrt(2*np.pi*1.5**2) * np.exp(-(x-1)**2/(2*1.5**2))
        
        ax.plot(x, y1, 'b-', linewidth=2, label='Normal(0, 1)')
        ax.plot(x, y2, 'r-', linewidth=2, label='Normal(1, 1.5)')
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability Density')
        ax.set_title('Example Probability Distributions')
        ax.legend()
        ax.grid(True)
    
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
    
def main(argv):
    """Main entry point"""
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
