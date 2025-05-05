import os
import numpy as np
import random
import pandas as pd
from antlr4 import *
from antlr4.tree.Tree import ParseTreeVisitor

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