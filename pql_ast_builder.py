from CompiledFiles.ProbDataMineListener import ProbDataMineListener
from pql_ast_util import *

class ASTBuilder(ProbDataMineListener):
    """
    AST Builder that creates clean Abstract Syntax Trees
    by visiting the parse tree and extracting only essential information
    """
    
    def __init__(self):
        self.ast_stack = []
        self.current_ast = None
    
    def build_ast(self, parser, tree):
        """Build AST from parse tree"""
        from antlr4 import ParseTreeWalker
        
        walker = ParseTreeWalker()
        walker.walk(self, tree)
        return self.current_ast
    
    def enterProgram(self, ctx):
        """Start building the program AST"""
        self.ast_stack.append([])  # List to collect statements
    
    def exitProgram(self, ctx):
        """Finish building the program AST"""
        statements = self.ast_stack.pop()
        self.current_ast = Program(statements=statements)
    
    def exitStatement(self, ctx):
        """Add completed statement to current program"""
        if self.ast_stack and hasattr(self, '_current_statement'):
            self.ast_stack[-1].append(self._current_statement)
            delattr(self, '_current_statement')
    
    def exitDataLoadStatement(self, ctx):
        """Build DataLoad AST node"""
        # Extract filename (remove quotes)
        filename_token = ctx.STRING()
        filename = filename_token.getText().strip('"\'') if filename_token else ""
        
        # Extract options if present
        options = {}
        if ctx.dataLoadOptions():
            for option_ctx in ctx.dataLoadOptions().dataLoadOption():
                key = option_ctx.ID().getText()
                value = option_ctx.STRING().getText().strip('"\'')
                options[key] = value
        
        self._current_statement = DataLoadStmt(filename=filename, options=options)
    
    def exitVariableDeclaration(self, ctx):
        """Build Variable Declaration AST node"""
        # Get variable name
        var_name = ctx.ID().getText()
        
        # Get distribution (should be built by exitDistribution)
        distribution = getattr(self, '_current_distribution', None)
        if distribution:
            self._current_statement = VariableDecl(name=var_name, distribution=distribution)
            delattr(self, '_current_distribution')
    
    def exitDistribution(self, ctx):
        """Build Distribution AST node"""
        # Get distribution type
        dist_type = ctx.ID().getText()
        
        # Get parameters
        params = []
        if ctx.expressionList():
            params = getattr(self, '_current_expressions', [])
            if hasattr(self, '_current_expressions'):
                delattr(self, '_current_expressions')
        
        self._current_distribution = Distribution(type=dist_type, params=params)
    
    def exitQueryStatement(self, ctx):
        """Build Query AST node"""
        # Determine query type from context
        query_type = None
        if ctx.probabilityQuery():
            query_type = "P"
        elif ctx.expectationQuery():
            query_type = "E"
        elif ctx.correlationQuery():
            query_type = "correlation"
        elif ctx.outlierQuery():
            query_type = "outliers"
        
        # Get parameters
        params = getattr(self, '_current_expressions', [])
        if hasattr(self, '_current_expressions'):
            delattr(self, '_current_expressions')
        
        # Get condition if present
        condition = getattr(self, '_current_condition', None)
        if hasattr(self, '_current_condition'):
            delattr(self, '_current_condition')
        
        self._current_statement = Query(type=query_type, params=params, condition=condition)
    
    def exitExpressionList(self, ctx):
        """Collect list of expressions"""
        expressions = []
        for expr_ctx in ctx.expression():
            # Each expression should have been processed
            if hasattr(self, '_temp_expressions'):
                expressions.extend(self._temp_expressions)
        
        self._current_expressions = expressions
        if hasattr(self, '_temp_expressions'):
            delattr(self, '_temp_expressions')
    
    def exitExpression(self, ctx):
        """Build expression AST nodes"""
        if not hasattr(self, '_temp_expressions'):
            self._temp_expressions = []
        
        # Handle different types of expressions
        if ctx.binaryExpression():
            expr = self._build_binary_expression(ctx.binaryExpression())
        elif ctx.ID():
            expr = Variable(name=ctx.ID().getText())
        elif ctx.NUMBER():
            # Handle both integers and floats
            number_text = ctx.NUMBER().getText()
            value = float(number_text) if '.' in number_text else int(number_text)
            expr = Literal(value=value)
        elif ctx.STRING():
            expr = Literal(value=ctx.STRING().getText().strip('"\''))
        elif ctx.BOOLEAN():
            expr = Literal(value=ctx.BOOLEAN().getText().lower() == 'true')
        else:
            # Default case
            expr = Literal(value=ctx.getText())
        
        self._temp_expressions.append(expr)
    
    def _build_binary_expression(self, ctx):
        """Build binary expression AST"""
        # This would need to be implemented based on your grammar
        # For now, return a placeholder
        left = Variable(name="left")  # You'd extract this from context
        right = Variable(name="right")  # You'd extract this from context
        op = "+"  # You'd extract the operator from context
        
        return BinaryOp(op=op, left=left, right=right)
    
    def exitConditionClause(self, ctx):
        """Build condition for queries"""
        # Extract condition expression
        if ctx.expression():
            # The expression should have been processed
            expressions = getattr(self, '_temp_expressions', [])
            if expressions:
                self._current_condition = expressions[0]  # Take first expression as condition


class ImprovedPQLParser:
    """Enhanced PQL Parser with better AST generation"""
    
    def __init__(self):
        self.parser = None
        self.ast_builder = ASTBuilder()
    
    def parse_to_ast(self, code):
        """Parse code directly to AST, skipping detailed parse tree"""
        try:
            # Create input stream and lexer
            from antlr4 import InputStream, CommonTokenStream
            from CompiledFiles.ProbDataMineLexer import ProbDataMineLexer
            from CompiledFiles.ProbDataMineParser import ProbDataMineParser
            
            input_stream = InputStream(code)
            lexer = ProbDataMineLexer(input_stream)
            token_stream = CommonTokenStream(lexer)
            parser = ProbDataMineParser(token_stream)
            
            # Parse to get parse tree
            tree = parser.program()
            
            # Convert parse tree to AST
            ast = self.ast_builder.build_ast(parser, tree)
            
            return {
                'success': True,
                'ast': ast,
                'tree_json': self._ast_to_json(ast)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _ast_to_json(self, ast_node):
        """Convert AST to JSON for JavaScript visualization"""
        if ast_node is None:
            return None
        
        # Use the same conversion logic as your JavaScript function
        node_type = ast_node.__class__.__name__
        
        if node_type == 'Program':
            return {
                'constructor': {'name': 'Program'},
                'statements': [self._ast_to_json(stmt) for stmt in ast_node.statements]
            }
        elif node_type == 'DataLoadStmt':
            return {
                'constructor': {'name': 'DataLoadStmt'},
                'filename': ast_node.filename,
                'options': ast_node.options
            }
        elif node_type == 'VariableDecl':
            return {
                'constructor': {'name': 'VariableDecl'},
                'name': ast_node.name,
                'distribution': self._ast_to_json(ast_node.distribution)
            }
        elif node_type == 'Distribution':
            return {
                'constructor': {'name': 'Distribution'},
                'type': ast_node.type,
                'params': [self._ast_to_json(param) for param in ast_node.params]
            }
        elif node_type == 'Query':
            result = {
                'constructor': {'name': 'Query'},
                'type': ast_node.type,
                'params': [self._ast_to_json(param) for param in ast_node.params]
            }
            if ast_node.condition:
                result['condition'] = self._ast_to_json(ast_node.condition)
            return result
        elif node_type == 'BinaryOp':
            return {
                'constructor': {'name': 'BinaryOp'},
                'op': ast_node.op,
                'left': self._ast_to_json(ast_node.left),
                'right': self._ast_to_json(ast_node.right)
            }
        elif node_type == 'Literal':
            return {
                'constructor': {'name': 'Literal'},
                'value': ast_node.value
            }
        elif node_type == 'Variable':
            return {
                'constructor': {'name': 'Variable'},
                'name': ast_node.name
            }
        else:
            return {
                'constructor': {'name': node_type},
                'data': str(ast_node)
            }