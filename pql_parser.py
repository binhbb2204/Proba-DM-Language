import os
import sys
from antlr4 import CommonTokenStream, InputStream, ParseTreeWalker
from antlr4.error.ErrorListener import ErrorListener

# Adjust path to ensure CompiledFiles can be imported
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CompiledFiles'))

# Import ANTLR4 generated classes from CompiledFiles
from CompiledFiles.ProbDataMineLexer import ProbDataMineLexer
from CompiledFiles.ProbDataMineParser import ProbDataMineParser
from CompiledFiles.ProbDataMineListener import ProbDataMineListener
from CompiledFiles.ProbDataMineVisitor import ProbDataMineVisitor

class PQLSyntaxErrorListener(ErrorListener):
    """Custom error listener for capturing syntax errors"""
    
    def __init__(self):
        super().__init__()
        self.errors = []
    
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append({
            'line': line,
            'column': column,
            'message': msg
        })

class PQLParser:
    """PQL Parser that uses ANTLR4-generated classes"""
    
    def __init__(self, grammar_path='ProbDataMine.g4'):
        """Initialize the parser with the grammar file path"""
        self.grammar_path = grammar_path
        self.initialized = False
    
    def initialize(self):
        """Initialize the parser (called once, can be called again to reload)"""
        self.initialized = True
        return True
    
    def parse(self, code):
        """Parse PQL code and return results"""
        try:
            input_stream = InputStream(code)
            
            lexer = ProbDataMineLexer(input_stream)
            lexer.removeErrorListeners()
            error_listener = PQLSyntaxErrorListener()
            lexer.addErrorListener(error_listener)
            
            token_stream = CommonTokenStream(lexer)
            
            parser = ProbDataMineParser(token_stream)
            parser.removeErrorListeners()
            parser.addErrorListener(error_listener)
            
            tree = parser.program()
            
            if error_listener.errors:
                return {
                    'success': False,
                    'errors': error_listener.errors
                }
            else:
                return {
                    'success': True,
                    'tree': tree,
                    'parser': parser
                }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'errors': [{
                    'line': 0,
                    'column': 0,
                    'message': f"Parser error: {str(e)}"
                }]
            }
    
    def visualize_tree(self, code):
        """Generate a visualization of the parse tree"""
        result = self.parse(code)
        
        if not result['success']:
            return result
        
        tree = result['tree']
        parser = result['parser']
        
        return {
            'success': True,
            'tree_json': self._convert_tree_to_json(tree, parser),
            'parse_tree': tree.toStringTree(recog=parser)
        }
    
    def _convert_tree_to_json(self, node, parser):
        """Convert an ANTLR parse tree node to a JSON structure for visualization"""
        if node is None:
            return None
            
        if node.getChildCount() == 0:
            return {
                'name': node.getText(),  
                'type': 'terminal',
                'children': []
            }
        else:
            rule_name = ''
            if hasattr(parser, 'ruleNames') and hasattr(node, 'getRuleIndex'):
                rule_index = node.getRuleIndex()
                if 0 <= rule_index < len(parser.ruleNames):
                    rule_name = parser.ruleNames[rule_index]
                else:
                    rule_name = str(type(node).__name__)
            
            result = {
                'name': rule_name,
                'type': 'non-terminal',
                'children': []
            }
            
            for i in range(node.getChildCount()):
                child = node.getChild(i)
                child_json = self._convert_tree_to_json(child, parser)
                if child_json:
                    result['children'].append(child_json)
            
            return result