import numpy as np
import re

class PQLVariableHandler:
    """Handler for PQL variable operations"""
    
    @staticmethod
    def parse_variable_assignment(statement, data, variables):
        """Parse and execute a variable assignment statement"""
        # Format: var x = data.column;
        try:
            # Extract variable name and expression
            match = re.match(r'var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*?);', statement)
            if not match:
                return None, "Invalid variable assignment syntax"
                
            var_name = match.group(1)
            expression = match.group(2).strip()
            
            # Handle data references
            if '.' in expression:
                parts = expression.split('.')
                if len(parts) != 2:
                    return None, f"Invalid data reference: {expression}"
                    
                dataset_name, column = parts
                
                # Handle 'data' as default dataset name
                if dataset_name == 'data':
                    dataset_name = 'default'
                
                if dataset_name not in data:
                    return None, f"Dataset '{dataset_name}' not found"
                    
                if column not in data[dataset_name]:
                    return None, f"Column '{column}' not found in dataset '{dataset_name}'"
                    
                # Assign variable from data
                variable_value = data[dataset_name][column].values
                variables[var_name] = variable_value
                
                return var_name, None
            
            # Handle other expressions later
            return None, f"Unsupported expression: {expression}"
            
        except Exception as e:
            return None, f"Error in variable assignment: {str(e)}"