import numpy as np
import re

class PQLVariableHandler:
    @staticmethod
    def parse_variable_assignment(statement, data, variables):
        """Parse and execute a variable assignment statement"""
        try:
            # Extract variable name and expression
            match = re.match(r'var\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*?);', statement)
            if not match:
                return None, "Invalid variable assignment syntax"
                
            var_name = match.group(1)
            expression = match.group(2).strip()
            
            # Handle dataset filtering with conditions
            if '[' in expression and ']' in expression:
                dataset_name = expression[:expression.find('[')].strip()
                filter_expr = expression[expression.find('[')+1:expression.find(']')].strip()
                
                if dataset_name not in data:
                    return None, f"Dataset '{dataset_name}' not found"
                    
                # Parse filter expression (e.g., "age < 150")
                operators = ['>=', '<=', '==', '>', '<', '!=']
                operator = None
                for op in operators:
                    if op in filter_expr:
                        operator = op
                        col, val = filter_expr.split(op)
                        break
                        
                if operator:
                    col = col.strip()
                    try:
                        val = float(val.strip())
                    except ValueError:
                        return None, f"Invalid numeric value in filter: {val}"
                    
                    df = data[dataset_name]
                    try:
                        if operator == '>':
                            filtered_df = df[df[col] > val].copy()
                        elif operator == '<':
                            filtered_df = df[df[col] < val].copy()
                        elif operator == '>=':
                            filtered_df = df[df[col] >= val].copy()
                        elif operator == '<=':
                            filtered_df = df[df[col] <= val].copy()
                        elif operator == '==':
                            filtered_df = df[df[col] == val].copy()
                        elif operator == '!=':
                            filtered_df = df[df[col] != val].copy()
                        
                        # Store filtered dataset in data dictionary
                        data[var_name] = filtered_df
                        return var_name, None
                        
                    except KeyError:
                        return None, f"Column '{col}' not found in dataset"
                
                return None, "Invalid filter expression"
            
            # Handle data references (existing code for data.column syntax)
            elif '.' in expression:
                # ...existing code...
                return var_name, None
            
            return None, f"Unsupported expression: {expression}"
            
        except Exception as e:
            return None, f"Error in variable assignment: {str(e)}"