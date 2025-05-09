import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import json
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from io import StringIO
from pql_variable_handler import PQLVariableHandler

class PQLExecutor:
    def __init__(self):
        # Store loaded datasets
        self.data = {}
        # Store defined variables
        self.variables = {}
        # Store distribution definitions
        self.distributions = {}
        
    def execute(self, code, parse_result):
        """Execute PQL code and return results"""
        if not parse_result['success']:
            return {
                'success': False,
                'error': 'Syntax error in PQL code',
                'details': parse_result['errors']
            }
            
        try:
            # Clear execution context for fresh run
            self.data = {}
            self.variables = {}
            self.distributions = {}
            
            # print("\n[INIT] Cleared data, variables, distributions")

            # Process the code line by line
            lines = code.strip().split('\n')
            results = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue

                # print(f"\n[LINE] {line}")
                # print("[STATE BEFORE] data:", self.data)
                # print("[STATE BEFORE] variables:", self.variables)
                # print("[STATE BEFORE] distributions:", self.distributions)
                
                if line.startswith('load_data'):
                    result = self._execute_load_data(line)
                    results.append(result)
                    testvar = self._execute_variable_declaration(line)
                    
                elif line.startswith('var') and '=' in line:
                    var_name, error = PQLVariableHandler.parse_variable_assignment(line, self.data, self.variables)
                    if error:
                        results.append({'type': 'error', 'message': error})
                    else:
                        results.append({
                            'type': 'variable_assignment',
                            'variable': var_name,
                            'source': 'data'
                        })
                        
                elif line.startswith('var') and 'follows' in line:
                    result = self._execute_variable_declaration(line)
                    results.append(result)
                    
                elif line.startswith('query'):
                    result = self._execute_query(line)
                    results.append(result)
                
                # print("[STATE AFTER] data:", self.data)
                # print("[STATE AFTER] variables:", self.variables)
                # print("[STATE AFTER] distributions:", self.distributions)

                # Add other statement types as needed
            
            # print("\n[FINAL STATE]")
            # print("data:", self.data)
            # print("variables:", self.variables)
            # print("distributions:", self.distributions)

            return {
                'success': True,
                'results': results
            }
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_params(self, statement):
        """Extract parameters from a statement with parentheses"""
        start = statement.find('(')
        end = statement.rfind(')')
        if start == -1 or end == -1:
            return []
            
        params_str = statement[start+1:end]
        # Simple parser for comma-separated params that respects quoted strings
        params = []
        current_param = ''
        in_quotes = False
        
        for char in params_str:
            if char == '"' and (not in_quotes or (in_quotes and current_param[-1:] != '\\')):
                in_quotes = not in_quotes
                current_param += char
            elif char == ',' and not in_quotes:
                params.append(current_param.strip())
                current_param = ''
            else:
                current_param += char
                
        if current_param:
            params.append(current_param.strip())
            
        return params
    
    def _execute_load_data(self, statement):
        """Execute a load_data statement"""
        try:
            # Parse load_data("filename.csv", name: dataset_name);
            filename_start = statement.find('"')
            filename_end = statement.find('"', filename_start + 1)
            
            if filename_start == -1 or filename_end == -1:
                return {'type': 'error', 'message': 'Invalid filename in load_data statement'}
                
            filename = statement[filename_start+1:filename_end]
            
            # Extract options
            options = {}
            options_part = statement[filename_end+1:]
            
            # Check for name option
            import re
            name_match = re.search(r'name\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)', options_part)
            if name_match:
                dataset_name = name_match.group(1)
                options['name'] = dataset_name
            else:
                dataset_name = 'default'
            
            # Load the data
            try:
                import pandas as pd
                import os
                
                # First try loading the file directly
                try:
                    df = pd.read_csv(filename)
                    actual_filename = filename
                except FileNotFoundError:
                    # If that fails, try looking in the data/ directory
                    data_path = os.path.join('data', filename)
                    df = pd.read_csv(data_path)
                    actual_filename = data_path
                
                self.data[dataset_name] = df
                
                # Prepare preview data (first few rows)
                preview_data = df.to_dict('records')
                
                return {
                    'type': 'load_data',
                    'dataset': dataset_name,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'preview': preview_data
                }
            except Exception as e:
                return {'type': 'error', 'message': f'Error loading data: {str(e)}'}
                
        except Exception as e:
            return {'type': 'error', 'message': f'Error parsing load_data statement: {str(e)}'}

    def _resolve_data_reference(self, data_ref):
        """Resolve data reference in the form 'data.column', 'dataset_name.column', or just 'dataset_name'"""
        if data_ref in self.data:
            # Return the first numeric column as a default
            numeric_cols = self.data[data_ref].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return self.data[data_ref][numeric_cols[0]].values
            return None
            
        if '.' in data_ref:
            parts = data_ref.split('.')
            if len(parts) != 2:
                return None
                
            dataset_name, column = parts
            
            if dataset_name == 'data':
                if 'default' in self.data and column in self.data['default']:
                    return self.data['default'][column].values
            elif dataset_name in self.data and column in self.data[dataset_name]:
                return self.data[dataset_name][column].values
                
        return None
    
    def _execute_variable_declaration(self, statement):
        """Execute a variable declaration statement with distribution"""
        # Parse var x follows Dist(params)
        parts = statement.split()
        if len(parts) < 3:
            return {'type': 'error', 'message': 'Invalid variable declaration'}
            
        var_name = parts[1]
        distribution_part = statement[statement.find('follows')+7:].strip(';').strip()
        
        # Store the distribution definition
        self.distributions[var_name] = distribution_part
        
        # Generate actual random samples based on the distribution
        samples = self._generate_from_distribution(distribution_part, 1000)
        self.variables[var_name] = samples
        
        return {
            'type': 'variable_declaration',
            'variable': var_name,
            'distribution': distribution_part,
            'samples': len(samples),
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'visualization': self._generate_histogram(samples, var_name)
        }
    
    def _generate_from_distribution(self, distribution_expr, size=1000):
        """Generate random samples from a distribution"""
        if 'Normal' in distribution_expr:
            # Normal(mean, std)
            params = self._extract_params(distribution_expr)
            if len(params) != 2:
                return np.zeros(size)
                
            mean = float(params[0])
            std = float(params[1])
            return np.random.normal(mean, std, size)
            
        elif 'LogNormal' in distribution_expr:
            params = self._extract_params(distribution_expr)
            if len(params) != 2:
                return np.zeros(size)
                
            meanlog = float(params[0])
            sdlog = float(params[1])
            return np.random.lognormal(meanlog, sdlog, size)
            
        elif 'Poisson' in distribution_expr:
            params = self._extract_params(distribution_expr)
            if len(params) != 1:
                return np.zeros(size)
                
            lam = float(params[0])
            return np.random.poisson(lam, size)
            
        elif 'Bernoulli' in distribution_expr:
            params = self._extract_params(distribution_expr)
            if len(params) != 1:
                return np.zeros(size)
                
            p = float(params[0])
            return np.random.binomial(1, p, size)
            
        elif 'EmpiricalDistribution' in distribution_expr:
            # Extract data reference
            data_ref = distribution_expr[distribution_expr.find('(')+1:distribution_expr.find(')')]
            if '.' not in data_ref:
                return np.zeros(size)
                
            data_name, column = data_ref.split('.')
            if data_name != 'data' or column not in self.data.get('default', pd.DataFrame()):
                return np.zeros(size)
                
            # Sample from empirical distribution
            data_values = self.data['default'][column].values
            return np.random.choice(data_values, size=size)
            
        # Add other distributions as needed
        return np.zeros(size)
        
    def _generate_histogram(self, data, title):
        """Generate a base64 encoded histogram image"""
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=30, alpha=0.7, color='skyblue')
        plt.title(f"Distribution of {title}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def _execute_variable_assignment(self, statement):
        """Execute a variable assignment statement"""
        # Parse var x = expr;
        parts = statement.split('=', 1)
        if len(parts) != 2:
            return {'type': 'error', 'message': 'Invalid variable assignment'}
            
        var_part = parts[0].strip()
        var_name = var_part.replace('var', '').strip()
        
        expr_part = parts[1].strip().rstrip(';').strip()
        
        # Evaluate expression (simplified)
        try:
            # For demo, just evaluate simple expressions
            value = eval(expr_part, {"__builtins__": {}}, self.variables)
            self.variables[var_name] = value
            
            return {
                'type': 'variable_assignment',
                'variable': var_name,
                'value': value
            }
            
        except Exception as e:
            return {'type': 'error', 'message': f'Error in expression: {str(e)}'}
    
    def _execute_query(self, statement):
        """Execute a query statement"""
        try:
            # Extract query type and parameters
            query_part = statement[statement.find('query')+5:].strip(';').strip()
            
            if query_part.startswith('P('):
                # Probability query
                params = query_part[2:-1].strip()
                # print(f"Debug: Processing probability for {params}")
                return self._execute_probability_query(params)
                
            elif query_part.startswith('E('):
                # Expected value query
                params = query_part[2:-1].strip()
                print(f"Debug: Processing expectation for {params}")
                return self._execute_expectation_query(params)
                
            elif query_part.startswith('correlation('):
                # Correlation query
                params = query_part[12:-1].strip().split(',')
                # print(f"Debug: Processing correlation for {params}")
                return self._execute_correlation_query(params)
                
            elif query_part.startswith('outliers('):
                # Outliers query
                params = query_part[9:-1].strip().split(',')
                print(f"Debug: Processing outliers for {params}")
                return self._execute_outliers_query(params)
                
            else:
                return {'type': 'error', 'message': f'Unknown query type: {query_part}'}
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'type': 'error', 'message': f'Error executing query: {str(e)}'}
    
    # def _execute_probability_query(self, params):
    #     """Execute a probability query P(X > 0)"""
    #     print("\n--- Executing Probability Query ---")
    #     print(f"params: {params}")
    #     if isinstance(params, str):
    #         params = [params]
    #     if not params:
    #         print("Error: No parameters provided.")
    #         return {'type': 'error', 'message': 'Invalid probability query'}
            
    #     # Handle conditional probability
    #     condition = None
    #     if len(params) > 1 and '|' in params[0]:
    #         # Handle conditional probability
    #         variable, condition = params[0].split('|')
    #         variable = variable.strip()
    #         condition = condition.strip()
    #         print(f"Conditional Probability Detected")
    #         print(f"Variable: {variable}")
    #         print(f"Condition: {condition}")
    #     else:
    #         variable = params[0]
    #         print(f"Variable: {variable} (no condition)")
            
    #     # Calculate probability
    #     try:
    #         print("Available variables:", self.variables.keys())
    #         print(f"Checking if '{variable}' in self.variables")
    #         if variable in self.variables:
    #             data = self.variables[variable]
    #             print(f"Debug: Resolved data type: {type(data)}, shape: {getattr(data, 'shape', None)}")
    #             print(f"Data for variable '{variable}': {data}")
    #             print(f"Data type: {type(data)}")
    #             if isinstance(data, np.ndarray):
    #                 if condition:
    #                     # Simplified conditional probability
    #                     result = float(np.mean(data > 0))
    #                     print(f"Conditional result: {result}")
    #                     return {
    #                         'type': 'query_result',
    #                         'query_type': 'probability',
    #                         'variable': variable,
    #                         'condition': condition,
    #                         # 'result': float(np.mean(data > 0))  # Placeholder
    #                         'result': result,  # Placeholder
    #                     }
    #                 else:
    #                     result = float(np.mean(data > 0))
    #                     print(f"Unconditional result: {result}")
    #                     return {
    #                         'type': 'query_result',
    #                         'query_type': 'probability',
    #                         'variable': variable,
    #                         'result': result,
    #                         'visualization': self._generate_probability_visualization(data, variable)
    #                     }
    #             else:
    #                 print(f"Variable '{variable}' not found in self.variables.")
    #                 print(f"Available variables: {list(self.variables.keys())}")
    #         return {'type': 'error', 'message': f'Cannot compute probability for {variable}'}
            
    #     except Exception as e:
    #         import traceback
    #         traceback.print_exc()
    #         return {'type': 'error', 'message': f'Error in probability query: {str(e)}'}
    # def _execute_probability_query(self, params):
    #     """Execute a probability query P(X > value) or P(X) for histogram"""
    #     print(f"[DEBUG] Received params: {params}")
        
    #     if not params:
    #         print("[DEBUG] Invalid params (empty)")
    #         return {'type': 'error', 'message': 'Invalid probability query'}

    #     condition = None
    #     variable = None

    #     # Detect condition in params
    #     for op in ['>', '<', '==']:
    #         if op in params:
    #             parts = params.split(op, 1)
    #             if len(parts) == 2:
    #                 variable = parts[0].strip()
    #                 try:
    #                     value = float(parts[1].strip())
    #                 except ValueError:
    #                     print(f"[DEBUG] Could not convert condition value to float: {parts[1].strip()}")
    #                     return {'type': 'error', 'message': 'Invalid numeric value in condition'}
    #                 condition = (op, value)
    #                 print(f"[DEBUG] Parsed condition: variable={variable}, op={op}, value={value}")
    #                 break

    #     if condition is None:
    #         variable = params.strip()
    #         print(f"[DEBUG] No condition found, using variable: {variable}")

    #     # Resolve data
    #     data = None
    #     if variable in self.variables:
    #         data = self.variables[variable]
    #         print(f"[DEBUG] Found variable '{variable}' in self.variables")
    #     else:
    #         data = self._resolve_data_reference(variable)
    #         if data is not None:
    #             print(f"[DEBUG] Resolved variable '{variable}' using _resolve_data_reference")

    #     if data is None:
    #         print(f"[DEBUG] Failed to resolve data for variable: {variable}")
    #         return {'type': 'error', 'message': f'Cannot resolve variable {variable}'}

    #     if not isinstance(data, (np.ndarray, list)):
    #         print(f"[DEBUG] Resolved data is not array-like: type={type(data)}")
    #         return {'type': 'error', 'message': f'Variable {variable} is not numeric data'}

    #     data = np.array(data)
    #     print(f"[DEBUG] Data sample: {data[:10]} (length: {len(data)})")

    #     if condition:
    #         op, val = condition
    #         print(f"[DEBUG] Applying condition: {op} {val}")
    #         if op == '>':
    #             prob = np.mean(data > val)
    #         elif op == '<':
    #             prob = np.mean(data < val)
    #         elif op == '==':
    #             prob = np.mean(data == val)
    #         else:
    #             print(f"[DEBUG] Unsupported operator: {op}")
    #             return {'type': 'error', 'message': f'Unsupported operator {op}'}

    #         print(f"[DEBUG] Computed probability: {prob}")
    #         return {
    #             'type': 'query_result',
    #             'query_type': 'probability',
    #             'variable': variable,
    #             'condition': f"{variable} {op} {val}",
    #             'result': float(prob),
    #             'visualization': self._generate_probability_visualization(data, variable)
    #         }
    #     else:
    #         print("[DEBUG] No condition specified; returning histogram")
    #         return {
    #             'type': 'query_result',
    #             'query_type': 'distribution',
    #             'variable': variable,
    #             'visualization': self._generate_histogram(data, variable)
    #         }
    # def _execute_probability_query(self, params):
    #     """Execute a probability query P(X > 0)"""
    #     print("\n--- Executing Probability Query ---")
    #     print(f"params: {params}")

    #     if isinstance(params, str):
    #         params = [params]
    #     if not params:
    #         print("Error: No parameters provided.")
    #         return {'type': 'error', 'message': 'Invalid probability query'}

    #     # Handle conditional probability
    #     condition = None
    #     if len(params) > 1 and '|' in params[0]:
    #         variable, condition = params[0].split('|')
    #         variable = variable.strip()
    #         condition = condition.strip()
    #         print(f"Conditional Probability Detected")
    #         print(f"Variable: {variable}")
    #         print(f"Condition: {condition}")
    #     else:
    #         variable = params[0].strip()
    #         print(f"Variable: {variable} (no condition)")

    #     try:
    #         print("Available variables:", self.variables.keys())
    #         print(f"Checking if '{variable}' in self.variables")
            
    #         data = None
    #         if variable in self.variables:
    #             data = self.variables[variable]
    #             print(f"[DEBUG] Found variable '{variable}' in self.variables")
    #         else:
    #             data = self._resolve_data_reference(variable)
    #             if data is not None:
    #                 print(f"[DEBUG] Resolved variable '{variable}' using _resolve_data_reference")
            
    #         # if data is None or not isinstance(data, np.ndarray):
    #         #     return {'type': 'error', 'message': f'Cannot compute probability for {variable}'}

    #         print(f"Data for variable '{variable}': {data}")
    #         print(f"Data type: {type(data)}")

    #         if condition:
    #             # This is just a placeholder logic, update with real condition parsing if needed
    #             result = float(np.mean(data > 0))
    #             print(f"Conditional result (placeholder): {result}")
    #             return {
    #                 'type': 'query_result',
    #                 'query_type': 'probability',
    #                 'variable': variable,
    #                 'condition': condition,
    #                 'result': result,
    #             }
    #         else:
    #             result = float(np.mean(data > 0))
    #             print(f"Unconditional result: {result}")
    #             return {
    #                 'type': 'query_result',
    #                 'query_type': 'probability',
    #                 'variable': variable,
    #                 'result': result,
    #                 'visualization': self._generate_probability_visualization(data, variable)
    #             }

    #     except Exception as e:
    #         import traceback
    #         traceback.print_exc()
    #         return {'type': 'error', 'message': f'Error in probability query: {str(e)}'}
    def _execute_probability_query(self, params):
        """Execute a probability query P(X > value) or conditional P(A | B)"""
        print(f"[DEBUG] Received params: {params}")
        
        if not params:
            return {'type': 'error', 'message': 'Invalid probability query'}
        
        if isinstance(params, list):
            params = params[0]
        params = params.strip()

        # Check for conditional probability (A | B)
        if '|' in params:
            print("[DEBUG] Detected conditional probability")
            target_expr, cond_expr = map(str.strip, params.split('|', 1))
        else:
            target_expr = params
            cond_expr = None

        def parse_expression(expr):
            """Parses expressions like 'user.age > 20' into variable, op, value"""
            for op in ['>=', '<=', '==', '>', '<']:
                if op in expr:
                    var, val = map(str.strip, expr.split(op))
                    try:
                        val = float(val)
                    except ValueError:
                        return None, None, None
                    return var, op, val
            return expr.strip(), None, None  # No operator

        def resolve_data(var):
            """Tries to resolve data either directly or through reference"""
            if var in self.variables:
                return self.variables[var]
            else:
                return self._resolve_data_reference(var)

        # Parse target expression
        target_var, target_op, target_val = parse_expression(target_expr)
        target_data = resolve_data(target_var)
        if target_data is None:
            return {'type': 'error', 'message': f"Cannot resolve variable '{target_var}'"}

        target_data = np.array(target_data)

        # If no condition, just return histogram or single prob
        if not cond_expr:
            if target_op is None:
                return {
                    'type': 'query_result',
                    'query_type': 'distribution',
                    'variable': target_var,
                    'visualization': self._generate_histogram(target_data, target_var)
                }
            else:
                if target_op == '>':
                    prob = np.mean(target_data > target_val)
                elif target_op == '<':
                    prob = np.mean(target_data < target_val)
                elif target_op == '==':
                    prob = np.mean(target_data == target_val)
                elif target_op == '>=':
                    prob = np.mean(target_data >= target_val)
                elif target_op == '<=':
                    prob = np.mean(target_data <= target_val)
                else:
                    return {'type': 'error', 'message': f"Unsupported operator {target_op}"}

                return {
                    'type': 'query_result',
                    'query_type': 'probability',
                    'variable': target_expr,
                    'condition': None,
                    'result': float(prob),
                    'visualization': self._generate_probability_visualization(target_data, target_var, target_op, target_val, prob)
                }

        # Parse condition expression
        cond_var, cond_op, cond_val = parse_expression(cond_expr)
        cond_data = resolve_data(cond_var)
        if cond_data is None:
            return {'type': 'error', 'message': f"Cannot resolve condition variable '{cond_var}'"}

        cond_data = np.array(cond_data)

        # Mask where condition is True
        if cond_op == '>':
            cond_mask = cond_data > cond_val
        elif cond_op == '<':
            cond_mask = cond_data < cond_val
        elif cond_op == '==':
            cond_mask = cond_data == cond_val
        elif cond_op == '>=':
            cond_mask = cond_data >= cond_val
        elif cond_op == '<=':
            cond_mask = cond_data <= cond_val
        else:
            return {'type': 'error', 'message': f"Unsupported condition operator {cond_op}"}

        if target_op is None:
            prob = float(np.mean(cond_mask))
        else:
            # Apply same mask to target_data
            if len(target_data) != len(cond_mask):
                return {'type': 'error', 'message': f"Data length mismatch between '{target_var}' and '{cond_var}'"}
            filtered_target = target_data[cond_mask]
            if len(filtered_target) == 0:
                return {'type': 'error', 'message': 'No data satisfies the condition'}
            if target_op == '>':
                prob = np.mean(filtered_target > target_val)
            elif target_op == '<':
                prob = np.mean(filtered_target < target_val)
            elif target_op == '==':
                prob = np.mean(filtered_target == target_val)
            elif target_op == '>=':
                prob = np.mean(filtered_target >= target_val)
            elif target_op == '<=':
                prob = np.mean(filtered_target <= target_val)
            else:
                return {'type': 'error', 'message': f"Unsupported target operator {target_op}"}

        def is_variable(s):
            return all(op not in s for op in ['>', '<', '==', '!=', '>=', '<='])
        
        return {
            'type': 'query_result',
            'query_type': 'probability',
            'variable': f"{target_expr} | {cond_expr}",
            'condition': f"{cond_expr}",
            'result': float(prob),
            'visualization': self._generate_conditional_probability_visualization(target_data, cond_data, target_var, target_op, target_val, cond_var, cond_op, cond_val, prob)
        }

    def _generate_conditional_probability_visualization(
        self,
        target_data, condition_data,
        target_variable, target_operator, target_value,
        condition_variable, condition_operator, condition_value,
        probability
    ):
        """2D scatter visualization showing only 'both' vs 'not both'"""
        plt.figure(figsize=(8, 6))

        if len(target_data) == 0 or len(condition_data) == 0:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.title(f"P({target_variable} {target_operator} {target_value} | "
                    f"{condition_variable} {condition_operator} {condition_value})")
        else:
            def apply_op(data, op, val):
                return {
                    '>': data > val,
                    '>=': data >= val,
                    '<': data < val,
                    '<=': data <= val,
                    '==': data == val,
                }.get(op, np.full_like(data, False, dtype=bool))

            condition_mask = apply_op(condition_data, condition_operator, condition_value)
            target_mask = apply_op(target_data, target_operator, target_value)
            both_mask = condition_mask & target_mask

            # Split data into "both" and "not both"
            not_both_mask = ~both_mask

            # Plot
            plt.scatter(condition_data[not_both_mask], target_data[not_both_mask],
                        color='black', alpha=0.4, label='Not satisfied')
            plt.scatter(condition_data[both_mask], target_data[both_mask],
                        color='red', alpha=0.8, label='Satisfy Both')

            plt.xlabel(condition_variable)
            plt.ylabel(target_variable)
            plt.title(f"P({target_variable} {target_operator} {target_value} | "
                    f"{condition_variable} {condition_operator} {condition_value}) = {probability:.4f}")
            plt.legend()
            plt.grid(alpha=0.3)

        # Save and encode
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()

        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        return base64.b64encode(image_png).decode('utf-8')


    def _generate_probability_visualization(self, data, variable, operator, target_value, probability):
        """Generate visualization for a simple probability query like P(variable > target_value)"""
        plt.figure(figsize=(8, 5))
        
        if len(data) == 0:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.title(f"Probability of {variable} {operator} {target_value}")
        else:
            # Histogram
            bins = min(30, len(np.unique(data)))
            counts, bin_edges, _ = plt.hist(data, bins=bins, alpha=0.5, color='skyblue', density=True)

            # Highlight area satisfying the condition
            condition_mask = None
            if operator == '>':
                condition_mask = bin_edges[:-1] > target_value
            elif operator == '<':
                condition_mask = bin_edges[1:] < target_value
            elif operator == '>=':
                condition_mask = bin_edges[:-1] >= target_value
            elif operator == '<=':
                condition_mask = bin_edges[1:] <= target_value
            elif operator == '==':
                condition_mask = (bin_edges[:-1] <= target_value) & (bin_edges[1:] >= target_value)
            else:
                condition_mask = np.full_like(bin_edges[:-1], False, dtype=bool)

            for i in range(len(condition_mask)):
                if condition_mask[i]:
                    plt.fill_between([bin_edges[i], bin_edges[i+1]], 0, counts[i],
                                    color='orange', alpha=0.6)

            # Draw threshold line
            plt.axvline(x=target_value, color='r', linestyle='--', 
                        label=f"{variable} {operator} {target_value}")

            # Label and title
            plt.title(f"Distribution of {variable} with P({variable} {operator} {target_value}) = {probability:.4f}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
        
        plt.grid(alpha=0.3)

        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')

    # def _generate_conditional_probability_visualization(
    #     self,
    #     target_data, condition_data,
    #     target_variable, target_operator, target_value,
    #     condition_variable, condition_operator, condition_value,
    #     probability
    # ):
    #     """Visualize P(target | condition) by showing how both variables overlap"""
    #     plt.figure(figsize=(8, 5))

    #     if len(target_data) == 0 or len(condition_data) == 0:
    #         plt.text(0.5, 0.5, "No data available", ha='center', va='center')
    #         plt.title(f"P({target_variable} {target_operator} {target_value} | "
    #                 f"{condition_variable} {condition_operator} {condition_value})")
    #     else:
    #         # We'll show the distribution of the *condition* variable
    #         bins = min(30, len(np.unique(condition_data)))
    #         counts, bin_edges, _ = plt.hist(condition_data, bins=bins, alpha=0.5, color='skyblue', density=True)

    #         # Mark where condition is satisfied
    #         cond_mask = None
    #         if condition_operator == '>':
    #             cond_mask = bin_edges[:-1] > condition_value
    #         elif condition_operator == '<':
    #             cond_mask = bin_edges[1:] < condition_value
    #         elif condition_operator == '>=':
    #             cond_mask = bin_edges[:-1] >= condition_value
    #         elif condition_operator == '<=':
    #             cond_mask = bin_edges[1:] <= condition_value
    #         elif condition_operator == '==':
    #             cond_mask = (bin_edges[:-1] <= condition_value) & (bin_edges[1:] >= condition_value)
    #         else:
    #             cond_mask = np.full_like(bin_edges[:-1], False, dtype=bool)

    #         for i in range(len(cond_mask)):
    #             if cond_mask[i]:
    #                 plt.fill_between([bin_edges[i], bin_edges[i+1]], 0, counts[i],
    #                                 color='orange', alpha=0.5)

    #         # Now highlight where BOTH condition and target are satisfied
    #         # This isn't shown directly in histogram — we annotate it as conditional
    #         plt.axvline(x=condition_value, color='r', linestyle='--',
    #                     label=f"{condition_variable} {condition_operator} {condition_value}")

    #         # Title and label
    #         plt.title(f"P({target_variable} {target_operator} {target_value} | "
    #                 f"{condition_variable} {condition_operator} {condition_value}) = {probability:.4f}")
    #         plt.xlabel(f"{condition_variable} distribution")
    #         plt.ylabel("Density")
    #         plt.legend()

    #     plt.grid(alpha=0.3)

    #     buffer = io.BytesIO()
    #     plt.savefig(buffer, format='png')
    #     plt.close()
        
    #     buffer.seek(0)
    #     image_png = buffer.getvalue()
    #     buffer.close()
        
    #     return base64.b64encode(image_png).decode('utf-8')

    def _execute_correlation_query(self, params):
        try:
            if len(params) != 2:
                return {'type': 'error', 'message': 'Correlation requires exactly 2 parameters'}
            
            # Extract the parameters
            expr1 = params[0].strip()
            expr2 = params[1].strip()
            
            # Try to resolve from variables first
            data1 = self.variables.get(expr1, None)
            data2 = self.variables.get(expr2, None)
            
            # If not found in variables, try to resolve as data references
            if data1 is None:
                data1 = self._resolve_data_reference(expr1)
            if data2 is None:
                data2 = self._resolve_data_reference(expr2)
                
            # If still not found, return error
            if data1 is None or data2 is None:
                return {'type': 'error', 'message': f'Could not resolve data references: {expr1}, {expr2}'}
                
            # Calculate correlation
            corr = np.corrcoef(data1, data2)[0, 1]
            
            return {
                'type': 'query_result',
                'query_type': 'correlation',
                'variables': [expr1, expr2],
                'result': float(corr),
                'visualization': self._generate_correlation_visualization(data1, data2, expr1, expr2, corr)
            }
        except Exception as e:
            return {'type': 'error', 'message': f'Error in correlation query: {str(e)}'}

    def _generate_correlation_visualization(self, data1, data2, var1, var2, corr):
        """Generate a scatter plot for correlation visualization."""
        plt.figure(figsize=(8, 5))
        
        # Plot scatter
        plt.scatter(data1, data2, alpha=0.6, edgecolor='k', color='skyblue')
        plt.title(f"Correlation between {var1} and {var2}")
        plt.xlabel(var1)
        plt.ylabel(var2)

        # Fit and plot regression line (optional but nice)
        if len(data1) > 1:
            m, b = np.polyfit(data1, data2, 1)
            plt.plot(data1, m * np.array(data1) + b, color='red', linestyle='--', label=f'Corr = {corr:.4f}')
            plt.legend()

        plt.grid(alpha=0.3)

        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()

        # Convert to base64
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        return base64.b64encode(image_png).decode('utf-8')

    # def _generate_probability_visualization(self, data, variable):
    #     """Generate visualization for probability query"""
    #     plt.figure(figsize=(8, 5))
        
    #     # For binary data
    #     if set(np.unique(data)) <= {0, 1}:
    #         counts = np.bincount(data.astype(int))
    #         plt.bar(['0', '1'], counts/len(data), color=['lightcoral', 'lightblue'])
    #         plt.title(f"Probability Distribution of {variable}")
    #         plt.ylabel("Probability")
            
    #     # For continuous data
    #     else:
    #         kde = stats.gaussian_kde(data)
    #         x = np.linspace(min(data), max(data), 1000)
    #         plt.plot(x, kde(x), 'b-')
    #         plt.fill_between(x, kde(x), alpha=0.3)
    #         plt.axvline(x=0, color='r', linestyle='--')
    #         pos_prob = np.mean(data > 0)
    #         plt.title(f"Density of {variable}, P({variable} > 0) = {pos_prob:.4f}")
    #         plt.xlabel("Value")
    #         plt.ylabel("Density")
            
    #     plt.grid(alpha=0.3)
        
    #     # Save to bytes buffer
    #     buffer = io.BytesIO()
    #     plt.savefig(buffer, format='png')
    #     plt.close()
        
    #     # Convert to base64
    #     buffer.seek(0)
    #     image_png = buffer.getvalue()
    #     buffer.close()
        
    #     return base64.b64encode(image_png).decode('utf-8')
    
    def _execute_expectation_query(self, params):
        """Execute an expectation query E(X)"""
        if not params:
            return {'type': 'error', 'message': 'Invalid expectation query'}
            
        # Handle conditional expectation
        condition = None
        if '|' in params:
            variable, condition = params.split('|', 1)
            variable = variable.strip()
            condition = condition.strip()
        else:
            variable = params.strip()
        
        print(f"Debug: Processing expectation for variable '{variable}', condition: {condition}")
        
        # Calculate expectation
        try:
            # Try to resolve from variables first
            data = None
            if variable in self.variables:
                print(f"Debug: Found variable '{variable}' in variables")
                data = self.variables[variable]
            else:
                # Try to resolve as data reference
                print(f"Debug: Trying to resolve '{variable}' as data reference")
                data = self._resolve_data_reference(variable)
                print(f"Debug: Data reference resolution result: {type(data)}")
            
            if data is None or not isinstance(data, np.ndarray):
                return {'type': 'error', 'message': f'Cannot compute expectation for {variable}'}
            
            if condition:
                # Parse and evaluate the condition
                try:
                    # For simple conditions like "Y > 0"
                    if '>' in condition:
                        cond_parts = condition.split('>')
                        cond_var = cond_parts[0].strip()
                        cond_val = float(cond_parts[1].strip())
                    
                        # Get the conditional variable data
                        cond_data = None
                        if cond_var in self.variables:
                            cond_data = self.variables[cond_var]
                        else:
                            cond_data = self._resolve_data_reference(cond_var)
                    
                        if cond_data is None:
                            return {'type': 'error', 'message': f'Cannot resolve condition variable {cond_var}'}
                    
                        # Create mask for condition
                        mask = cond_data > cond_val
                        if np.sum(mask) == 0:
                            return {'type': 'error', 'message': 'No data points satisfy the condition'}
                    
                        # Calculate conditional expectation
                        conditional_mean = np.mean(data[mask])
                    
                        return {
                            'type': 'query_result',
                            'query_type': 'expectation',
                            'variable': variable,
                            'condition': condition,
                            'result': float(conditional_mean),
                            'visualization': self._generate_conditional_expectation_visualization(
                                data, cond_data, mask, variable, condition, conditional_mean
                            )
                        }
                    
                    # Handle other conditions (==, <)
                    else:
                        return {'type': 'error', 'message': f'Unsupported condition format: {condition}'}
                except Exception as e:
                    return {'type': 'error', 'message': f'Error evaluating condition: {str(e)}'}
            else:
                # Unconditional expectation
                return {
                    'type': 'query_result',
                    'query_type': 'expectation',
                    'variable': variable,
                    'result': float(np.mean(data)),
                    'visualization': self._generate_expectation_visualization(data, variable)
                }
        except Exception as e:
            return {'type': 'error', 'message': f'Error in expectation query: {str(e)}'}
    
    def _generate_expectation_visualization(self, data, variable):
        """Generate visualization for expectation query"""
        plt.figure(figsize=(8, 5))
        
        # Handle potential empty or constant data
        if len(data) == 0:
            plt.text(0.5, 0.5, "No data available", ha='center', va='center')
            plt.title(f"Distribution and Expectation of {variable}")
        else:
            plt.hist(data, bins=min(30, len(np.unique(data))), alpha=0.7, color='skyblue', density=True)
            mean_val = np.mean(data)
            plt.axvline(x=mean_val, color='r', linestyle='--', 
                       label=f'E({variable}) = {mean_val:.4f}')
            plt.legend()
            plt.title(f"Distribution and Expectation of {variable}")
            plt.xlabel("Value")
            plt.ylabel("Density")
        
        plt.grid(alpha=0.3)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def _execute_outliers_query(self, params):
        """Execute an outliers query"""
        if not params:
            return {'type': 'error', 'message': 'Invalid outliers query'}
            
        var_name = params[0].strip()
        
        try:
            # Try to resolve from variables first
            data = None
            if var_name in self.variables:
                data = self.variables[var_name]
            else:
                # Try to resolve as data reference
                data = self._resolve_data_reference(var_name)
                
            # If still not found, return error
            if data is None:
                return {'type': 'error', 'message': f'Could not resolve data reference: {var_name}'}
                
            # Simple Z-score based outlier detection with error handling
            try:
                z_scores = np.abs(stats.zscore(data))
                outliers = np.where(z_scores > 2.5)[0]
            except:
                # Fallback for cases where Z-scores can't be calculated
                # (e.g., constant data)
                mean = np.mean(data)
                std = np.std(data)
                if std == 0:
                    outliers = np.array([])
                else:
                    outliers = np.where(np.abs(data - mean) > 2.5 * std)[0]
                    
            return {
                'type': 'query_result',
                'query_type': 'outliers',
                'variable': var_name,
                'outlier_indices': outliers.tolist(),
                'outlier_count': len(outliers),
                'visualization': self._generate_outliers_visualization(data, outliers, var_name)
            }
            
        except Exception as e:
            return {'type': 'error', 'message': f'Error in outliers query: {str(e)}'}
    
    def _generate_outliers_visualization(self, data, outlier_indices, variable):
        """Generate visualization for outliers query"""
        plt.figure(figsize=(8, 5))
        
        # Plot all points
        plt.scatter(range(len(data)), data, alpha=0.6, label='Normal')
        
        # Highlight outliers
        if len(outlier_indices) > 0:
            plt.scatter(outlier_indices, data[outlier_indices], color='red', label='Outliers')
            
        plt.title(f"Outliers Detection for {variable}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def _execute_clustering(self, statement):
        """Execute a clustering statement"""
        params = self._extract_params(statement)
        if len(params) < 2:
            return {'type': 'error', 'message': 'Invalid clustering statement'}
            
        dataset_name = params[0].strip()
        
        # Parse clustering options
        options = {}
        for param in params[1:]:
            if ':' in param:
                key, value = param.split(':', 1)
                options[key.strip()] = value.strip()
                
        # Extract dimensions and k
        dimensions = []
        if 'dimensions' in options:
            dim_str = options['dimensions']
            dim_str = dim_str.strip('[]')
            dimensions = [d.strip() for d in dim_str.split(',')]
            
        k = 3  # Default
        if 'k' in options:
            k = int(options['k'])
            
        # Perform clustering
        try:
            if dataset_name in self.data:
                data = self.data[dataset_name]
                
                if dimensions and all(dim in data.columns for dim in dimensions):
                    X = data[dimensions].values
                    
                    # Apply KMeans
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    clusters = kmeans.fit_predict(X)
                    
                    # Add cluster column to dataset
                    data['cluster'] = clusters
                    
                    # Evaluate clustering
                    silhouette = silhouette_score(X, clusters) if len(np.unique(clusters)) > 1 else 0
                    
                    return {
                        'type': 'clustering_result',
                        'dataset': dataset_name,
                        'dimensions': dimensions,
                        'k': k,
                        'silhouette_score': float(silhouette),
                        'cluster_sizes': [int(np.sum(clusters == i)) for i in range(k)],
                        'visualization': self._generate_clustering_visualization(X, clusters, dimensions)
                    }
            
            return {'type': 'error', 'message': f'Cannot perform clustering on {dataset_name}'}
            
        except Exception as e:
            return {'type': 'error', 'message': f'Error in clustering: {str(e)}'}
    
    def _generate_clustering_visualization(self, X, clusters, dimensions):
        """Generate visualization for clustering results"""
        plt.figure(figsize=(8, 5))
        
        # If 2D data, plot scatter
        if X.shape[1] >= 2:
            plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            if len(dimensions) >= 2:
                plt.xlabel(dimensions[0])
                plt.ylabel(dimensions[1])
            plt.title(f"K-Means Clustering Result (k={len(np.unique(clusters))})")
            plt.colorbar(label='Cluster')
        else:
            # For 1D data
            for i in np.unique(clusters):
                plt.hist(X[clusters == i, 0], alpha=0.5, label=f'Cluster {i}')
            plt.xlabel(dimensions[0] if dimensions else 'Feature')
            plt.title("1D Clustering")
            plt.legend()
            
        plt.grid(alpha=0.3)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def _execute_association(self, statement):
        """Execute an association rule mining statement"""
        params = self._extract_params(statement)
        if not params:
            return {'type': 'error', 'message': 'Invalid association rule mining statement'}
            
        dataset_name = params[0].strip()
        
        # Parse options
        options = {}
        for param in params[1:]:
            if ':' in param:
                key, value = param.split(':', 1)
                options[key.strip()] = value.strip()
                
        min_support = float(options.get('min_support', '0.1'))
        min_confidence = float(options.get('min_confidence', '0.5'))
        
        try:
            if dataset_name in self.data:
                data = self.data[dataset_name]
                
                # For simplicity, binarize categorical columns
                # In a real implementation, this would be more sophisticated
                categorical_cols = data.select_dtypes(include=['object']).columns
                
                # Mock association rules results for demonstration
                return {
                    'type': 'association_result',
                    'dataset': dataset_name,
                    'min_support': min_support,
                    'min_confidence': min_confidence,
                    'rules_count': 5,
                    'rules': [
                        {"antecedent": "A", "consequent": "B", "support": 0.3, "confidence": 0.75},
                        {"antecedent": "B", "consequent": "C", "support": 0.25, "confidence": 0.8},
                        {"antecedent": "A,D", "consequent": "C", "support": 0.15, "confidence": 0.65}
                    ]
                }
            
            return {'type': 'error', 'message': f'Cannot find associations in {dataset_name}'}
            
        except Exception as e:
            return {'type': 'error', 'message': f'Error in association rules: {str(e)}'}
    
    def _execute_classification(self, statement):
        """Execute a classification statement"""
        params = self._extract_params(statement)
        if len(params) < 2:
            return {'type': 'error', 'message': 'Invalid classification statement'}
            
        dataset_name = params[0].strip()
        
        # Parse options
        options = {}
        target = None
        for param in params[1:]:
            if ':' in param:
                key, value = param.split(':', 1)
                if key.strip() == 'target':
                    target = value.strip()
                else:
                    options[key.strip()] = value.strip()
        
        if not target:
            return {'type': 'error', 'message': 'Classification requires a target variable'}
            
        # Get classifier type
        classifier_type = options.get('classifier', 'decision_tree')
        
        try:
            if dataset_name in self.data:
                data = self.data[dataset_name]
                
                if target in data.columns:
                    # Mock classification results for demonstration
                    return {
                        'type': 'classification_result',
                        'dataset': dataset_name,
                        'target': target,
                        'classifier': classifier_type,
                        'accuracy': 0.85,
                        'feature_importance': [
                            {"feature": "feature1", "importance": 0.4},
                            {"feature": "feature2", "importance": 0.3},
                            {"feature": "feature3", "importance": 0.2}
                        ],
                        'confusion_matrix': [[45, 5], [10, 40]]
                    }
            
            return {'type': 'error', 'message': f'Cannot perform classification on {dataset_name}'}
            
        except Exception as e:
            return {'type': 'error', 'message': f'Error in classification: {str(e)}'}

    def _generate_conditional_expectation_visualization(self, data, cond_data, mask, variable, condition, conditional_mean):
        """Generate visualization for conditional expectation query"""
        plt.figure(figsize=(8, 5))
        
        # Scatter plot with condition highlighted
        plt.scatter(cond_data[~mask], data[~mask], alpha=0.3, color='gray', label='Not in condition')
        plt.scatter(cond_data[mask], data[mask], alpha=0.6, color='blue', label='Satisfies condition')
        
        # Add horizontal line for conditional expectation
        plt.axhline(y=conditional_mean, color='r', linestyle='--', 
                   label=f'E({variable}|{condition}) = {conditional_mean:.4f}')
        
        plt.legend()
        plt.title(f"Conditional Expectation of {variable} given {condition}")
        plt.xlabel(condition.split()[0])  # Assuming condition is like "Y > 0"
        plt.ylabel(variable)
        plt.grid(alpha=0.3)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        
        # Convert to base64
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(image_png).decode('utf-8')