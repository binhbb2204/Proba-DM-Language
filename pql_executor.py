import numpy as np
import pandas as pd
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
            
            # Process the code line by line
            lines = code.strip().split('\n')
            results = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('//'):
                    continue
                
                if line.startswith('load_data'):
                    result = self._execute_load_data(line)
                    results.append(result)
                    
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
                    
                # Add other statement types as needed
            
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
        """Resolve data reference in the form 'data.column' or 'dataset_name.column'"""
        if not data_ref or '.' not in data_ref:
            return None
            
        parts = data_ref.split('.')
        if len(parts) != 2:
            return None
            
        dataset_name, column = parts
        
        # Handle the case when the dataset is referenced as 'data'
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
                return self._execute_probability_query(params)
                
            elif query_part.startswith('E('):
                # Expected value query
                params = query_part[2:-1].strip()
                return self._execute_expectation_query(params)
                
            elif query_part.startswith('correlation('):
                # Correlation query
                params = query_part[12:-1].strip().split(',')
                return self._execute_correlation_query(params)
                
            elif query_part.startswith('outliers('):
                # Outliers query
                params = query_part[9:-1].strip().split(',')
                return self._execute_outliers_query(params)
                
            else:
                return {'type': 'error', 'message': f'Unknown query type: {query_part}'}
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'type': 'error', 'message': f'Error executing query: {str(e)}'}
    
    def _execute_probability_query(self, params):
        """Execute a probability query P(X > 0)"""
        if not params:
            return {'type': 'error', 'message': 'Invalid probability query'}
            
        # Handle conditional probability
        condition = None
        if len(params) > 1 and '|' in params[0]:
            # Handle conditional probability
            variable, condition = params[0].split('|')
            variable = variable.strip()
            condition = condition.strip()
        else:
            variable = params[0]
            
        # Calculate probability
        try:
            if variable in self.variables:
                data = self.variables[variable]
                if isinstance(data, np.ndarray):
                    if condition:
                        # Simplified conditional probability
                        return {
                            'type': 'query_result',
                            'query_type': 'probability',
                            'variable': variable,
                            'condition': condition,
                            'result': float(np.mean(data > 0))  # Placeholder
                        }
                    else:
                        return {
                            'type': 'query_result',
                            'query_type': 'probability',
                            'variable': variable,
                            'result': float(np.mean(data > 0)),
                            'visualization': self._generate_probability_visualization(data, variable)
                        }
            
            return {'type': 'error', 'message': f'Cannot compute probability for {variable}'}
            
        except Exception as e:
            return {'type': 'error', 'message': f'Error in probability query: {str(e)}'}
        
    def _execute_correlation_query(self, params):
        """Execute a correlation query"""
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
                return {
                    'type': 'error', 
                    'message': f'Could not resolve data references: {expr1}, {expr2}'
                }
                
            # Calculate correlation
            import numpy as np
            corr = np.corrcoef(data1, data2)[0, 1]
            
            return {
                'type': 'correlation',
                'expr1': expr1,
                'expr2': expr2,
                'value': float(corr)
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'type': 'error', 'message': f'Error in correlation query: {str(e)}'}
    
    def _generate_probability_visualization(self, data, variable):
        """Generate visualization for probability query"""
        plt.figure(figsize=(8, 5))
        
        # For binary data
        if set(np.unique(data)) <= {0, 1}:
            counts = np.bincount(data.astype(int))
            plt.bar(['0', '1'], counts/len(data), color=['lightcoral', 'lightblue'])
            plt.title(f"Probability Distribution of {variable}")
            plt.ylabel("Probability")
            
        # For continuous data
        else:
            kde = stats.gaussian_kde(data)
            x = np.linspace(min(data), max(data), 1000)
            plt.plot(x, kde(x), 'b-')
            plt.fill_between(x, kde(x), alpha=0.3)
            plt.axvline(x=0, color='r', linestyle='--')
            pos_prob = np.mean(data > 0)
            plt.title(f"Density of {variable}, P({variable} > 0) = {pos_prob:.4f}")
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
    
    def _execute_expectation_query(self, params):
        """Execute an expectation query E(X)"""
        if not params:
            return {'type': 'error', 'message': 'Invalid expectation query'}
            
        # Handle conditional expectation
        condition = None
        if len(params) > 1 and '|' in params[0]:
            # Handle conditional expectation
            variable, condition = params[0].split('|')
            variable = variable.strip()
            condition = condition.strip()
        else:
            variable = params[0]
            
        # Calculate expectation
        try:
            if variable in self.variables:
                data = self.variables[variable]
                if isinstance(data, np.ndarray):
                    if condition:
                        # Simplified conditional expectation
                        return {
                            'type': 'query_result',
                            'query_type': 'expectation',
                            'variable': variable,
                            'condition': condition,
                            'result': float(np.mean(data))  # Placeholder
                        }
                    else:
                        return {
                            'type': 'query_result',
                            'query_type': 'expectation',
                            'variable': variable,
                            'result': float(np.mean(data)),
                            'visualization': self._generate_expectation_visualization(data, variable)
                        }
            
            return {'type': 'error', 'message': f'Cannot compute expectation for {variable}'}
            
        except Exception as e:
            return {'type': 'error', 'message': f'Error in expectation query: {str(e)}'}
    
    def _generate_expectation_visualization(self, data, variable):
        """Generate visualization for expectation query"""
        plt.figure(figsize=(8, 5))
        
        plt.hist(data, bins=30, alpha=0.7, color='skyblue', density=True)
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
    
    def _execute_correlation_query(self, params):
        """Execute a correlation query"""
        if len(params) != 2:
            return {'type': 'error', 'message': 'Correlation query requires two variables'}
            
        var1 = params[0].strip()
        var2 = params[1].strip()
        
        try:
            if var1 in self.variables and var2 in self.variables:
                data1 = self.variables[var1]
                data2 = self.variables[var2]
                
                if len(data1) == len(data2):
                    corr_coef = np.corrcoef(data1, data2)[0, 1]
                    
                    return {
                        'type': 'query_result',
                        'query_type': 'correlation',
                        'variables': [var1, var2],
                        'result': float(corr_coef),
                        'visualization': self._generate_correlation_visualization(data1, data2, var1, var2, corr_coef)
                    }
            
            return {'type': 'error', 'message': f'Cannot compute correlation between {var1} and {var2}'}
            
        except Exception as e:
            return {'type': 'error', 'message': f'Error in correlation query: {str(e)}'}
    
    def _generate_correlation_visualization(self, data1, data2, var1, var2, corr):
        """Generate scatter plot for correlation"""
        plt.figure(figsize=(8, 5))
        
        plt.scatter(data1, data2, alpha=0.6)
        plt.title(f"Correlation between {var1} and {var2}: {corr:.4f}")
        plt.xlabel(var1)
        plt.ylabel(var2)
        
        # Add regression line
        m, b = np.polyfit(data1, data2, 1)
        x_line = np.linspace(min(data1), max(data1), 100)
        plt.plot(x_line, m * x_line + b, 'r--')
        
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
            if var_name in self.variables:
                data = self.variables[var_name]
                
                # Simple Z-score based outlier detection
                z_scores = np.abs(stats.zscore(data))
                outliers = np.where(z_scores > 2.5)[0]
                
                return {
                    'type': 'query_result',
                    'query_type': 'outliers',
                    'variable': var_name,
                    'outlier_indices': outliers.tolist(),
                    'outlier_count': len(outliers),
                    'visualization': self._generate_outliers_visualization(data, outliers, var_name)
                }
            
            return {'type': 'error', 'message': f'Cannot find outliers for {var_name}'}
            
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
