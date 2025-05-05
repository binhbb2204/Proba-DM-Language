import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    """Class for creating visualizations based on query results"""
    
    @staticmethod
    def plot_probability_distribution(fig, prob_result):
        """Plot probability distribution"""
        ax = fig.add_subplot(111)
        
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
    
    @staticmethod
    def plot_expected_values(fig, exp_results):
        """Plot expected values"""
        ax = fig.add_subplot(111)
        
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
    
    @staticmethod
    def plot_clusters(fig, clustering_result):
        """Plot clustering visualization"""
        ax = fig.add_subplot(111, projection='3d')
        
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
    
    @staticmethod
    def plot_correlation(fig, corr_result):
        """Plot correlation visualization"""
        ax = fig.add_subplot(111)
        ax.bar([f"{corr_result['expr1']} vs {corr_result['expr2']}"], [corr_result['value']],
               color='skyblue')
        ax.set_ylim(-1, 1)
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Correlation Query Result')
        ax.grid(True, axis='y')
    
    @staticmethod
    def plot_outliers(fig, outlier_result):
        """Plot outliers visualization"""
        ax = fig.add_subplot(111)
        values = outlier_result['values']
        ax.scatter(range(len(values)), values, c='red', marker='x')
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(outlier_result['expressions'], rotation=45)
        ax.set_ylabel('Value')
        ax.set_title('Outliers Query Result')
        ax.grid(True)
    
    @staticmethod
    def plot_association_rules(fig, assoc_result):
        """Plot association rules visualization"""
        ax = fig.add_subplot(111)
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
    
    @staticmethod
    def plot_classification_results(fig, class_result):
        """Plot classification results visualization"""
        ax = fig.add_subplot(111)
        ax.bar(['Accuracy'], [class_result['accuracy']], color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title(f"Classification Results for {class_result['target']}")
        ax.grid(True, axis='y')
    
    @staticmethod
    def plot_default_distribution(fig):
        """Plot default distribution"""
        ax = fig.add_subplot(111)
        
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
        
    @staticmethod
    def create_visualization(fig, results):
        """Generate visualizations based on query results"""
        fig.clear()
        
        prob_queries = {k: v for k, v in results.items() if k.startswith('P_')}
        exp_queries = {k: v for k, v in results.items() if k.startswith('E_')}
        corr_queries = {k: v for k, v in results.items() if k.startswith('corr_')}
        outlier_queries = {k: v for k, v in results.items() if k.startswith('outliers_')}
        
        if 'clustering' in results:
            Visualizer.plot_clusters(fig, results['clustering'])
        elif 'association_rules' in results:
            Visualizer.plot_association_rules(fig, results['association_rules'])
        elif 'classification' in results:
            Visualizer.plot_classification_results(fig, results['classification'])
        elif prob_queries:
            Visualizer.plot_probability_distribution(fig, list(prob_queries.values())[0])
        elif exp_queries:
            Visualizer.plot_expected_values(fig, list(exp_queries.values()))
        elif corr_queries:
            Visualizer.plot_correlation(fig, list(corr_queries.values())[0])
        elif outlier_queries:
            Visualizer.plot_outliers(fig, list(outlier_queries.values())[0])
        else:
            Visualizer.plot_default_distribution(fig)