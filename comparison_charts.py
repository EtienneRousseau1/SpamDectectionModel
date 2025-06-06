import matplotlib.pyplot as plt
import numpy as np

# Data for the models
models = ['BERT', 'Random Forest', 'Logistic Regression']

# Metrics data
accuracy_scores = [0.998, 0.914, 0.683]
f1_scores = [0.952, 0.913, 0.694]
roc_auc_scores = [None, 0.968, 0.747]
avg_precision_scores = [None, 0.973, 0.758]

def create_comparison_chart(metric_values, metric_name, output_filename):
    plt.figure(figsize=(10, 6))
    
    # Filter out None values and corresponding model names
    valid_data = [(model, value) for model, value in zip(models, metric_values) if value is not None]
    valid_models, valid_values = zip(*valid_data)
    
    bars = plt.bar(valid_models, valid_values)
    
    # Customize the chart
    plt.title(f'{metric_name} Comparison Across Models', fontsize=14, pad=20)
    plt.ylabel(metric_name, fontsize=12)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Customize grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'comparison_charts/{output_filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create directory for charts
import os
os.makedirs('comparison_charts', exist_ok=True)

# Create individual charts
create_comparison_chart(accuracy_scores, 'Accuracy', 'accuracy_comparison')
create_comparison_chart(f1_scores, 'F1 Score', 'f1_score_comparison')
create_comparison_chart(roc_auc_scores, 'ROC AUC', 'roc_auc_comparison')
create_comparison_chart(avg_precision_scores, 'Average Precision', 'avg_precision_comparison')
