"""
Functions for plotting fine-tuning results
"""

# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import Modules
from src.utils import get_project_root

def plot_in_out_domain(logfile, metric):
    """Plot in domain vs. out of domain metrics from a log file"""
    # Read log file
    logfilepath = os.path.join(get_project_root(), 'logs', logfile)
    log_df = pd.read_csv(logfilepath)
    
    plt.figure(figsize=(6, 6))
    # plt.xlim(axes_min, axes_max)
    # plt.ylim(axes_min, axes_max)

    sample_sizes = log_df['sample_size'].unique()

    # Plot in-domain and out-of-domain metrics for each sample size
    for size in sample_sizes:
        subset = log_df[log_df['sample_size'] == size]
        
        avg_in_metric = subset[f'eval_in_{metric}'].mean()
        avg_out_metric = subset[f'eval_out_{metric}'].mean()

        plt.scatter(avg_in_metric, avg_out_metric, label=f'{size}-shot', marker='+', s=200)
    
    # Equalize axes and plot diagonal line
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    axes_range = (min(x_min, y_min), max(x_max, y_max))
    plt.xlim(axes_range)
    plt.ylim(axes_range)
    
    plt.plot(axes_range, axes_range, 'k--', alpha=0.2)
    
    num_trials = len(subset)
    model_name = log_df['model_name'][0]

    plt.title(f'In-Domain vs. Out-of-Domain {metric.capitalize()} ({num_trials} trials, {model_name})')
    plt.xlabel('In-Domain')
    plt.ylabel('Out-of-Domain')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_learning_curves(logfile):
    """Plot training and validation loss from a log file"""
    # Read log file
    logfilepath = os.path.join(get_project_root(), 'logs', logfile)
    log_df = pd.read_csv(logfilepath)
    
    plt.figure(figsize=(6, 6))

    sample_sizes = log_df['sample_size'].unique()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot average training and validation loss for each sample size
    for i, size in enumerate(sample_sizes):
        subset = log_df[log_df['sample_size'] == size]
        avg_train_loss = subset.groupby('epoch')['train_loss'].mean()
        avg_val_loss = subset.groupby('epoch')['val_loss'].mean()

        color = color_cycle[i % len(color_cycle)]
        plt.plot(avg_train_loss.index, avg_train_loss, label=f'{size}-shot Train Loss', linestyle='-', marker='o', markersize=4, color=color)
        plt.plot(avg_val_loss.index, avg_val_loss, label=f'{size}-shot Val Loss', linestyle='--', marker='x', markersize=4, color=color)
        
    num_trials = len(subset.groupby('epoch')['train_loss'])
    model_name = log_df['model_name'][0]
    
    plt.title(f'Learning Curves ({num_trials} trials, {model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()