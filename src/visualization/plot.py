"""
Functions for plotting fine-tuning results
"""

# Import libraries
import os
import pandas as pd
import numpy as np
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
        # for model_name in 
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
    finetuning_method = logfile.split('_')[0]

    plt.title(f'{metric.capitalize()} ({finetuning_method}, {num_trials} trials, {model_name})')
    plt.xlabel('In-Domain')
    plt.ylabel('Out-of-Domain')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_in_out_domain_subplots(logfiles, metrics=['accuracy', 'runtime', 'peak_memory_gb', 'loss'], group_by=None):
    """Plot in vs. out-of-domain data in separate subplots for a desired grouping"""
    
    #TODO: draw line for zeroshot baseline
    #TODO: draw diagonal line and square plots
    
    # Combine logfiles into single dataframe
    combined_df = pd.DataFrame()
    for logfile in logfiles:
        logfilepath = os.path.join(get_project_root(), 'logs', logfile)
        finetuning_method = logfile.split('_')[0]
        temp_df = pd.read_csv(logfilepath)
        temp_df['finetuning_method'] = finetuning_method
        combined_df = pd.concat([combined_df, temp_df])
    
    # Get members of grouping
    if group_by:
        unique_groups = combined_df[group_by].unique()
    else:
        unique_groups = ['All Data']
    
    num_cols = len(metrics)
    
    for group in unique_groups:
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
        axes = axes.flatten()

        for subplot, metric in enumerate(metrics):
            ax = axes[subplot]
            if group_by:
                group_df = combined_df[combined_df[group_by] == group]
            else:
                group_df = combined_df

            # Use pandas groupby to handle the complex grouping logic
            for (model_name, sample_size, finetuning_method), subset in group_df.groupby(['model_name', 'sample_size', 'finetuning_method']):
                avg_in_metric = subset[f'eval_in_{metric}'].mean()
                avg_out_metric = subset[f'eval_out_{metric}'].mean()

                label = f"{finetuning_method} ({sample_size}, {model_name})"
                ax.scatter(avg_in_metric, avg_out_metric, label=label, marker='+', s=200)

            ax.set_title(f'{metric.capitalize()} - {group}')
            ax.set_xlabel('In-Domain')
            ax.set_ylabel('Out-of-Domain')
            ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            fig.legend(handles, labels, bbox_to_anchor=(1, 0.53), loc='lower left', ncol=1)

        plt.tight_layout()
        plt.show()

    
def plot_learning_curves(logfile, subplot_cols=None):
    """Plot learning curves from a log file"""
    # Read log file
    logfilepath = os.path.join(get_project_root(), 'logs', logfile)
    log_df = pd.read_csv(logfilepath)
    
    plt.figure(figsize=(6, 6))

    sample_sizes = log_df['sample_size'].unique()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if subplot_cols is not None:
        # Calculate the number of rows for subplots
        num_rows = np.ceil(len(sample_sizes) / subplot_cols).astype(int)
        fig, axes = plt.subplots(num_rows, subplot_cols, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        for i, size in enumerate(sample_sizes):
            ax = axes[i]
            subset = log_df[log_df['sample_size'] == size]
            avg_train_loss = subset.groupby('epoch')['train_loss'].mean()
            avg_val_loss = subset.groupby('epoch')['val_loss'].mean()

            color = color_cycle[i % len(color_cycle)]
            ax.plot(avg_train_loss.index, avg_train_loss, linestyle='-', marker='o', markersize=4, label='Train Loss')
            ax.plot(avg_val_loss.index, avg_val_loss, linestyle='--', marker='x', markersize=4, label='Val Loss')

            ax.set_title(f'{size}-shot Learning Curve')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)

        # Adjust layout and hide empty subplots if necessary
        plt.tight_layout()
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
    else:
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