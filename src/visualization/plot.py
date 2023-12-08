"""
Functions for plotting fine-tuning results.
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
    """Plot in vs. out-of-domain data in separate subplots for a desired grouping."""
    
    # Combine logfiles into single dataframe
    combined_df = pd.DataFrame()
    for logfile in logfiles:
        logfilepath = os.path.join(get_project_root(), 'logs', logfile)
        if 'fewshot_lora' in logfile:
            finetuning_method = 'fewshot_lora'
        elif 'context_distillation' in logfile:
            if 'recursive' in logfile:
                finetuning_method = 'recursive_context_distillation'
            else:
                finetuning_method = 'context_distillation'
        else:
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

            # Plot by group
            for (model_name, sample_size, finetuning_method), subset in group_df.groupby(['model_name', 'sample_size', 'finetuning_method']):
                avg_in_metric = subset[f'eval_in_{metric}'].mean()
                avg_out_metric = subset[f'eval_out_{metric}'].mean()

                if finetuning_method == 'context_distillation':
                    label = f"CD ({sample_size})"
                elif finetuning_method == 'recursive_context_distillation':
                    label = f"recursive CD ({sample_size})"
                else:
                    label = f"{finetuning_method} ({sample_size})"
                
                ax.scatter(avg_in_metric, avg_out_metric, label=label, marker='+', s=200)
            if metric == 'runtime':
                title = f"Runtime (s)"
            elif metric == 'peak_memory_gb':
                title = "Peak Memory Usage (GB)"
            else:
                title = f"{metric.capitalize()}"
            ax.set_title(title)
            ax.set_xlabel('In-domain')
            ax.set_ylabel('Out-of-domain')
            ax.grid(True)
            
            # Equalize axes and plot diagonal line
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            axes_range = (min(x_min, y_min), max(x_max, y_max))
            ax.set_xlim(axes_range)
            ax.set_ylim(axes_range)
            
            ax.plot(axes_range, axes_range, 'k--', alpha=0.2)
            
            # Zeroshot baseline for current metric
            zeroshot_subset = group_df[group_df['finetuning_method'] == 'zeroshot']
            zeroshot_in_avg = zeroshot_subset[f'eval_in_{metric}'].mean()
            zeroshot_out_avg = zeroshot_subset[f'eval_out_{metric}'].mean()

            ax.scatter([zeroshot_in_avg], [ax.get_ylim()[0]], color='red', marker='|', s=300, linewidth=2, alpha=0.7, zorder=5)  # Vertical line
            ax.scatter([ax.get_xlim()[0]], [zeroshot_out_avg], color='red', marker='_', s=300, linewidth=2, alpha=0.7, zorder=5, label='zeroshot baseline')  # Horizontal line
        
        # Sort and move legend
        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: x[1])
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        fig.legend(sorted_handles, sorted_labels, bbox_to_anchor=(0.995, 0.49), loc='lower left', ncol=1, title=group)

        plt.tight_layout()
        filepath = os.path.join(get_project_root(), 'experiments/figures', f"metrics_{group}.png")
        fig.savefig(filepath, bbox_inches='tight')
        plt.show()
    
def plot_learning_curves(logfile, subplot=True):
    """Plot learning curves from a log file"""
    # Read log file
    logfilepath = os.path.join(get_project_root(), 'logs', logfile)
    log_df = pd.read_csv(logfilepath)
    if 'fewshot_lora' in logfile:
        finetuning_method = 'fewshot_lora'
    else:
        finetuning_method = 'fewshot'

    sample_sizes = log_df['sample_size'].unique()
    model_names = log_df['model_name'].unique()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 'x']
    
    if subplot is not None:
        # Calculate the number of rows for subplots
        fig, axes = plt.subplots(1, len(sample_sizes), figsize=(5*len(sample_sizes), 5))
        axes = axes.flatten()

        for i, model_name in enumerate(model_names):
            # color = color_cycle[i+1]
            marker = markers[i]
            for subplot, sample_size in enumerate(sample_sizes):
                ax = axes[subplot]
                subset = log_df[(log_df['sample_size'] == sample_size) & (log_df['model_name'] == model_name)]
                avg_train_loss = subset.groupby('epoch')['train_loss'].mean()
                avg_val_loss = subset.groupby('epoch')['val_loss'].mean()
                
                # color = color_cycle[subplot+1]
                ax.plot(avg_train_loss.index, avg_train_loss, linestyle='-', marker=marker, markersize=4, color='g', label=f'Train Loss ({model_name})')
                ax.plot(avg_val_loss.index, avg_val_loss, linestyle='--', marker=marker, markersize=4, color='darkorange', label=f'Val Loss ({model_name})')

                ax.set_title(f'{sample_size}-shot')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True)
                
        # Sort and move legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.05), loc='lower center', ncol=4)

        plt.tight_layout()
        filepath = os.path.join(get_project_root(), 'experiments/figures', f"learning_curves_{finetuning_method}.png")
        fig.savefig(filepath, bbox_inches='tight')
        plt.show()