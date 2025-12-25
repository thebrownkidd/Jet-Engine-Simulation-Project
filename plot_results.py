# Plot Generator for RUL Test Results
# Author: thebrownkidd
# Date: 2025-09-13

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns

# ======== CONFIG ========
results_dir = "Test_on_training_data"  # Directory where results are saved
detailed_file = "signal_accuracy.csv"  # Detailed results file
summary_file = "summary_by_n.csv"      # Summary by n file
output_dir = "Test_on_training_data/plots"  # Where to save the plots

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to find the most recent results if multiple exist
def find_latest_results():
    if os.path.exists(os.path.join(results_dir, summary_file)):
        return results_dir
    
    # Look for subdirectories with timestamp patterns
    subdirs = glob.glob(os.path.join(results_dir, "*"))
    valid_dirs = [d for d in subdirs if os.path.isdir(d) and os.path.exists(os.path.join(d, summary_file))]
    
    if not valid_dirs:
        raise FileNotFoundError(f"No results found in {results_dir} or its subdirectories")
    
    # Sort by modification time to get the latest
    latest_dir = sorted(valid_dirs, key=os.path.getmtime, reverse=True)[0]
    return latest_dir

def load_results(base_dir):
    """Load both detailed and summary results"""
    summary_path = os.path.join(base_dir, summary_file)
    detailed_path = os.path.join(base_dir, detailed_file)
    
    summary_df = pd.read_csv(summary_path)
    detailed_df = pd.read_csv(detailed_path)
    
    print(f"Loaded summary data with {len(summary_df)} n_ar values")
    print(f"Loaded detailed data with {len(detailed_df)} entries")
    
    return summary_df, detailed_df

def plot_metrics_by_n(summary_df, output_dir):
    """Create line plots for metrics across different n values"""
    plt.figure(figsize=(12, 8))
    
    # Plot RUL metrics
    plt.subplot(2, 2, 1)
    plt.plot(summary_df['n_ar'], summary_df['avg_rul_rmse'], 'o-', color='blue', label='RUL RMSE')
    plt.plot(summary_df['n_ar'], summary_df['avg_rul_error'], 's--', color='red', label='RUL Error')
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('Error')
    plt.title('RUL Error Metrics by n')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot RUL accuracy
    plt.subplot(2, 2, 2)
    plt.plot(summary_df['n_ar'], summary_df['avg_rul_accuracy'], 'o-', color='green')
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('Accuracy')
    plt.title('RUL Accuracy by n')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    # Plot signal prediction metrics
    plt.subplot(2, 2, 3)
    plt.plot(summary_df['n_ar'], summary_df['avg_rmse'], 'o-', color='blue', label='RMSE')
    plt.plot(summary_df['n_ar'], summary_df['avg_mae'], 's--', color='red', label='MAE')
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('Error')
    plt.title('Signal Prediction Error by n')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot signal prediction accuracy
    plt.subplot(2, 2, 4)
    plt.plot(summary_df['n_ar'], summary_df['avg_accuracy'], 'o-', color='green')
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('Accuracy')
    plt.title('Signal Prediction Accuracy by n')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_by_n.png"), dpi=300)
    plt.close()
    
    print(f"Saved metrics by n plot to {os.path.join(output_dir, 'metrics_by_n.png')}")

def plot_rmse_distribution(detailed_df, output_dir):
    """Create boxplots showing the distribution of RMSE values for each n"""
    plt.figure(figsize=(12, 10))
    
    # RUL RMSE Distribution
    plt.subplot(2, 1, 1)
    sns.boxplot(x='n_ar', y='rul_error', data=detailed_df)
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('RUL Error')
    plt.title('Distribution of RUL Error by n')
    plt.grid(True, alpha=0.3)
    
    # Signal RMSE Distribution
    plt.subplot(2, 1, 2)
    sns.boxplot(x='n_ar', y='rmse_ar', data=detailed_df)
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('Signal RMSE')
    plt.title('Distribution of Signal RMSE by n')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_distribution.png"), dpi=300)
    plt.close()
    
    print(f"Saved RMSE distribution plot to {os.path.join(output_dir, 'rmse_distribution.png')}")

def plot_stopping_signals(summary_df, detailed_df, output_dir):
    """Plot the distribution of stopping signals across different n values"""
    # Extract signal columns
    signal_cols = [col for col in summary_df.columns if col.startswith('signal_') and col.endswith('_count')]
    
    if not signal_cols:
        print("No stopping signal data found in summary file")
        return
    
    # Create a dataframe for plotting
    plot_data = []
    for _, row in summary_df.iterrows():
        n_ar = row['n_ar']
        for col in signal_cols:
            signal_num = int(col.split('_')[1])
            count = row[col] if not pd.isna(row[col]) else 0
            plot_data.append({
                'n_ar': n_ar,
                'signal': f'Signal {signal_num}',
                'count': count
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    if plot_df.empty:
        print("No stopping signal data to plot")
        return
    
    # Plot stacked bar chart
    plt.figure(figsize=(12, 6))
    
    # Convert to pivot table for stacked bars
    pivot_df = plot_df.pivot(index='n_ar', columns='signal', values='count').fillna(0)
    
    # Add 'No threshold exceeded' column if available
    no_threshold_counts = []
    for n_ar in pivot_df.index:
        n_results = len(detailed_df[detailed_df['n_ar'] == n_ar])
        signal_sum = pivot_df.loc[n_ar].sum()
        no_threshold_counts.append(n_results - signal_sum)
    
    pivot_df['No threshold exceeded'] = no_threshold_counts
    
    # Plot stacked bar chart
    pivot_df.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('Count')
    plt.title('Distribution of Stopping Signals by n')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stopping_signals.png"), dpi=300)
    plt.close()
    
    print(f"Saved stopping signals plot to {os.path.join(output_dir, 'stopping_signals.png')}")

def plot_extended_predictions(summary_df, output_dir):
    """Plot percentage of predictions that extended beyond sequence end"""
    plt.figure(figsize=(10, 6))
    
    plt.bar(summary_df['n_ar'], summary_df['extended_percent'], color='purple')
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('Percentage (%)')
    plt.title('Percentage of Engines with Predictions Extending Beyond Sequence End')
    plt.grid(True, alpha=0.3)
    
    # Add percentage labels on top of bars
    for i, v in enumerate(summary_df['extended_percent']):
        plt.text(summary_df['n_ar'].iloc[i], v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "extended_predictions.png"), dpi=300)
    plt.close()
    
    print(f"Saved extended predictions plot to {os.path.join(output_dir, 'extended_predictions.png')}")

def create_heatmap(detailed_df, output_dir):
    """Create a heatmap showing RUL error by engine and n value"""
    # Pivot the data to create a matrix of engines vs n_ar values
    pivot_df = detailed_df.pivot(index='engine_idx', columns='n_ar', values='rul_error')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, cmap='viridis_r', annot=True, fmt=".1f", linewidths=.5)
    plt.title('RUL Error by Engine and n_ar')
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('Engine Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rul_error_heatmap.png"), dpi=300)
    plt.close()
    
    print(f"Saved RUL error heatmap to {os.path.join(output_dir, 'rul_error_heatmap.png')}")

def main():
    try:
        # Find the latest results directory
        results_dir = find_latest_results()
        print(f"Using results from: {results_dir}")
        
        # Load the results
        summary_df, detailed_df = load_results(results_dir)
        
        # Create the plots
        plot_metrics_by_n(summary_df, output_dir)
        plot_rmse_distribution(detailed_df, output_dir)
        plot_stopping_signals(summary_df, detailed_df, output_dir)
        plot_extended_predictions(summary_df, output_dir)
        create_heatmap(detailed_df, output_dir)
        
        print("All plots have been generated successfully!")
        
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()