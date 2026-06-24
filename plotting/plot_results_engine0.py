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
output_dir = "Test_on_training_data/plots_engine0"  # Where to save the plots
TARGET_ENGINE = 0  # Filter for only engine 0

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
    """Load both detailed and summary results and filter for engine 0"""
    summary_path = os.path.join(base_dir, summary_file)
    detailed_path = os.path.join(base_dir, detailed_file)
    
    # Load the original data
    full_summary_df = pd.read_csv(summary_path)
    full_detailed_df = pd.read_csv(detailed_path)
    
    # Filter the detailed data for engine 0 only
    detailed_df = full_detailed_df[full_detailed_df['engine_idx'] == TARGET_ENGINE].copy()
    
    # Create a new summary dataframe using only engine 0 data
    # Group by 'n_ar' and calculate the same metrics as in the original summary
    summary_df = detailed_df.groupby('n_ar').agg({
        'rul_rmse': 'mean',
        'rul_error': 'mean',
        'rul_accuracy': 'mean',
        'rmse_ar': 'mean',
        'mae_ar': 'mean',
        'accuracy': 'mean',
        'extended': lambda x: 100 * x.mean()  # Convert to percentage
    }).reset_index()
    
    # Rename columns to match original summary
    summary_df = summary_df.rename(columns={
        'rul_rmse': 'avg_rul_rmse',
        'rul_error': 'avg_rul_error',
        'rul_accuracy': 'avg_rul_accuracy',
        'rmse_ar': 'avg_rmse',
        'mae_ar': 'avg_mae',
        'accuracy': 'avg_accuracy',
        'extended': 'extended_percent'
    })
    
    # Add signal columns if they exist in the original
    signal_cols = [col for col in full_summary_df.columns if col.startswith('signal_') and col.endswith('_count')]
    if signal_cols:
        # For each n_ar value, count the occurrences of each signal type for engine 0
        for col in signal_cols:
            signal_num = int(col.split('_')[1])
            signal_counts = detailed_df[detailed_df['stop_signal'] == signal_num].groupby('n_ar').size().reset_index()
            signal_counts.columns = ['n_ar', col]
            
            # Merge with summary_df
            if not signal_counts.empty:
                summary_df = pd.merge(summary_df, signal_counts, on='n_ar', how='left')
                summary_df[col] = summary_df[col].fillna(0)
    
    print(f"Filtered data for engine {TARGET_ENGINE}")
    print(f"Engine {TARGET_ENGINE} summary has {len(summary_df)} n_ar values")
    print(f"Engine {TARGET_ENGINE} detailed data has {len(detailed_df)} entries")
    
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
    plt.savefig(os.path.join(output_dir, "metrics_by_n_engine0.png"), dpi=300)
    plt.close()
    
    print(f"Saved metrics by n plot to {os.path.join(output_dir, 'metrics_by_n_engine0.png')}")

def plot_rmse_distribution(detailed_df, output_dir):
    """Create boxplots showing the distribution of RMSE values for each n"""
    plt.figure(figsize=(12, 10))
    
    # For a single engine, we'll use strip plots instead of boxplots 
    # since there's only one data point per n_ar value
    
    # RUL RMSE Distribution
    plt.subplot(2, 1, 1)
    sns.stripplot(x='n_ar', y='rul_error', data=detailed_df, size=10, jitter=False, marker='o', color='blue')
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('RUL Error')
    plt.title('Distribution of RUL Error by n')
    plt.grid(True, alpha=0.3)
    
    # Signal RMSE Distribution
    plt.subplot(2, 1, 2)
    sns.stripplot(x='n_ar', y='rmse_ar', data=detailed_df, size=10, jitter=False, marker='o', color='blue')
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('Signal RMSE')
    plt.title('Distribution of Signal RMSE by n')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_distribution_engine0.png"), dpi=300)
    plt.close()
    
    print(f"Saved RMSE distribution plot to {os.path.join(output_dir, 'rmse_distribution_engine0.png')}")

def plot_stopping_signals(summary_df, detailed_df, output_dir):
    """Plot the distribution of stopping signals across different n values"""
    # Extract signal columns
    signal_cols = [col for col in summary_df.columns if col.startswith('signal_') and col.endswith('_count')]
    
    if not signal_cols and 'stop_signal' in detailed_df.columns:
        # If no signal columns in summary but stop_signal in detailed data,
        # create the signal columns from the detailed data
        for signal in detailed_df['stop_signal'].dropna().unique():
            col_name = f'signal_{int(signal)}_count'
            signal_counts = detailed_df[detailed_df['stop_signal'] == signal].groupby('n_ar').size().reset_index()
            signal_counts.columns = ['n_ar', col_name]
            summary_df = pd.merge(summary_df, signal_counts, on='n_ar', how='left')
            summary_df[col_name] = summary_df[col_name].fillna(0)
            signal_cols.append(col_name)
    
    if not signal_cols:
        print("No stopping signal data found for Engine 0")
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
    
    if plot_df.empty or plot_df['count'].sum() == 0:
        print("No stopping signal data to plot for Engine 0")
        return
    
    # Plot stacked bar chart
    plt.figure(figsize=(12, 6))
    
    # Convert to pivot table for stacked bars
    pivot_df = plot_df.pivot(index='n_ar', columns='signal', values='count').fillna(0)
    
    # Add 'No threshold exceeded' column if applicable
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
    plt.savefig(os.path.join(output_dir, "stopping_signals_engine0.png"), dpi=300)
    plt.close()
    
    print(f"Saved stopping signals plot to {os.path.join(output_dir, 'stopping_signals_engine0.png')}")

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
    plt.savefig(os.path.join(output_dir, "extended_predictions_engine0.png"), dpi=300)
    plt.close()
    
    print(f"Saved extended predictions plot to {os.path.join(output_dir, 'extended_predictions_engine0.png')}")

def create_line_plot(detailed_df, output_dir):
    """Create a line plot showing RUL error across n values for engine 0"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(detailed_df['n_ar'], detailed_df['rul_error'], 'o-', color='blue', linewidth=2)
    plt.xlabel('Autoregressive Starting Point (n)')
    plt.ylabel('RUL Error')
    plt.title('RUL Error by n_ar for Engine 0')
    plt.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, txt in enumerate(detailed_df['rul_error']):
        plt.annotate(f"{txt:.1f}", 
                    (detailed_df['n_ar'].iloc[i], detailed_df['rul_error'].iloc[i]),
                    xytext=(0, 5), textcoords='offset points', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rul_error_line_engine0.png"), dpi=300)
    plt.close()
    
    print(f"Saved RUL error line plot to {os.path.join(output_dir, 'rul_error_line_engine0.png')}")

def main():
    try:
        # Find the latest results directory
        results_dir = find_latest_results()
        print(f"Using results from: {results_dir}")
        
        # Load the results filtered for engine 0
        summary_df, detailed_df = load_results(results_dir)
        
        # Create the plots for engine 0 only
        plot_metrics_by_n(summary_df, output_dir)
        plot_rmse_distribution(detailed_df, output_dir)
        plot_stopping_signals(summary_df, detailed_df, output_dir)
        plot_extended_predictions(summary_df, output_dir)
        create_line_plot(detailed_df, output_dir)
        
        print(f"All plots have been generated successfully for engine {TARGET_ENGINE}!")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()