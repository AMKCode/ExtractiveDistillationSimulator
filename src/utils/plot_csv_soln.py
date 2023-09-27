import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 

def file_has_header(file_path):
    if not os.path.exists(file_path):
        return False
    try:
        df = pd.read_csv(file_path)  # Read only the first row to check for header
    except pd.errors.EmptyDataError:
        print(file_path, "empty data error")
        return False
    return df.columns.size > 0  # Check if there are column names (header)

def plot_csv_data(passed_csv_path, failed_csv_path, labels, plot_path):
    _, ax = plt.subplots()
    # Check headers for each file and inform which one lacks it
    if not os.path.exists(passed_csv_path):
        print("Passed csv doesnt exist")
        return
    
    if not file_has_header(passed_csv_path):
        print(f"The file '{passed_csv_path}' does not exist or lacks a header.")
        return
    
    if not file_has_header(failed_csv_path):
        print(f"The file '{failed_csv_path}' does not exist or lacks a header.")
        return

    # Helper function to plot data
    def plot_data(df, label, color):
        if not df.empty and all(col in df.columns for col in labels):
            ax.scatter(df[labels[0]], df[labels[1]], alpha=0.5, label=label, color=color, s=1)

    # Read and plot passed cases
    df_passed = pd.read_csv(passed_csv_path)
    plot_data(df_passed, 'Passed Cases', 'b')

    # Read and plot failed cases
    df_failed = pd.read_csv(failed_csv_path)
    plot_data(df_failed, 'Failed Cases', 'r')

    # Set plot properties
    ax.set_aspect('equal')
    ax.plot([0, 1], [1, 0])
    ax.set_title('Test Cases')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.legend()
    plt.savefig(plot_path)
    
