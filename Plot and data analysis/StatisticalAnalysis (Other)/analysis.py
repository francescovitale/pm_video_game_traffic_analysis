import os
import pandas as pd
import numpy as np

# --- Configuration ---
# Root directory name
INPUT_DIR = "Input"
OUTPUT_DIR = "Output"

# List of runs (folders 1, 2, 3, 4, 5)
RUNS = [1, 2, 3, 4, 5]

# List of method suffixes based on your filenames
METHODS = ["copod", "hbos", "iforest", "pca", "zscore"]

def parse_metrics_file(file_path):
    """
    Reads a Metrics file and extracts the AUC value.
    Expected format: "AUC: 0.12345..."
    """
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if "AUC:" in line:
                    # Split by ':' and take the second part
                    value_str = line.split("AUC:")[1].strip()
                    return float(value_str)
        print(f"Warning: 'AUC:' tag not found in {file_path}")
        return np.nan
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return np.nan
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.nan

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    aggregated_data = []

    print("Processing AUC metrics...")

    for method in METHODS:
        auc_values = []
        
        # Iterate through each run folder
        for run_id in RUNS:
            # Construct path: Input/1/Other/Metrics_copod.txt
            file_name = f"Metrics_{method}.txt"
            file_path = os.path.join(INPUT_DIR, str(run_id), "Other", file_name)
            
            auc = parse_metrics_file(file_path)
            auc_values.append(auc)

        # Calculate Statistics (ignoring NaNs)
        clean_values = [v for v in auc_values if not np.isnan(v)]
        
        if clean_values:
            mean_auc = np.mean(clean_values)
            std_auc = np.std(clean_values)
        else:
            mean_auc = 0.0
            std_auc = 0.0

        # Build row for DataFrame
        row = {
            "Method": method,
            "Mean_AUC": mean_auc,
            "Std_AUC": std_auc,
            "Num_Runs": len(clean_values)
        }
        
        # Add individual run values for transparency
        for idx, val in enumerate(auc_values):
            row[f"Run_{RUNS[idx]}"] = val
            
        aggregated_data.append(row)

    # Create DataFrame and Save
    df = pd.DataFrame(aggregated_data)
    
    # Reorder columns for readability
    cols = ["Method", "Mean_AUC", "Std_AUC", "Num_Runs"] + [f"Run_{r}" for r in RUNS]
    df = df[cols]
    
    output_csv = os.path.join(OUTPUT_DIR, "final_auc_statistics.csv")
    df.to_csv(output_csv, index=False)
    
    print("-" * 30)
    print(df.to_string(index=False))
    print("-" * 30)
    print(f"Successfully saved statistics to: {output_csv}")

if __name__ == "__main__":
    main()