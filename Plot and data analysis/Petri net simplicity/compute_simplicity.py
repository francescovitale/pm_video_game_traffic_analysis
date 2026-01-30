import os
import glob
import re
import statistics
import csv
import pm4py
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from collections import defaultdict

def calculate_mean_simplicity(root_dir="Input", output_csv="simplicity_results.csv"):
    # Dictionary to store results: keys are (WS, NC), values are lists of simplicity scores
    results = defaultdict(list)
    
    # Construct search pattern: Input/WS_x/NC_y/.../PetriNets/.../*.pnml
    search_pattern = os.path.join(root_dir, "WS_*", "NC_*", "ProcessMining", "Training", "PetriNets", "*", "*.pnml")
    
    print(f"Scanning for files in: {search_pattern} ...")
    files = glob.glob(search_pattern)
    
    if not files:
        print("No .pnml files found. Please check the 'root_dir' variable and your folder structure.")
        return

    print(f"Found {len(files)} Petri nets. calculating simplicity...")

    # Regex to extract 'x' (Window Size) and 'y' (Num Clusters)
    # Handles both Windows (\) and Unix (/) separators
    path_regex = re.compile(r"WS_([^/\\]+)[/\\]NC_([^/\\]+)")

    for file_path in files:
        match = path_regex.search(file_path)
        if not match:
            continue
            
        ws_val = match.group(1)
        nc_val = match.group(2)
        
        try:
            # 1. Read the Petri net
            # pm4py.read_pnml returns (net, initial_marking, final_marking)
            net, im, fm = pm4py.read_pnml(file_path)
            
            # 2. Calculate Arc-Degree Simplicity
            # Returns a float (0.0 to 1.0)
            simplicity_score = simplicity_evaluator.apply(net)
            
            # 3. Store score
            results[(ws_val, nc_val)].append(simplicity_score)
            
        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")

    # --- Export to CSV ---
    try:
        # Sort keys for consistent output order
        sorted_keys = sorted(results.keys(), key=lambda k: (try_int(k[0]), try_int(k[1])))
        
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write Header
            writer.writerow(['Window_Size', 'Num_Clusters', 'Mean_Simplicity', 'Net_Count'])
            
            # Write Data
            for ws, nc in sorted_keys:
                scores = results[(ws, nc)]
                mean_val = statistics.mean(scores)
                count = len(scores)
                writer.writerow([ws, nc, mean_val, count])
        
        print(f"\nSuccess! Results exported to: {os.path.abspath(output_csv)}")
        
    except IOError as e:
        print(f"Error writing to CSV file: {e}")

def try_int(val):
    """Helper to sort numerically if inputs are numbers."""
    try:
        return int(val)
    except ValueError:
        return val

if __name__ == "__main__":
    calculate_mean_simplicity()