import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Patch

# Set plot style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["font.size"] = 16

input_dir = "Input/"
output_dir = "Output/"

def read_occurrences():
    clash_royale_agg = {}
    rocket_league_agg = {}

    # Walk through the entire directory tree (Input/1/..., Input/2/..., etc.)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Filter for specific metrics files
            if file.startswith("Metrics_") and file.endswith(".txt"):
                file_path = os.path.join(root, file)

                # Parse metadata from filename: Metrics_WS_2_NC_2.txt
                try:
                    temp = file.split(".txt")[0]
                    parts = temp.split("_")
                    window_size = parts[1] + "_" + parts[2]      # e.g., WS_2
                    number_of_states = parts[3] + "_" + parts[4] # e.g., NC_2
                except IndexError:
                    continue # Skip files that don't match the naming convention

                # Initialize dictionary structure if not exists
                if window_size not in clash_royale_agg:
                    clash_royale_agg[window_size] = {}
                    rocket_league_agg[window_size] = {}
                
                if number_of_states not in clash_royale_agg[window_size]:
                    clash_royale_agg[window_size][number_of_states] = {}
                    rocket_league_agg[window_size][number_of_states] = {}

                # Read and parse file content
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                current_section = None

                for line in lines:
                    line = line.strip()
                    if not line: continue

                    # Detect sections
                    if "Clash royale occurrences:" in line:
                        current_section = "CR"
                        continue
                    elif "Rocket league occurrences:" in line:
                        current_section = "RL"
                        continue
                    elif "Timing:" in line or "AUC:" in line:
                        current_section = None
                        continue
                    
                    # Parse values if inside a valid section
                    if current_section and ":" in line:
                        state_label, value_str = line.split(":")
                        state_label = state_label.strip()
                        try:
                            value = float(value_str.strip())
                        except ValueError:
                            continue

                        # Select target dictionary
                        target_dict = clash_royale_agg if current_section == "CR" else rocket_league_agg
                        
                        # Aggregate (SUM values across runs)
                        if state_label not in target_dict[window_size][number_of_states]:
                            target_dict[window_size][number_of_states][state_label] = 0.0
                        
                        target_dict[window_size][number_of_states][state_label] += value

    return clash_royale_agg, rocket_league_agg

def plot_distributions(clash_royale_occurrences, rocket_league_occurrences):

    intersection = {}
    cos_sim = {}

    window_lengths = list(clash_royale_occurrences.keys())
    # Sort window lengths to ensure processing order if needed (optional)
    window_lengths.sort()

    for window_length in window_lengths:

        intersection[window_length] = {}
        cos_sim[window_length] = {}

        numbers_of_states = list(clash_royale_occurrences[window_length].keys())
        numbers_of_states.sort() # Ensure consistent order

        # create output directory per window size
        if not os.path.exists(output_dir + window_length):
            os.makedirs(output_dir + window_length)

        for number_of_states in numbers_of_states:

            intersection[window_length][number_of_states] = 0.0
            cos_sim[window_length][number_of_states] = 0.0

            fig = plt.figure(figsize=(5, 3))
            ax = plt.gca()

            ax.grid(axis="y", linestyle="--", linewidth=0.5)
            ax.set_ylim(0.0, 1.0)

            # Get raw dictionary
            cr_data = clash_royale_occurrences[window_length][number_of_states]
            rl_data = rocket_league_occurrences[window_length][number_of_states]

            # Sort states logically: STATE_0, STATE_1, ... UNKNOWN
            def state_sorter(key):
                if "UNKNOWN" in key:
                    return 999999 # Force UNKNOWN to end
                return int(key.split("_")[-1])

            states = sorted(list(cr_data.keys()), key=state_sorter)
            
            # Create X-axis labels
            states_labels = []
            for s in states:
                if "UNKNOWN" in s:
                    states_labels.append("s$_{unk}$")
                else:
                    # Extract index from STATE_X and add 1
                    idx = int(s.split("_")[-1]) + 1
                    states_labels.append("s$_" + str(idx) + "$")

            # align values based on sorted states list to ensure matching pairs
            clash_royale_values = [cr_data.get(s, 0.0) for s in states]
            rocket_league_values = [rl_data.get(s, 0.0) for s in states]

            # Normalize
            sum_cr = sum(clash_royale_values)
            sum_rl = sum(rocket_league_values)

            # Avoid division by zero if empty
            if sum_cr > 0:
                clash_royale_values = [v / sum_cr for v in clash_royale_values]
            if sum_rl > 0:
                rocket_league_values = [v / sum_rl for v in rocket_league_values]

            num_labels = len(states_labels)
            group_width = 0.8
            bar_width = group_width / 2

            x = range(num_labels)

            ax.bar([i - bar_width/2 for i in x],
                   clash_royale_values,
                   width=bar_width,
                   label="Clash Royale",
                   color="gainsboro",
                   hatch="//",
                   edgecolor="black",
                   linewidth=1.2)

            ax.bar([i + bar_width/2 for i in x],
                   rocket_league_values,
                   width=bar_width,
                   label="Rocket League",
                   color="dimgray",
                   hatch="\\\\",
                   edgecolor="black",
                   linewidth=1.2)

            ax.set_xlim(-0.5, num_labels - 0.5)
            ax.set_xticks(x, states_labels, rotation=10, weight="bold")
            ax.set_yticks(list(np.arange(0.0, 1.01, 0.1)))
            ax.set_ylabel("p$_{x}$(s)", fontweight="bold")

            # ---- metrics ----
            intersection[window_length][number_of_states] = round(
                sum(min(clash_royale_values[k], rocket_league_values[k]) for k in range(len(states))) * 100, 2
            )

            dot_product = sum(a * b for a, b in zip(clash_royale_values, rocket_league_values))
            magnitude_a = math.sqrt(sum(a**2 for a in clash_royale_values))
            magnitude_b = math.sqrt(sum(b**2 for b in rocket_league_values))
            
            if magnitude_a > 0 and magnitude_b > 0:
                cos_sim_val = (dot_product / (magnitude_a * magnitude_b)) * 100
            else:
                cos_sim_val = 0.0
                
            cos_sim[window_length][number_of_states] = round(cos_sim_val, 2)

            # ---- title: ONLY cosine similarity ----
            ax.set_title(
                number_of_states.split("_")[-1]
                + " states, CosSim = "
                + str(cos_sim[window_length][number_of_states])
                + "%",
                fontweight="bold"
            )

            ax.legend(
                loc="upper left",
                fontsize=14,
                frameon=True,
                handlelength=1.2,
                labelspacing=0.3
            )

            # ---- save individual plot ----
            filename = (
                output_dir
                + window_length
                + "/"
                + window_length
                + "_"
                + number_of_states
                + "_COMPARISON.pdf"
            )
            plt.savefig(filename, bbox_inches="tight")
            plt.close(fig)

        # ---- write similarity summary per window ----
        with open(output_dir + window_length + "/Similarity.txt", "w") as f:
            # Sort number_of_states for clean output (2, 3, 4, 5...)
            sorted_nc = sorted(intersection[window_length].keys(), key=lambda x: int(x.split("_")[-1]))
            
            for number_of_states in sorted_nc:
                f.write(
                    number_of_states.split("_")[-1]
                    + " states | I: "
                    + str(intersection[window_length][number_of_states])
                    + "%, CosSim: "
                    + str(cos_sim[window_length][number_of_states])
                    + "%\n"
                )

# Run the functions
if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    clash_royale_occurrences, rocket_league_occurrences = read_occurrences()
    plot_distributions(clash_royale_occurrences, rocket_league_occurrences)