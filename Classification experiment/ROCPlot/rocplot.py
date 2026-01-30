import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Patch
from sklearn.metrics import auc as compute_auc


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["font.size"] = 10

input_dir = "Input/"

output_dir = "Output/"


# This funciton heavily depends on the input file's text
def read_roc_curves():

	roc_curves = {}
	
	for file in os.listdir(input_dir):
		window_size = file.split(".csv")[0].split("_")[2] + "_" + file.split(".csv")[0].split("_")[3]
		number_clusters = file.split(".csv")[0].split("_")[4] + "_" + file.split(".csv")[0].split("_")[5]
		
		if window_size not in roc_curves.keys():
			roc_curves[window_size] = {}
			
		roc_curves[window_size][number_clusters] = pd.read_csv(input_dir + file)

	return roc_curves
	
def plot_roc_curves(roc_curves):

	auc_values = {}
	window_lengths = list(roc_curves.keys())

	# Define 4 distinct shades of grey (light → dark)
	grey_colors = ["#b0b0b0", "#808080", "#505050", "#202020"]
	linestyles = ['-', '--', ':', '-.']

	# Loop through the window lengths (or whatever your 4 ROC curves represent)
	for window_length in window_lengths:
		auc_values[window_length] = {}
		fig, ax = plt.subplots(figsize=(3, 2))
		

		# Loop through number_of_states inside each window_length
		for idx, (number_of_states, roc_data) in enumerate(roc_curves[window_length].items()):
			# Pick color from 4 greys
			color = grey_colors[idx]
			linestyle = linestyles[idx]
			fpr = roc_data["fpr"]
			tpr = roc_data["tpr"]

			auc_val = compute_auc(fpr, tpr)
			auc_values[window_length][number_of_states] = auc_val

			

			ax.plot(
				fpr,
				tpr,
				color=color,
				linewidth=2,
				linestyle=linestyle,
				label = str(number_of_states.split("_")[-1]) + " states, AUC=" + str(round(auc_val * 100, 2)) + "%"
			)

		# Plot diagonal (random baseline)
		ax.plot([0, 1], [0, 1], linestyle="--", color="lightgray")

		# Labeling and style
		ax.set_ylabel("TPR (%)", fontweight = "bold")
		ax.set_xlabel("FPR (%)", fontweight = "bold")
		#ax.set_title("ROC curve, WL=" + window_length.split("_")[-1], fontsize='small', fontweight = "bold")
		ax.legend(fontsize=7, loc="lower right")
		ax.grid(axis="y", linestyle="--", linewidth=0.5)
		
		# Convert axis scales to percentages (0–100)
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)

		# Define tick positions and labels
		ticks = [i * 0.175 for i in range(0, 7)]  # 0.0, 0.175, 0.35, ..., 1.05
		ticks = [min(t, 1.0) for t in ticks]      # Ensure max is exactly 1.0

		# Define labels as percentages
		tick_labels = [f"{round(t*100, 1)}" for t in ticks]  # e.g., 17.5%
		# Apply to both axes
		ax.set_xticks(ticks)
		ax.set_yticks(ticks)
		ax.set_xticklabels(tick_labels)
		ax.set_yticklabels(tick_labels)
		

		plt.savefig(output_dir + window_length + "_ROC.pdf", bbox_inches='tight')
		plt.close(fig)
			

	return None
	

roc_curves = read_roc_curves()
plot_roc_curves(roc_curves)	