import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Patch


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["font.size"] = 16

input_dir = "Input/"

output_dir = "Output/"


# This funciton heavily depends on the input file's text
def read_occurrences():

	clash_royale_occurrences = {}
	rocket_league_occurrences = {}
	
	for file in os.listdir(input_dir):
		metrics_file = open(input_dir + file)
		lines = metrics_file.readlines()
		
		temp = file.split(".txt")[0]
		window_size = temp.split("_")[1] + "_" + temp.split("_")[2]
		number_of_states = temp.split("_")[3] + "_" + temp.split("_")[4]
		
		if window_size not in list(clash_royale_occurrences.keys()):
			clash_royale_occurrences[window_size] = {}
		
		clash_royale_occurrences[window_size][number_of_states] = {}
		
		for i in range(0, int(number_of_states[-1]) + 1):
			if i < int(number_of_states[-1]):
				clash_royale_occurrences[window_size][number_of_states]["STATE_" + str(i)] = int(lines[2+i].split(":")[-1])
			else:
				clash_royale_occurrences[window_size][number_of_states]["UNKNOWN"] = int(lines[2+i].split(":")[-1])
		
		if window_size not in list(rocket_league_occurrences.keys()):
			rocket_league_occurrences[window_size] = {}
			
		rocket_league_occurrences[window_size][number_of_states] = {}	
		
		for i in range(0, int(number_of_states[-1]) + 1):
			if i < int(number_of_states[-1]):
				rocket_league_occurrences[window_size][number_of_states]["STATE_" + str(i)] = int(lines[2+int(number_of_states[-1])+2+i].split(":")[-1])
			else:
				rocket_league_occurrences[window_size][number_of_states]["UNKNOWN"] = int(lines[2+int(number_of_states[-1])+2+i].split(":")[-1])
		
		metrics_file.close()


	return clash_royale_occurrences, rocket_league_occurrences
	
def plot_distributions(clash_royale_occurrences, rocket_league_occurrences):

	intersection = {}
	cos_sim = {}

	window_lengths = list(clash_royale_occurrences.keys())

	for window_length in window_lengths:
	
		intersection[window_length] = {}
		cos_sim[window_length] = {}
	
		fig = plt.figure(figsize=(15,3))
	
		numbers_of_states = list(clash_royale_occurrences[window_length].keys())
		for idx_plot, number_of_states in enumerate(numbers_of_states):
		
			intersection[window_length][number_of_states] = 0.0
			cos_sim[window_length][number_of_states] = 0.0
		
			ax = fig.add_subplot(1,len(numbers_of_states),idx_plot+1)
			ax.grid(axis="y", linestyle="--", linewidth=0.5)
			ax.set_ylim(0.0, 1.0)
		
			states = list(clash_royale_occurrences[window_length][number_of_states].keys())
			states_labels = ["s$_" + str(idx_s + 1) + "$" for idx_s,state in enumerate(states)]
			states_labels[-1] = "s$_{unk}$"
			
			clash_royale_values = list(clash_royale_occurrences[window_length][number_of_states].values())
			rocket_league_values = list(rocket_league_occurrences[window_length][number_of_states].values())
			
			clash_royale_values = [v/sum(clash_royale_values) for v in clash_royale_values]
			rocket_league_values = [v/sum(rocket_league_values) for v in rocket_league_values]

			num_labels = len(states_labels)
			group_width = 0.8  # total width per tick
			bar_width = group_width / 2  # two bars per tick

			x = range(num_labels)

			if idx_plot < len(numbers_of_states)-1:
				ax.bar([i - bar_width/2 for i in x], clash_royale_values, width=bar_width, 
					   color="gainsboro", edgecolor="black", hatch="//", linewidth=1.2)
				ax.bar([i + bar_width/2 for i in x], rocket_league_values, width=bar_width, 
					   color="dimgray", edgecolor="black", hatch="\\\\", linewidth=1.2)
			else:
				ax.bar([i - bar_width/2 for i in x], clash_royale_values, width=bar_width, 
					   label="Clash Royale", color="gainsboro", hatch="//", edgecolor="black", linewidth=1.2)
				ax.bar([i + bar_width/2 for i in x], rocket_league_values, width=bar_width, 
					   label="Rocket League", color="dimgray", hatch="\\\\", edgecolor="black", linewidth=1.2)

			# Dynamically adjust x-axis limits to ensure consistent visual spacing
			ax.set_xlim(-0.5, num_labels - 0.5)
			
			intersection[window_length][number_of_states] = round(sum(min(clash_royale_values[k], rocket_league_values[k]) for k in range(0,len(states))) * 100,2)
			dot_product = sum(a * b for a, b in zip(clash_royale_values, rocket_league_values))
			magnitude_a = math.sqrt(sum(a**2 for a in clash_royale_values))
			magnitude_b = math.sqrt(sum(b**2 for b in rocket_league_values))
			cos_sim[window_length][number_of_states] = round((dot_product / (magnitude_a * magnitude_b)) * 100,2)

			ax.set_title(number_of_states.split("_")[-1] + " states\nI=" + str(intersection[window_length][number_of_states]) + "%, CosSim=" + str(cos_sim[window_length][number_of_states]) + "%", fontsize='medium', fontweight = "bold")
			ax.set_xticks(x, states_labels, rotation = 10, weight="bold")
			ax.set_yticks(list(np.arange(0.0, 1.01, 0.1)))
			if idx_plot == 0:
				ax.set_ylabel("p$_{x}$(s)", fontweight = "bold")
			#plt.title("$I_{CR,RL}=$" + str(intersection) + "%")
			#ax.legend()
			
		leg = plt.figlegend(loc = 'lower center', ncol=2, prop={'size': 12, 'weight':'bold'}, bbox_to_anchor=(0.5, -0.1), borderpad=0.1)
		for legline in leg.get_lines():
			legline.set_linewidth(2.5)
		
		if not os.path.exists(output_dir + window_length):
			os.makedirs(output_dir + window_length) 
		
		plt.savefig(output_dir + window_length + "/" + window_length + "_COMPARISON.pdf", bbox_inches='tight')
		plt.close(fig)
		
		similarity_file = open(output_dir + window_length + "/Similarity.txt", "w")
		for number_of_states in intersection[window_length]:
			similarity_file.write(number_of_states.split("_")[-1] + " states I: " + str(intersection[window_length][number_of_states]) + "%, CosSim: " + str(cos_sim[window_length][number_of_states]) + "%\n")
		similarity_file.close()
	
clash_royale_occurrences, rocket_league_occurrences = read_occurrences()
plot_distributions(clash_royale_occurrences, rocket_league_occurrences)

	