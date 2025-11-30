from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pm4py
import random
import os
import sys
import math
import random
import pandas as pd
import numpy as np
from itertools import permutations
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import time
import func_timeout

input_dir = "Input/"
input_test_dir = input_dir + "Test/"
input_test_eventlogs_dir = input_test_dir + "EventLogs/"
input_test_eventlogs_clashroyale_dir = input_test_eventlogs_dir + "ClashRoyale/"
input_test_eventlogs_rocketleague_dir = input_test_eventlogs_dir + "RocketLeague/"
input_training_dir = input_dir + "Training/"
input_training_eventlogs_dir = input_training_dir + "EventLogs/"
input_training_petrinets_dir = input_training_dir + "PetriNets/"
input_validation_dir = input_dir + "Validation/"
input_validation_eventlogs_dir = input_validation_dir + "EventLogs/"

output_dir = "Output/"

parameters = {}
parameters[log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY] = 'CaseID'

def read_data():

	reference_user = "192.168.0.2"

	petri_nets = {}
	validation_event_logs = {}
	test_event_logs = {}
	
	for state in os.listdir(input_training_petrinets_dir + reference_user + "/"):
		if state.find(".pnml") != -1:
			petri_nets[state.split(".")[0]] = {}
			petri_nets[state.split(".")[0]]["network"], petri_nets[state.split(".")[0]]["initial_marking"], petri_nets[state.split(".")[0]]["final_marking"] = pnml_importer.apply(input_training_petrinets_dir + reference_user + "/" + state)
	
	
	for user in os.listdir(input_validation_eventlogs_dir):
		validation_event_logs[user] = {}
		for state in os.listdir(input_validation_eventlogs_dir + user + "/"):
			validation_event_logs[user][state.split(".")[0]] = xes_importer.apply(input_validation_eventlogs_dir + user + "/" + state)
	
	test_event_logs["ClashRoyale"] = {}
	
	for user in os.listdir(input_test_eventlogs_clashroyale_dir):
		test_event_logs["ClashRoyale"][user] = {}
		for state in os.listdir(input_test_eventlogs_clashroyale_dir + user + "/"):
			test_event_logs["ClashRoyale"][user][state.split(".")[0]] = xes_importer.apply(input_test_eventlogs_clashroyale_dir + user + "/" + state) 
	
	test_event_logs["RocketLeague"] = {}
	for user in os.listdir(input_test_eventlogs_rocketleague_dir):
		test_event_logs["RocketLeague"][user] = {}
		for state in os.listdir(input_test_eventlogs_rocketleague_dir + user + "/"):
			test_event_logs["RocketLeague"][user][state.split(".")[0]] = xes_importer.apply(input_test_eventlogs_rocketleague_dir + user + "/" + state) 
	
	return petri_nets, validation_event_logs, test_event_logs
	
def compute_trace_fitness(validation_event_logs, test_event_logs, petri_nets):

	timing = []
	user_wise_fitness = {}
	validation_fitness_values = {}
	clash_royale_fitness_values = {}
	rocket_league_fitness_values = {}
	
	
	states = list(petri_nets.keys())
	for state in states:
		validation_fitness_values[state] = []
		clash_royale_fitness_values[state] = []
		rocket_league_fitness_values[state] = []
	
	for user in validation_event_logs:
		for state in validation_event_logs[user]:
			for trace in validation_event_logs[user][state]:
				start = time.time()
				aligned_trace = alignments.apply_log([trace], petri_nets[state]["network"], petri_nets[state]["initial_marking"], petri_nets[state]["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
				timing.append(time.time() - start)
				validation_fitness_values[state].append(replay_fitness.evaluate(aligned_trace, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"])
	
	for user in test_event_logs["ClashRoyale"]:
		for state in test_event_logs["ClashRoyale"][user]:
			for trace in test_event_logs["ClashRoyale"][user][state]:
				start = time.time()
				aligned_trace = alignments.apply_log([trace], petri_nets[state]["network"], petri_nets[state]["initial_marking"], petri_nets[state]["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
				timing.append(time.time() - start)
				clash_royale_fitness_values[state].append(replay_fitness.evaluate(aligned_trace, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"])
				
	for user in test_event_logs["RocketLeague"]:
		for state in test_event_logs["RocketLeague"][user]:
			for trace in test_event_logs["RocketLeague"][user][state]:
				start = time.time()
				aligned_trace = alignments.apply_log([trace], petri_nets[state]["network"], petri_nets[state]["initial_marking"], petri_nets[state]["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
				timing.append(time.time() - start)
				rocket_league_fitness_values[state].append(replay_fitness.evaluate(aligned_trace, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"])
	
	timing = sum(timing)/len(timing)
	
	'''
	for user in event_logs:
		user_wise_fitness[user] = []
		for user_state in event_logs[user]:
			actual_state = user_state
			for trace in event_logs[user][user_state]:
				temp = {}
				fitness = {}
				for ref_state in petri_nets:
					aligned_trace = alignments.apply_log([trace], petri_nets[ref_state]["network"], petri_nets[ref_state]["initial_marking"], petri_nets[ref_state]["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
					fitness[ref_state] = replay_fitness.evaluate(aligned_trace, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
					temp[ref_state] = fitness[ref_state]
				max_fitness = -1
				max_state = ""
				for state in fitness:
					if fitness[state] > max_fitness:
						max_fitness = fitness[state]
						max_state = state
				temp["pred"] = max_state
				temp["actual"] = actual_state
				user_wise_fitness[user].append(temp)
	'''			
				

	return validation_fitness_values, clash_royale_fitness_values, rocket_league_fitness_values, timing

def compute_thresholds(validation_fitness_values):

	thresholds = {}
	
	for state in validation_fitness_values:
		# four ways of computing the threshold: min, max, median, mean
		#thresholds[state] = statistics.median(validation_fitness_values[state])
		thresholds[state] = sum(validation_fitness_values[state])/len(validation_fitness_values[state])
		#thresholds[state] = min(validation_fitness_values[state])

	return thresholds
	
def compute_auc(clash_royale_fitness_values, rocket_league_fitness_values, thresholds, trace_fraction):


	clash_royale_occurrences = {}
	rocket_league_occurrences = {}
	roc = {}
	
	states = list(petri_nets.keys())
	
	for state in states:
		clash_royale_occurrences[state] = 0
		rocket_league_occurrences[state] = 0
	
	clash_royale_occurrences["UNKNOWN"]	= 0
	rocket_league_occurrences["UNKNOWN"] = 0

	for state in clash_royale_fitness_values:
		correct_classifications = 0
		for fitness_value in clash_royale_fitness_values[state]:
			if fitness_value >= thresholds[state]:
				clash_royale_occurrences[state] = clash_royale_occurrences[state] + 1
				correct_classifications = correct_classifications + 1
			else:
				clash_royale_occurrences["UNKNOWN"] = clash_royale_occurrences["UNKNOWN"] + 1	
	
	
	for state in rocket_league_fitness_values:
		correct_classifications = 0
		for fitness_value in rocket_league_fitness_values[state]:
			if fitness_value < thresholds[state]:
				rocket_league_occurrences["UNKNOWN"] = rocket_league_occurrences["UNKNOWN"] + 1
				correct_classifications = correct_classifications + 1
			else:
				rocket_league_occurrences[state] = rocket_league_occurrences[state] + 1
		
	clash_royale_dataset = {}
	rocket_league_dataset = {}
	
	
	for state in clash_royale_fitness_values:
		clash_royale_dataset[state] = []
		traces_per_subsession = int(len(clash_royale_fitness_values[state])*trace_fraction)
		n_subsessions = int(len(clash_royale_fitness_values[state])/int(traces_per_subsession))
		for i in range(0,n_subsessions):
			clash_royale_dataset[state].append(clash_royale_fitness_values[state][i*traces_per_subsession:i*traces_per_subsession+traces_per_subsession])
			
	for state in rocket_league_fitness_values:
		rocket_league_dataset[state] = []
		traces_per_subsession = int(len(rocket_league_fitness_values[state])*trace_fraction)
		n_subsessions = int(len(rocket_league_fitness_values[state])/int(traces_per_subsession))
		for i in range(0,n_subsessions):
			rocket_league_dataset[state].append(rocket_league_fitness_values[state][i*traces_per_subsession:i*traces_per_subsession+traces_per_subsession])

	scores = []
	labels = []

	for state in clash_royale_dataset:
		for subset in clash_royale_dataset[state]:
			unknowns = 0
			correct_classifications = 0
			for fitness_value in subset:
				if fitness_value < thresholds[state]:
					unknowns = unknowns + 1
			scores.append(unknowns)
			labels.append(0)
	
	for state in rocket_league_dataset:
		for subset in rocket_league_dataset[state]:
			unknowns = 0
			correct_classifications = 0
			for fitness_value in subset:
				if fitness_value < thresholds[state]:
					unknowns = unknowns + 1
			scores.append(unknowns)
			labels.append(1)
	
	auc = roc_auc_score(labels, scores)
	
	roc["fpr"], roc["tpr"], roc["thresholds"] = roc_curve(labels, scores)
	
	# CONTINUE FROM HERE	

	return roc, auc, clash_royale_occurrences, rocket_league_occurrences

def compute_state_occurrences(test_event_logs, petri_nets, thresholds):

	clash_royale_occurrences = {}
	rocket_league_occurrences = {}
	
	states = list(petri_nets.keys())
	
	for state in states:
		clash_royale_occurrences[state] = 0
		rocket_league_occurrences[state] = 0
	
	clash_royale_occurrences["UNKNOWN"]	= 0
	rocket_league_occurrences["UNKNOWN"] = 0
		
	for user in test_event_logs["ClashRoyale"]:
		for state in test_event_logs["ClashRoyale"][user]:
			for trace in test_event_logs["ClashRoyale"][user][state]:
				temp = {}
				for state_to_compare in petri_nets:
					aligned_trace = alignments.apply_log([trace], petri_nets[state_to_compare]["network"], petri_nets[state_to_compare]["initial_marking"], petri_nets[state_to_compare]["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
					temp[state_to_compare] = replay_fitness.evaluate(aligned_trace, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
				
				
				for idx,state_to_compare in enumerate(thresholds):
					if temp[state_to_compare] >= thresholds[state_to_compare]:
						clash_royale_occurrences[state_to_compare] = clash_royale_occurrences[state_to_compare] + 1
						break
					if idx == len(thresholds) - 1:
						clash_royale_occurrences["UNKNOWN"] = clash_royale_occurrences["UNKNOWN"] + 1
				
	print(clash_royale_occurrences)
	
	for user in test_event_logs["RocketLeague"]:
		for state in test_event_logs["RocketLeague"][user]:			
			for trace in test_event_logs["RocketLeague"][user][state]:
				temp = {}
				for state_to_compare in petri_nets:
					aligned_trace = alignments.apply_log([trace], petri_nets[state_to_compare]["network"], petri_nets[state_to_compare]["initial_marking"], petri_nets[state_to_compare]["final_marking"], parameters=parameters, variant=alignments.Variants.VERSION_STATE_EQUATION_A_STAR)
					temp[state_to_compare] = replay_fitness.evaluate(aligned_trace, variant=replay_fitness.Variants.ALIGNMENT_BASED)["log_fitness"]
				
				for idx,state_to_compare in enumerate(thresholds):
					if temp[state_to_compare] >= thresholds[state_to_compare]:
						rocket_league_occurrences[state_to_compare] = rocket_league_occurrences[state_to_compare] + 1
						break
					if idx == len(thresholds) - 1:
						rocket_league_occurrences["UNKNOWN"] = rocket_league_occurrences["UNKNOWN"] + 1
						
	print(rocket_league_occurrences)	

	sys.exit()

	return clash_royale_occurrences, rocket_league_occurrences

def plot_distributions(clash_royale_occurrences, rocket_league_occurrences):

	# normalize clash royale distribution
	total = sum(clash_royale_occurrences.values())
	normalized_clash_royale_occurrences = {k: v / total for k, v in clash_royale_occurrences.items()}
	
	#normalize rocket league occurrences
	total = sum(rocket_league_occurrences.values())
	normalized_rocket_league_occurrences = {k: v / total for k, v in rocket_league_occurrences.items()}
	
	states = list(clash_royale_occurrences.keys())
	
	clash_royale_probabilities = [normalized_clash_royale_occurrences[s] for s in states]
	rocket_league_probabilities = [normalized_rocket_league_occurrences[s] for s in states]

	x = range(len(keys))

	plt.bar([i - 0.2 for i in x], clash_royale_probabilities, width=0.4, label="CR")
	plt.bar([i + 0.2 for i in x], rocket_league_probabilities, width=0.4, label="RL")

	plt.xticks(x, keys)
	plt.ylabel("Probability")
	plt.title("Probability Distributions of Applications")
	plt.legend()
	plt.show()
	
	
	


	return intersection

def write_accuracy_timing(roc, auc, clash_royale_occurrences, rocket_league_occurrences, timing):

	metrics_file = open(output_dir + "Metrics.txt", "w")
	
	metrics_file.write("AUC: " + str(auc) + "\n")
	metrics_file.write("Clash royale occurrences:\n")
	for state in clash_royale_occurrences:
		metrics_file.write("\t" + state + ": " + str(clash_royale_occurrences[state]) + "\n")
	metrics_file.write("Rocket league occurrences:\n")
	for state in rocket_league_occurrences:
		metrics_file.write("\t" + state + ": " + str(rocket_league_occurrences[state]) + "\n")	
	metrics_file.write("Timing: " + str(timing))
			
	metrics_file.close()

	roc_data = pd.DataFrame({
    'fpr': roc["fpr"],
    'tpr': roc["tpr"],
    'threshold': roc["thresholds"]
	})

	# Save to CSV
	roc_data.to_csv(output_dir + "roc_curve.csv", index=False)	

	return None


try:
	trace_fraction = float(sys.argv[1])
except:
	print("Enter the right number of input arguments")
	sys.exit()

petri_nets, validation_event_logs, test_event_logs = read_data()
validation_fitness_values, clash_royale_fitness_values, rocket_league_fitness_values, timing = compute_trace_fitness(validation_event_logs, test_event_logs, petri_nets)
thresholds = compute_thresholds(validation_fitness_values)
roc, auc, clash_royale_occurrences, rocket_league_occurrences = compute_auc(clash_royale_fitness_values, rocket_league_fitness_values, thresholds, trace_fraction)
#clash_royale_occurrences, rocket_league_occurrences = compute_state_occurrences(test_event_logs, petri_nets, thresholds)
write_accuracy_timing(roc, auc, clash_royale_occurrences, rocket_league_occurrences, timing)

'''
fitness, precision, f1 = compute_inter_user_similarity_fitness_precision_f1(petri_nets, event_logs)
write_inter_user_similarity_metrics(fitness, precision, f1)
fitness, precision, f1 = compute_inter_state_similarity_fitness_precision_f1(petri_nets, event_logs)
write_inter_state_similarity_metrics(fitness, precision, f1)
'''

