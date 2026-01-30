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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import time
import func_timeout


input_dir = "Input/PD/"
input_eventlogs_dir = input_dir + "EventLogs/"

output_dir = "Output/PD/"
output_petrinets_dir = output_dir + "PetriNets/"
output_metrics_dir = output_dir + "Metrics/"

variant = ""

def read_event_logs():

	event_logs = {}

	for user in os.listdir(input_eventlogs_dir):
		event_logs[user] = {}
		for state in os.listdir(input_eventlogs_dir + "/" + user):
			event_logs[user][state.split(".xes")[0]] = xes_importer.apply(input_eventlogs_dir + user + "/" + state)

	return event_logs
	
	
def process_discovery(event_logs, variant, noise_threshold):

	petri_nets = {}
	
	for user in event_logs:
		petri_nets[user] = {}
		for state in event_logs[user]:
			petri_nets[user][state] = {}
			if variant == "im":
				petri_nets[user][state]["network"], petri_nets[user][state]["initial_marking"], petri_nets[user][state]["final_marking"] = pm4py.discover_petri_net_inductive(event_logs[user][state], noise_threshold = noise_threshold)
			elif variant == "ilp":
				petri_nets[user][state]["network"], petri_nets[user][state]["initial_marking"], petri_nets[user][state]["final_marking"] = pm4py.discover_petri_net_ilp(event_logs[user][state], alpha=1-noise_threshold)	
		
	return petri_nets	

def save_petri_nets(petri_nets):

	for user in petri_nets:
		if not os.path.exists(output_petrinets_dir + user):
			os.mkdir(output_petrinets_dir + user)
			
		for state in petri_nets[user]:
			pnml_exporter.apply(petri_nets[user][state]["network"], petri_nets[user][state]["initial_marking"], output_petrinets_dir + user + "/" + state + ".pnml", final_marking = petri_nets[user][state]["final_marking"])
			pm4py.save_vis_petri_net(petri_nets[user][state]["network"], petri_nets[user][state]["initial_marking"], petri_nets[user][state]["final_marking"], output_petrinets_dir + user + "/" + state + ".png")
	
	return None
	
	
try:
	variant = sys.argv[1]
	noise_threshold = float(sys.argv[2])
except IndexError:
	print("Enter the right number of input arguments.")
	sys.exit()

event_logs = read_event_logs()
petri_nets = process_discovery(event_logs, variant, noise_threshold)
save_petri_nets(petri_nets)