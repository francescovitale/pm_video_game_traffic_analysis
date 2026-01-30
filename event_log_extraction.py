import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import sys
import pandas as pd
import numpy as np
import os

from random import seed
import random
from random import randint

import math

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
import pm4py
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

import dpkt
import socket

from collections import defaultdict

from balanced_kmeans import kmeans_equal
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tabulate import tabulate

# Mapping TCP flag values to readable names
TCP_FLAGS = {
	dpkt.tcp.TH_SYN: "SYN",
	dpkt.tcp.TH_ACK: "ACK",
	dpkt.tcp.TH_FIN: "FIN",
	dpkt.tcp.TH_RST: "RST",
	dpkt.tcp.TH_PUSH: "PSH",
	dpkt.tcp.TH_URG: "URG"
}

input_dir = "Input/ELE/"
input_data_dir = input_dir + "Data/"

output_dir = "Output/ELE/"

output_processmining_dir = output_dir + "ProcessMining/"
output_processmining_training_dir = output_processmining_dir + "Training/"
output_processmining_validation_dir = output_processmining_dir + "Validation/"
output_processmining_test_dir = output_processmining_dir + "Test/"
output_processmining_test_clash_royale_dir = output_processmining_test_dir + "ClashRoyale/"
output_processmining_test_rocket_league_dir = output_processmining_test_dir + "RocketLeague/"

output_other_dir = output_dir + "Other/"
output_other_training_dir = output_other_dir + "Training/"
output_other_validation_dir = output_other_dir + "Validation/"
output_other_test_dir = output_other_dir + "Test/"
output_other_test_clash_royale_dir = output_other_test_dir + "ClashRoyale/"
output_other_test_rocket_league_dir = output_other_test_dir + "RocketLeague/"


def read_data():
	game_sessions = {}

	for user in os.listdir(input_dir):
		game_sessions[user] = []

		for pcap_file in os.listdir(os.path.join(input_dir, user)):
			sessions = defaultdict(list) 
			session_tracker = {} 
			messages = []  

			with open(os.path.join(input_dir, user, pcap_file), "rb") as f:
				pcap = dpkt.pcap.Reader(f)

				for timestamp, buf in pcap:
					eth = dpkt.ethernet.Ethernet(buf) 
					if not isinstance(eth.data, dpkt.ip.IP):
						continue
					
					ip = eth.data
					src_ip = socket.inet_ntoa(ip.src)
					dst_ip = socket.inet_ntoa(ip.dst)

					if not isinstance(ip.data, dpkt.tcp.TCP):
						continue
						
					if src_ip == user:
						direction = "C_to_S"
					else:
						direction = "S_to_C"		
					
					tcp = ip.data
					session_key = (src_ip, dst_ip, tcp.sport, tcp.dport)

					flag_names = [name for bit, name in TCP_FLAGS.items() if tcp.flags & bit]
					tcp_flag = "+".join(flag_names) if flag_names else "OTHER"

					payload_size = len(tcp.data)

					if session_key not in session_tracker:
						session_number = len(sessions[(src_ip, dst_ip)]) + 1
						sessions[(src_ip, dst_ip)].append(session_number)
						session_tracker[session_key] = session_number

					if session_key in session_tracker:
						session_number = session_tracker[session_key]
						messages.append({
							"TIMESTAMP": timestamp,
							"DIRECTION": direction,
							"SOURCE_IP": src_ip,
							"SOURCE_PORT": tcp.sport,
							"DESTINATION_IP": dst_ip,
							"DESTINATION_PORT": tcp.dport,
							"SESSION_NUMBER": session_number,
							"TCP_FLAG": tcp_flag,
							"PAYLOAD_SIZE": payload_size
						})

					
					# Handle session end (FIN or RST)
					if (tcp.flags & dpkt.tcp.TH_FIN) or (tcp.flags & dpkt.tcp.TH_RST):
						session_tracker.pop(session_key, None)  # Remove session
					

			# Convert to DataFrame
			df = pd.DataFrame(messages)
			game_sessions[user].append(df)
			
	#for user in game_sessions:
	#	game_sessions[user] = random.choice(game_sessions[user])
	
	users = list(game_sessions.keys())
	users.remove("192.168.0.2")
	users.insert(0,"192.168.0.2")
	game_sessions = {k: game_sessions[k] for k in users}

	return game_sessions

def feature_extraction(game_sessions, fe_window_size):

	features = ["AVG_PAYLOAD", "N_SERVERS", "N_USER_PORTS", "N_ACK", "N_SYN", "N_FIN", "N_PSH", "N_RST"]
	windows = {}
	
	for user in game_sessions:
		windows[user] = []
		for idx_s, session in enumerate(game_sessions[user]):
			
			#session = game_sessions[user]
			session_length = len(session)
			n_windows = math.floor(session_length/fe_window_size)
			rows = []
			for i in range(0,n_windows):
				window = session.iloc[i*fe_window_size:i*fe_window_size+fe_window_size]
					
				avg_payload_size = sum(list(window["PAYLOAD_SIZE"]))/len(window)
				n_different_ips = list(set(list(set(list(window["DESTINATION_IP"]))) + list(set(list(window["SOURCE_IP"])))))
				try:
					n_different_ips.remove(user)
				except:
					pass
				n_different_ips = len(n_different_ips)
				n_user_ports = list(set(list(set(list(window["DESTINATION_PORT"]))) + list(set(list(window["SOURCE_PORT"])))))
				try:
					n_user_ports.remove(443)
				except:
					pass
				n_user_ports = len(n_user_ports)
				tcp_flags = list(window["TCP_FLAG"])
				n_acks = 0
				n_syns = 0
				n_fins = 0
				n_pshs = 0
				n_rsts = 0
				for flag in tcp_flags:
					if flag.find("ACK") != -1:
						n_acks = n_acks + 1
					if flag.find("SYN") != -1:
						n_syns = n_syns + 1
					if flag.find("FIN") != -1:
						n_fins = n_fins + 1
					if flag.find("PSH") != -1:
						n_pshs = n_pshs + 1
					if flag.find("RST") != -1:
						n_rsts = n_rsts + 1

				windows[user].append([avg_payload_size, n_different_ips, n_user_ports, n_acks, n_syns, n_fins, n_pshs, n_rsts])				
						
				for j in range(0, fe_window_size):
					rows.append([avg_payload_size, n_different_ips, n_user_ports, n_acks, n_syns, n_fins, n_pshs, n_rsts])
				
			fe_df = pd.DataFrame(columns=features, data=rows)
			
			
				
			if len(fe_df) < len(session):
				session.drop(session.tail(len(session)-len(fe_df)).index,inplace = True)
				
			game_sessions[user][idx_s] = pd.concat([session, fe_df], axis=1)

	for user in windows:
		windows[user] = pd.DataFrame(columns=features, data=windows[user])

	return game_sessions, windows
	
def normalize_cluster_game_sessions(game_sessions, normalization_type, clustering_type, n_clusters):

	features = ["AVG_PAYLOAD", "N_SERVERS", "N_USER_PORTS", "N_ACK", "N_SYN", "N_FIN", "N_PSH", "N_RST"]
	reference_user = "192.168.0.2"
	clustered_game_sessions = {}

	for idx_s, session in enumerate(game_sessions[reference_user]):
		session = session[features]
		try:
			reference_user_session_data = pd.concat([reference_user_session_data, session], axis=0, ignore_index=True)
		except:
			reference_user_session_data = session
	
	reference_user_session_data, normalization_parameters = normalize_dataset(reference_user_session_data, 0, normalization_type, None)
	reference_user_session_data, clustering_parameters = cluster_dataset(reference_user_session_data, 0, None, n_clusters, clustering_type)

	for idx_u, user in enumerate(game_sessions):
		clustered_game_sessions[user] = []
		for idx_s, session in enumerate(game_sessions[user]):
			current_session_data = session
			current_session_data = current_session_data[features]
			current_session_data, _ = normalize_dataset(current_session_data, 1, normalization_type, normalization_parameters)
			#current_session_data, _ = compress_dataset(current_session_data, 1, 2, compression_parameters)
			current_session_data, _ = cluster_dataset(current_session_data, 1, clustering_parameters, n_clusters, clustering_type)
			game_sessions[user][idx_s].drop(features, axis=1, inplace=True)
			game_sessions[user][idx_s]["Cluster"] = current_session_data["Cluster"]
			clustered_game_sessions[user].append(game_sessions[user][idx_s])
				
	return clustered_game_sessions

def normalize_dataset(dataset, reuse_parameters, normalization_technique, normalization_parameters_in):
	
	normalized_dataset = dataset.copy()
	normalization_parameters = {}

	if reuse_parameters == 0:
		if normalization_technique == "zscore":
			for column in normalized_dataset:
				column_values = normalized_dataset[column].values
				column_values_mean = np.mean(column_values)
				column_values_std = np.std(column_values)
				if column_values_std == 0:
					column_values_std = 1
				column_values = (column_values - column_values_mean)/column_values_std
				normalized_dataset[column] = column_values
				normalization_parameters[column+"_mean"] = column_values_mean
				normalization_parameters[column+"_std"] = column_values_std
		elif normalization_technique == "min-max":
			column_intervals = get_intervals(dataset)
			for column in normalized_dataset:
				column_data = normalized_dataset[column].tolist()
				intervals = column_intervals[column]
				if intervals[0] != intervals[1]:
					for idx,sample in enumerate(column_data):
						column_data[idx] = (sample-intervals[0])/(intervals[1]-intervals[0])
					
				normalized_dataset[column] = column_data
			
			for column in column_intervals:
				normalization_parameters[column+"_min"] = column_intervals[column][0]
				normalization_parameters[column+"_max"] = column_intervals[column][1]

	else:
		if normalization_technique == "zscore":
			for label in normalized_dataset:
				mean = normalization_parameters_in[label+"_mean"]
				std = normalization_parameters_in[label+"_std"]
				parameter_values = normalized_dataset[label].values
				parameter_values = (parameter_values - float(mean))/float(std)
				normalized_dataset[label] = parameter_values

		elif normalization_technique == "min-max":
			for label in normalized_dataset:
				min = normalization_parameters_in[label+"_min"]
				max = normalization_parameters_in[label+"_max"]
				parameter_values = normalized_dataset[label].values
				if min != max:
					for idx,sample in enumerate(parameter_values):
						parameter_values[idx] = (sample-min)/(max-min)
				normalized_dataset[label] = parameter_values
	
	return normalized_dataset, normalization_parameters	

def get_intervals(timeseries):

	intervals = {}
	
	columns = list(timeseries.columns)
	for column in columns:
		intervals[column] = [9999999999, -9999999999]
	for column in timeseries:
		temp_max = timeseries[column].max()
		temp_min = timeseries[column].min()
		if intervals[column][0] > temp_min:
			intervals[column][0] = temp_min
		if intervals[column][1] < temp_max:
			intervals[column][1] = temp_max

	return intervals

def compress_dataset(dataset, reuse_parameters, n_components, compression_parameters_in):
	compressed_dataset = dataset.copy()
	compression_parameters = None

	if reuse_parameters == 0:
		compression_parameters = PCA(n_components=n_components)
		compressed_dataset = compression_parameters.fit_transform(compressed_dataset)
		columns = []
		for i in range(0, n_components):
			columns.append("PC_"+ str(i))
		compressed_dataset = pd.DataFrame(data=compressed_dataset, columns=columns)
	else:
		compressed_dataset = compression_parameters_in.transform(compressed_dataset)
		columns = []
		for i in range(0, n_components):
			columns.append("PC_"+ str(i))
		compressed_dataset = pd.DataFrame(data=compressed_dataset, columns=columns)


	return compressed_dataset, compression_parameters
	
def cluster_dataset(dataset, reuse_parameters, clustering_parameters_in, n_clusters, clustering_technique):
	clustered_dataset = dataset.copy()
	clustering_parameters = {}

	if reuse_parameters == 0:

		if clustering_technique == "gmm":
			gaussian_mixture = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0).fit(dataset)
			clustering_parameters["weights"] = gaussian_mixture.weights_
			clustering_parameters["means"] = gaussian_mixture.means_
			clustering_parameters["covariances"] = gaussian_mixture.covariances_
			labels_probability = gaussian_mixture.predict_proba(dataset)
			for j in range(n_clusters):
				temp = []
				for i in range(0,len(dataset)):
					temp.append(labels_probability[i][j])
				clustered_dataset["Cluster_" + str(j)] = temp
			cluster_ids = []
			for row_idx,row in clustered_dataset.iterrows():
				clusters = []
				for i in range(0, n_clusters):
					clusters.append(row["Cluster_" + str(i)])
				cluster_ids.append(clusters.index(max(clusters)))
			for i in range(0,n_clusters):
				clustered_dataset = clustered_dataset.drop(axis=1, columns="Cluster_"+str(i))
			clustered_dataset["Cluster"] = cluster_ids


		elif clustering_technique == "kmeans" or clustering_technique == "agglomerative":
			if clustering_technique == "agglomerative":
				cluster_configuration = AgglomerativeClustering(n_clusters=n_clusters, metric='cityblock', linkage='average')
				cluster_labels = cluster_configuration.fit_predict(clustered_dataset)
			elif clustering_technique == "kmeans":
				kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(clustered_dataset)
				cluster_labels = kmeans.labels_

			clustered_dataset["Cluster"] = cluster_labels
			cluster_labels = cluster_labels.tolist()
			used = set();
			clusters = [x for x in cluster_labels if x not in used and (used.add(x) or True)]

			instances_sets = {}
			centroids = {}
			
			for cluster in clusters:
				instances_sets[cluster] = []
				centroids[cluster] = []
				
			
			temp = clustered_dataset
			for index, row in temp.iterrows():
				instances_sets[int(row["Cluster"])].append(row.values.tolist())
			
			n_features_per_instance = len(instances_sets[0][0])-1
			
			for instances_set_label in instances_sets:
				instances = instances_sets[instances_set_label]
				for idx, instance in enumerate(instances):
					instances[idx] = instance[0:n_features_per_instance]
				for i in range(0,n_features_per_instance):
					values = []
					for instance in instances:
						values.append(instance[i])
					centroids[instances_set_label].append(np.mean(values))
					
			clustering_parameters = centroids
			
		elif clustering_technique == "dbscan":
			db = DBSCAN(eps=15, min_samples=2).fit(dataset)
			cluster_labels = db.labels_
			clustered_dataset["Cluster"] = cluster_labels
			used = set();
			clusters = [x for x in cluster_labels if x not in used and (used.add(x) or True)]

			instances_sets = {}
			centroids = {}
			
			for cluster in clusters:
				instances_sets[cluster] = []
				centroids[cluster] = []

			temp = clustered_dataset
			for index, row in temp.iterrows():
				instances_sets[int(row["Cluster"])].append(row.values.tolist())

			n_features_per_instance = len(instances_sets[list(instances_sets.keys())[0]][0])-1
			
			for instances_set_label in instances_sets:
				instances = instances_sets[instances_set_label]
				for idx, instance in enumerate(instances):
					instances[idx] = instance[0:n_features_per_instance]
				for i in range(0,n_features_per_instance):
					values = []
					for instance in instances:
						values.append(instance[i])
					centroids[instances_set_label].append(np.mean(values))
					
			clustering_parameters = centroids
		
	else:

		if clustering_technique == "gmm":
			gaussian_mixture = GaussianMixture(n_components=n_clusters, covariance_type='full')
			gaussian_mixture.weights_ = clustering_parameters_in["weights"]
			gaussian_mixture.means_ = clustering_parameters_in["means"]
			gaussian_mixture.covariances_ = clustering_parameters_in["covariances"]
			gaussian_mixture.precisions_cholesky_ = _compute_precision_cholesky(clustering_parameters_in["covariances"], 'full')
			labels_probability = gaussian_mixture.predict_proba(dataset)
			for j in range(n_clusters):
				temp = []
				for i in range(0,len(dataset)):
					temp.append(labels_probability[i][j])
				clustered_dataset["Cluster_" + str(j)] = temp
			cluster_ids = []
			for row_idx,row in clustered_dataset.iterrows():
				clusters = []
				for i in range(0, n_clusters):
					clusters.append(row["Cluster_" + str(i)])
				cluster_ids.append(clusters.index(max(clusters)))
			for i in range(0,n_clusters):
				clustered_dataset = clustered_dataset.drop(axis=1, columns="Cluster_"+str(i))
			clustered_dataset["Cluster"] = cluster_ids

		elif clustering_technique == "kmeans" or clustering_technique == "agglomerative":
			clusters = []
			for index, instance in clustered_dataset.iterrows():
				min_value = float('inf')
				min_centroid = -1
				for centroid in clustering_parameters_in:
					centroid_coordinates = np.array([float(i) for i in clustering_parameters_in[centroid]])
					dist = np.linalg.norm(instance.values-centroid_coordinates)
					if dist<min_value:
						min_value = dist
						min_centroid = centroid
				clusters.append(min_centroid)
			
			clustered_dataset["Cluster"] = clusters
		

	return clustered_dataset, clustering_parameters	

def event_log_extraction(game_sessions, n_clusters, fe_window_size):

	event_logs = {}
	
	for user in game_sessions:
		event_logs[user] = {}
		for i in range(0, n_clusters):
			event_logs[user]["STATE_" + str(i)] = None
		for idx_s, session in enumerate(game_sessions[user]):	
			#session = game_sessions[user]
			traces = {}
			for i in range(0,n_clusters):
				traces["STATE_" + str(i)] = []
			n_traces = int(len(session)/fe_window_size)
			for i in range(0,n_traces):
				packet_data = session.iloc[i*fe_window_size:i*fe_window_size+fe_window_size]
				state = "STATE_" + str(list(packet_data["Cluster"])[0])
				trace = extract_trace(packet_data)
				traces[state].append(trace)
			
		for state in traces:
			event_logs[user][state] = build_event_log(traces[state])
			
			variants = pm4py.get_variants_as_tuples(event_logs[user][state], activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
			total_variants = 0
			for variant in variants:
				total_variants = total_variants + len(variants[variant])
			ranked_variants = rank_variants(variants)
			event_logs[user][state] = filter_event_log(ranked_variants, total_variants, 0.35)
			
			'''
			try:
				event_logs[user][state] = pm4py.filter_variants_top_k(event_logs[user][state],k=10,activity_key='concept:name',timestamp_key='time:timestamp',case_id_key='case:concept:name')
				#event_logs[user][state] = pm4py.filter_variants_by_coverage_percentage(event_logs[user][state], 0.00875, activity_key='concept:name', timestamp_key='time:timestamp', case_id_key='case:concept:name')
			except:
				pass
			'''	
				
	
	

	return event_logs
	
def build_event_log(traces):
	
	event_log = []
	for idx,trace in enumerate(traces):
		caseid = idx
		for idx_e, event in enumerate(trace):
			event_timestamp = timestamp_builder(idx_e)
			state_transition = event
			event = [caseid, state_transition, event_timestamp]
			event_log.append(event)
	
	event_log = pd.DataFrame(event_log, columns=['CaseID', 'Event', 'Timestamp'])
	event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
	event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
	event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
	event_log = log_converter.apply(event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
		
	return event_log

def rank_variants(variants):

	ranked_variants = dict(sorted(variants.items(), key=lambda item: len(item[1]), reverse=True))

	return ranked_variants
	
def filter_event_log(ranked_variants, total_variants, coverage_percentage):

	variants = []
	variants_amount_to_achieve = int(total_variants*coverage_percentage)
	variants_amount_achieved = 0
	
	for variant in ranked_variants:
		if variants_amount_achieved < variants_amount_to_achieve:
			variants_amount_achieved = variants_amount_achieved + len(ranked_variants[variant])
			variants.append(ranked_variants[variant])
		else:
			break
			
	event_log = sum(variants, [])
	event_log = (pm4py.objects.log.obj.EventLog)(event_log)
		
	return event_log

def timestamp_builder(number):
	
	ss = number
	mm, ss = divmod(ss, 60)
	hh, mm = divmod(mm, 60)
	ignore, hh = divmod(hh, 24)
	
	ss = ss%60
	mm = mm%60
	hh = hh%24
	
	return "1900-01-01T"+str(hh)+":"+str(mm)+":"+str(ss)		
	
def extract_trace(packet_data):

	trace = []

	for idx, packet in packet_data.iterrows():
		trace.append(packet["DIRECTION"] + "_" + packet["TCP_FLAG"])
	
	return trace

def save_game_sessions(game_sessions, windows):

	training_users = ["192.168.0.2"]
	
	# this is how we ensure randomicity
	
	all_users = [
		"192.168.0.13", "192.168.0.25", "192.168.0.29",
		"192.168.0.33", "192.168.0.44", "192.168.0.48",
		"192.168.0.51"
	]
	num_validation = 3
	num_test = 4
	shuffled_users = all_users.copy()
	random.shuffle(shuffled_users)
	validation_users = shuffled_users[:num_validation]
	test_users_clash_royale = shuffled_users[num_validation:num_validation + num_test]
	test_users_rocket_league = ["192.168.0.5", "192.168.0.39", "192.168.0.42", "192.168.0.50"]


	for user in training_users:
		if not os.path.exists(output_processmining_training_dir + user):
			os.mkdir(output_processmining_training_dir + user)
		for state in game_sessions[user]:
			xes_exporter.apply(game_sessions[user][state], output_processmining_training_dir + user + "/" + state + ".xes")
		
		windows[user].to_csv(output_other_training_dir + user + ".csv", index=False)
		
		
	for user in validation_users:
		if not os.path.exists(output_processmining_validation_dir + user):
			os.mkdir(output_processmining_validation_dir + user)
		for state in game_sessions[user]:
			xes_exporter.apply(game_sessions[user][state], output_processmining_validation_dir + user + "/" + state + ".xes")
			
		windows[user].to_csv(output_other_validation_dir + user + ".csv", index=False)	

	for user in test_users_clash_royale:
		if not os.path.exists(output_processmining_test_clash_royale_dir + user):
			os.mkdir(output_processmining_test_clash_royale_dir + user)
		for state in game_sessions[user]:
			xes_exporter.apply(game_sessions[user][state], output_processmining_test_clash_royale_dir + user + "/" + state + ".xes")
			
		windows[user].to_csv(output_other_test_clash_royale_dir + user + ".csv", index=False)		

	for user in test_users_rocket_league:
		if not os.path.exists(output_processmining_test_rocket_league_dir + user):
			os.mkdir(output_processmining_test_rocket_league_dir + user)
		for state in game_sessions[user]:
			xes_exporter.apply(game_sessions[user][state], output_processmining_test_rocket_league_dir + user + "/" + state + ".xes")
			
		windows[user].to_csv(output_other_test_rocket_league_dir + user + ".csv", index=False)			


	return None

try:
	fe_window_size = int(sys.argv[1])
	normalization_type = sys.argv[2]
	clustering_type = sys.argv[3]
	if clustering_type == "kmeans" or clustering_type == "gmm":
		n_clusters = int(sys.argv[4])
except IndexError:
	print("Enter the right number of input arguments")
	sys.exit()
	
game_sessions = read_data()
game_sessions, windows = feature_extraction(game_sessions, fe_window_size)
game_sessions = normalize_cluster_game_sessions(game_sessions, normalization_type, clustering_type, n_clusters)
game_sessions = event_log_extraction(game_sessions, n_clusters, fe_window_size)
save_game_sessions(game_sessions, windows)