import os
import re
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from numpy.linalg import norm

# --- CONFIGURATION ---
ROOT_DIR = "Input"

# We will run the analysis for these distinct metrics
METRICS_TO_ANALYZE = ["AUC", "Cosine_Sim"]

# Regex patterns
FILENAME_PATTERN = re.compile(r"Metrics_WS_(\d+)_NC_(\d+)")
AUC_PATTERN = re.compile(r"AUC:\s+([0-9.]+)")
OCCURRENCE_HEADER = re.compile(r"(.*) occurrences:")
STATE_PATTERN = re.compile(r"(STATE_\d+|UNKNOWN):\s+(\d+)")

def calculate_cosine_similarity(content):
	"""
	Parses the text content to find two blocks of 'occurrences'
	and computes the cosine similarity between their count vectors.
	"""
	lines = content.split('\n')
	vectors = []
	current_vector = {}
	capturing = False
	
	for line in lines:
		line = line.strip()
		
		# 1. Check if a new occurrences block starts
		if OCCURRENCE_HEADER.search(line):
			if current_vector:
				vectors.append(current_vector)
			current_vector = {}
			capturing = True
			continue
			
		# 2. Capture counts if we are inside a block
		if capturing:
			match = STATE_PATTERN.match(line)
			if match:
				key = match.group(1)	   # e.g., STATE_0 or UNKNOWN
				val = int(match.group(2))  # e.g., 329
				current_vector[key] = val
			elif line.startswith("Timing") or line == "":
				# Stop capturing if block ends (Timing line or empty)
				if current_vector:
					vectors.append(current_vector)
					current_vector = {}
				capturing = False

	# Ensure we found exactly two blocks to compare
	if len(vectors) < 2:
		return np.nan

	

	# Align keys (states) to ensure vectors are ordered identically
	# We take the union of keys to handle potential missing 0-counts safely
	all_keys = sorted(set(vectors[0].keys()) | set(vectors[1].keys()))
	
	vec_a = np.array([vectors[0].get(k, 0) for k in all_keys])
	vec_b = np.array([vectors[1].get(k, 0) for k in all_keys])

	

	# Compute Cosine Similarity: (A . B) / (|A| * |B|)
	dot_product = np.dot(vec_a, vec_b)
	
	
	
	norm_a = norm(vec_a)
	norm_b = norm(vec_b)
	
	if norm_a == 0 or norm_b == 0:
		return 0.0
		
	return dot_product / (norm_a * norm_b)

def extract_data(root_dir):
	data_records = []
	print(f"Scanning '{root_dir}' for ProcessMining files...")
	
	for dirpath, dirnames, filenames in os.walk(root_dir):
		if os.path.basename(dirpath) != "ProcessMining":
			continue

		parent_dir = os.path.dirname(dirpath)
		run_id = os.path.basename(parent_dir)

		for filename in filenames:
			name_match = FILENAME_PATTERN.search(filename)
			if name_match:
				ws_val = int(name_match.group(1))
				nc_val = int(name_match.group(2))
				
				file_path = os.path.join(dirpath, filename)
				try:
					with open(file_path, 'r', encoding='utf-8') as f:
						content = f.read()
						
						# 1. Get AUC
						auc_val = np.nan
						auc_match = AUC_PATTERN.search(content)
						if auc_match:
							auc_val = float(auc_match.group(1))
						
						# 2. Get Cosine Similarity
						cos_val = calculate_cosine_similarity(content)
						
						if not np.isnan(auc_val) and not np.isnan(cos_val):
							data_records.append({
								'Run': run_id,
								'WS': ws_val,
								'NC': nc_val,
								'Configuration': f"WS{ws_val}_NC{nc_val}",
								'AUC': auc_val,
								'Cosine_Sim': cos_val
							})
				except Exception as e:
					print(f"Error reading {file_path}: {e}")

	return pd.DataFrame(data_records)

# --- STATISTICAL FUNCTIONS (Reused for both metrics) ---

def check_assumptions(model, df, metric):
	print(f"\n🔍 Checking Assumptions for: {metric}")
	assumptions_met = True

	# 1. Normality
	shapiro_p = stats.shapiro(model.resid)[1]
	print(f"   1. Normality (Shapiro-Wilk): p={shapiro_p:.5f} -> {'✅ Normal' if shapiro_p > 0.05 else '⚠️ Not Normal'}")
	if shapiro_p <= 0.05: assumptions_met = False

	# 2. Homogeneity
	# Group by factors to check variance consistency
	grouped_data = [group[metric].values for name, group in df.groupby(['WS', 'NC'])]
	levene_p = stats.levene(*grouped_data, center='median')[1]
	print(f"   2. Homogeneity (Levene):	 p={levene_p:.5f} -> {'✅ Equal Var' if levene_p > 0.05 else '⚠️ Unequal Var'}")
	if levene_p <= 0.05: assumptions_met = False

	return assumptions_met

def perform_selective_friedman(df, metric, output_file):
	print(f"\n⚠️ Assumptions Violated -> Performing SELECTIVE FRIEDMAN Tests ({metric})")
	results = []

	def run_friedman(group_col, label):
		pivot = df.pivot_table(index='Run', columns=group_col, values=metric, aggfunc='median')
		if pivot.isnull().values.any(): return np.nan, "Missing Data"
		data = [pivot[col].values for col in pivot.columns]
		if len(data) < 3:
			 if len(data) == 2:
				 s, p = stats.wilcoxon(data[0], data[1])
				 return p, "Wilcoxon"
			 return np.nan, "Not enough groups"
		s, p = stats.friedmanchisquare(*data)
		return p, "Friedman"

	# 1. Interaction (Combinations)
	p_comb, test_comb = run_friedman('Configuration', 'Combinations')
	results.append({'Factor': 'Combinations', 'Test': test_comb, 'p-value': p_comb})
	
	# 2. WS
	p_ws, test_ws = run_friedman('WS', 'WS')
	results.append({'Factor': 'WS', 'Test': test_ws, 'p-value': p_ws})

	# 3. NC
	p_nc, test_nc = run_friedman('NC', 'NC')
	results.append({'Factor': 'NC', 'Test': test_nc, 'p-value': p_nc})

	# Print & Save
	res_df = pd.DataFrame(results)
	print(res_df)
	res_df.to_csv(output_file, index=False)

def perform_parametric_anova(model, output_file):
	print("\n✅ Assumptions Met -> Performing PARAMETRIC Two-Way ANOVA")
	anova_table = sm.stats.anova_lm(model, typ=2)
	print(anova_table)
	anova_table.to_csv(output_file)

def analyze_metric(df, metric):
	"""
	Orchestrates the full analysis for a single metric (AUC or Cosine_Sim).
	"""
	print("\n" + "#"*60)
	print(f"ANALYZING METRIC: {metric}")
	print("#"*60)
	
	stats_file = f"{metric}_Statistics.csv"
	sig_file = f"{metric}_Significance_Results.csv"

	# 1. Summary Statistics Matrix
	grouped = df.groupby(['WS', 'NC'])
	stats_list = []
	for (ws, nc), group in grouped:
		vals = group[metric]
		mean = vals.mean()
		sem = stats.sem(vals)
		ci = sem * stats.t.ppf((1 + 0.95) / 2., len(vals) - 1) if len(vals) > 1 else 0
		stats_list.append({
			'WS': ws, 'NC': nc, 
			f'Mean_{metric}': mean, 'Std_Dev': vals.std(),
			'CI_Lower': mean - ci, 'CI_Upper': mean + ci
		})
	stats_df = pd.DataFrame(stats_list)
	
	# Save Matrix
	stats_df.sort_values(by=['WS', 'NC']).to_csv(stats_file, index=False)
	print(f"-> Statistics matrix saved to {stats_file}")

	# 2. Identify Best Pair
	best = stats_df.sort_values(by=f'Mean_{metric}', ascending=False).iloc[0]
	print(f"\n🏆 BEST CONFIGURATION ({metric}):")
	print(f"   WS={int(best['WS'])}, NC={int(best['NC'])}")
	print(f"   Mean: {best[f'Mean_{metric}']:.4f} (CI: {best['CI_Lower']:.4f} - {best['CI_Upper']:.4f})")

	# 3. Significance Testing
	try:
		# Define model formula dynamically based on metric name
		formula = f'{metric} ~ C(WS) + C(NC) + C(WS):C(NC)'
		model = ols(formula, data=df).fit()
		
		if check_assumptions(model, df, metric):
			perform_parametric_anova(model, sig_file)
		else:
			perform_selective_friedman(df, metric, sig_file)
			
	except Exception as e:
		print(f"Error in statistical analysis for {metric}: {e}")

# --- MAIN ---
def main():
	if not os.path.exists(ROOT_DIR):
		print(f"Directory '{ROOT_DIR}' not found.")
		return

	# 1. Extract Data (includes Cosine Calc)
	full_df = extract_data(ROOT_DIR)
	
	if full_df.empty:
		print("No data found.")
		return

	# 2. Loop through metrics and analyze each independently
	for metric in METRICS_TO_ANALYZE:
		analyze_metric(full_df, metric)

if __name__ == "__main__":
	main()