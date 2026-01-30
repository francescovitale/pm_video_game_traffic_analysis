import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.copod import COPOD

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_DIR = "Input/"
INPUT_TRAINING_DIR = os.path.join(INPUT_DIR, "Training/")
INPUT_VALIDATION_DIR = os.path.join(INPUT_DIR, "Validation/")
INPUT_TEST_DIR = os.path.join(INPUT_DIR, "Test/")
INPUT_TEST_CLASH_DIR = os.path.join(INPUT_TEST_DIR, "ClashRoyale/")
INPUT_TEST_ROCKET_DIR = os.path.join(INPUT_TEST_DIR, "RocketLeague/")

OUTPUT_DIR = "Output/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def read_data():
    
    def load_folder(folder_path):
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not files:
            print(f"Warning: No CSV files found in {folder_path}")
            return pd.DataFrame()
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    training_windows = load_folder(INPUT_TRAINING_DIR)
    validation_windows = load_folder(INPUT_VALIDATION_DIR)
    
    test_windows = {
        "ClashRoyale": load_folder(INPUT_TEST_CLASH_DIR),
        "RocketLeague": load_folder(INPUT_TEST_ROCKET_DIR)
    }

    return training_windows, validation_windows, test_windows

def get_anomaly_scores(classifier, X):
    model_type = classifier["type"]

    if model_type == "zscore":
        z_scores = np.abs((X - classifier["mean"]) / classifier["std"])
        return z_scores.mean(axis=1)

    model = classifier["model"]
    
    pyod_models = ["hbos", "pca", "knn", "copod"]
    if model_type in pyod_models:
        return model.decision_function(X)

    if model_type == "iforest":
        return -model.decision_function(X)

    raise ValueError(f"Unsupported classifier type: {model_type}")

def train_classifier(training_windows, validation_windows, classifier_type):
    X_train = training_windows.values
    
    c_type = classifier_type.lower()
    classifier = {"type": c_type}

    if c_type == "iforest":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1))
        ])
        model.fit(X_train)
        classifier["model"] = model

    elif c_type == "hbos":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", HBOS(contamination=0.05, n_bins=20))
        ])
        model.fit(X_train)
        classifier["model"] = model

    elif c_type == "pca":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", PCA(contamination=0.05, random_state=42))
        ])
        model.fit(X_train)
        classifier["model"] = model

    elif c_type == "knn":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNN(contamination=0.05, n_neighbors=5, n_jobs=-1))
        ])
        model.fit(X_train)
        classifier["model"] = model
    
    elif c_type == "copod":
        model = Pipeline([
            ("scaler", StandardScaler()), 
            ("model", COPOD(contamination=0.05))
        ])
        model.fit(X_train)
        classifier["model"] = model

    elif c_type == "zscore":
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        classifier["mean"] = mean
        classifier["std"] = std

    else:
        raise ValueError("Unknown classifier. Options: iforest, hbos, zscore, pca, knn, copod")

    return classifier

def compute_auc(classifier, test_windows, trace_fraction=0.2):
    y_true = []
    y_scores = []

    def split_into_fractions(X):
        n = len(X)
        size = max(1, int(n * trace_fraction))
        return [X[i:i + size] for i in range(0, n, size)]

    if not test_windows["ClashRoyale"].empty:
        chunks = split_into_fractions(test_windows["ClashRoyale"].values)
        for subset in chunks:
            raw_scores = get_anomaly_scores(classifier, subset)
            y_scores.append(np.mean(raw_scores))
            y_true.append(0)

    if not test_windows["RocketLeague"].empty:
        chunks = split_into_fractions(test_windows["RocketLeague"].values)
        for subset in chunks:
            raw_scores = get_anomaly_scores(classifier, subset)
            y_scores.append(np.mean(raw_scores))
            y_true.append(1)

    if not y_true:
        print("Error: No test data available.")
        return None, 0.0

    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    return roc_data, auc

def write_metrics(roc_data, auc):
    with open(os.path.join(OUTPUT_DIR, "Metrics.txt"), "w") as f:
        f.write(f"AUC: {auc}\n")
    
    if roc_data:
        roc_df = pd.DataFrame({
            'fpr': roc_data["fpr"],
            'tpr': roc_data["tpr"],
            'threshold': roc_data["thresholds"]
        })
        roc_df.to_csv(os.path.join(OUTPUT_DIR, "roc_curve.csv"), index=False)
    
    print(f"Success. AUC: {auc:.4f}")

if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            print("Usage: python script.py <classifier_type> <trace_fraction>")
            print("Options: iforest, hbos, zscore, pca, knn, copod")
            sys.exit(1)
            
        classifier_type_arg = sys.argv[1]
        trace_fraction_arg = float(sys.argv[2])
        
        print(f"Running {classifier_type_arg} (Fraction: {trace_fraction_arg})...")

        train_df, val_df, test_dfs = read_data()
        
        if train_df.empty:
            print("Error: Training data is empty.")
            sys.exit(1)

        clf = train_classifier(train_df, val_df, classifier_type_arg)
        roc_info, auc_score = compute_auc(clf, test_dfs, trace_fraction_arg)
        write_metrics(roc_info, auc_score)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()