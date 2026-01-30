# Requirements to run the method

## Packages
This project has been executed on a Windows 10 machine with Python 3.11.5. A few libraries have been used within Python modules. Among these, there are:

- pm4py 2.7.11.11
- scipy 1.11.2
- scikit-learn 1.3.0
- dpkt 1.9.8

Please note that the list above is not comprehensive and there could be other requirements for running the project.

## Data
The data used in this implementation comes from ".pcap" files recording the network traffic data of individual users' video game sessions. For each user, identified by a local IP, several video game sessions are recorded. To work, the ".pcap" files of each user must be collected under a dedicated folder within the "Input/ELE" directory. For example, if a user has IP 192.168.0.2, a directory named "192.168.0.2" with (possibly multiple) ".pcap" files must be in place.

The data used in the experimentation can be found at: https://doi.org/10.5281/zenodo.17772680. The dataset contains the PCAPs collected from different devices while the gaming event was taking place. In particular, the PCAPs regard two different games: Clash Royale and Rocket League. 
The Clash Royale devices correspond to the following IPs: 192.168.0.2, 192.168.0.13, 192.168.0.25, 192.168.0.29, 192.168.0.33, 192.168.0.44, 192.168.0.48, and 192.168.0.51. 
The Rocket League devices correspond to the following IPs: 192.168.0.5, 192.168.0.39, 192.168.0.42, and 192.168.0.50.

# Execution instructions and project description

The method runs by executing the DOS user_similarity_experimentation.bat script. This script includes experimental parameters to set: 

- The window size to use for feature extraction (fe_window_size)
- The type of normalization applied to the extracted features (normalization_type)
- The type of clustering applied to recognize events (clustering_type)
- The number of clusters to use during clustering (n_clusters)
- The process discovery variant (pd_variant)
- The noise threshold to use during process discovery (noise_threshold)

Once the data has been placed and the variables correctly set, the script executes the two "event_log_extraction.py" and "process_discovery.py" scripts in sequence, resulting in as many event logs and Petri nets as states and users. The results are collected under the "Results" folder.

# Classification experiment

This repository also contains the results of a classification experiment, with related diagrams, against the network data of Clash Royale and Rocket League. The classification experiment can be found under the "Classification experiment" folder. The quantitative results of the experiment can be found under the AnalysisResults folder, which contains the cosine similarity, intersection, and AUC measures for each window length-state space configuration.
