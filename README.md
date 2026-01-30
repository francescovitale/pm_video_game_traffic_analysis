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

# Method description

The method execution requires the placement of Clash Royale and Rocket League data within the corresponding folders under Data/ClashRoyale and Data/RocketLeague. The data can be downloaded from Zenodo (follow the description above).

The method runs by executing the DOS framework.bat script. This script includes experimental parameters to set: 

- The window size to use for feature extraction (fe_window_size)
- The number of clusters to use during clustering (n_clusters)
- The noise threshold to use during process discovery (noise_threshold)
- The number of repetitions (n_reps)

The script cleans the environment and prepares the Results folder. Then, it calls two further DOS scripts. The first is event_log_extraction.bat, which pre-processes the PCAP data and produces several event logs for each Clash Royale/Rocket League device. In particular, for each device, it creates as many event logs as the number of clusters (i.e., the number of states). The second script is process_discovery.bat, which uses the device selected as the training Clash Royale device to extract as many Petri nets as the number of clusters. All the results from each run will be saved under the Results folder.

# Classification experiment

This repository also contains the results of a classification experiment against the network data of Clash Royale and Rocket League. The classification experiment can be found under the "Classification experiment" folder. This folder contains two scripts: execute_experiment_processmining.bat and execute_experiment_other.bat. The first script applies process mining-based classification of Clash Royale and Rocket League traffic. In particular, for each run, it copies the content of FrameworkResults for the selected combination of window size and number of clusters under Classification/Input. Then, it executes evaluation_processmining.py, which implements the logic for classifying segments of Clash Royale and Rocket League traffic. The execute_experiment_other.bat script has a similar objective, but instead applies other one-class classifiers to compare the results with the process mining-based classification. All the results of the execution of the scripts can be found under the AnalysisResults folder.

# Plot and data analysis

This repository contains some analyses performed against the results achieved from the classification experiment. In particular, the Plot and data analysis folder contains:

- BarPlot folder, which contains a script that plots the probability mass functions associated with the Clash Royale and Rocket League classifications in order to determine the number of trace segments classified as unknown.
- Petri net simplicity folder, which contains a script that calculates the arc-degree simplicity of the training Petri nets associated with the states extracted during the execution of the method.
- StatisticalAnalysis (ProcessMining), which contains a script that performs a few statistical analyses from the classification experiment results regarding the process mining-based classification.
- StatisticalAnalysis (Other), which contains a script that performs a few statistical analyses from the classification experiment results regarding the one-class classifiers classification.




