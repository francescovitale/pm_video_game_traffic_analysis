:: Useful commands:

:: cd <directory>
:: copy path\to\file destination\path
:: xcopy path\to\dirs destination\path /E
:: rmdir "<dir_name>" /s /q
:: ren path\to\file <name>
:: del /F /Q path\to\file 

:: Options:

:: fe_window_size=<integer>
:: normalization_type=[min-max, zscore]
:: clustering_type=[kmeans, gmm, agglomerative]
:: n_clusters=<integer>
:: variant=[im, ilp]
:: noise_threshold=<float>
:: n_reps=<int>

set fe_window_size=5
set n_clusters=3
set normalization_type=min-max
set clustering_type=kmeans
set variant=im
set noise_threshold=0.5

for /D %%p IN ("Results\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

call clean_environment

mkdir Results\PetriNets
mkdir Results\EventLogs
	
python event_log_extraction.py %fe_window_size% %normalization_type% %clustering_type% %n_clusters%
	
xcopy Output\ELE Input\PD\EventLogs /E
xcopy Output\ELE Results\EventLogs /E
	
python process_discovery.py %variant% %noise_threshold%
	
xcopy Output\PD\PetriNets Results\PetriNets /E








