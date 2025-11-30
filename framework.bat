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

set fe_window_size=6 7 8
set n_clusters=2 3 4 5
set noise_threshold=0.0

for /D %%p IN ("Results\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for /D %%p IN ("Input\ELE\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

xcopy Data\ClashRoyale Input\ELE /E
xcopy Data\RocketLeague Input\ELE /E

for %%w in (%fe_window_size%) do (

	mkdir Results\Experiment_1
	mkdir Results\Experiment_2

	mkdir Results\Experiment_1\WS_%%w
	mkdir Results\Experiment_2\WS_%%w
	
	for %%c in (%n_clusters%) do (
	
		mkdir Results\Experiment_1\WS_%%w\NC_%%c
		mkdir Results\Experiment_1\WS_%%w\NC_%%c\Training
		mkdir Results\Experiment_1\WS_%%w\NC_%%c\Training\EventLogs
		mkdir Results\Experiment_1\WS_%%w\NC_%%c\Training\PetriNets
		mkdir Results\Experiment_1\WS_%%w\NC_%%c\Test
		mkdir Results\Experiment_1\WS_%%w\NC_%%c\Test\EventLogs
		mkdir Results\Experiment_1\WS_%%w\NC_%%c\Test\PetriNets
	
		mkdir Results\Experiment_2\WS_%%w\NC_%%c
		mkdir Results\Experiment_2\WS_%%w\NC_%%c\Training
		mkdir Results\Experiment_2\WS_%%w\NC_%%c\Training\EventLogs
		mkdir Results\Experiment_2\WS_%%w\NC_%%c\Training\PetriNets
		mkdir Results\Experiment_2\WS_%%w\NC_%%c\Validation
		mkdir Results\Experiment_2\WS_%%w\NC_%%c\Validation\EventLogs
		mkdir Results\Experiment_2\WS_%%w\NC_%%c\Test
		mkdir Results\Experiment_2\WS_%%w\NC_%%c\Test\EventLogs
		mkdir Results\Experiment_2\WS_%%w\NC_%%c\Test\EventLogs\ClashRoyale
		mkdir Results\Experiment_2\WS_%%w\NC_%%c\Test\EventLogs\RocketLeague
		
		call event_log_extraction %%w %%c
		
		xcopy Output\ELE\Experiment_1\Training Results\Experiment_1\WS_%%w\NC_%%c\Training\EventLogs /E
		xcopy Output\ELE\Experiment_1\Test Results\Experiment_1\WS_%%w\NC_%%c\Test\EventLogs /E
		
		xcopy Output\ELE\Experiment_2\Training Results\Experiment_2\WS_%%w\NC_%%c\Training\EventLogs /E
		xcopy Output\ELE\Experiment_2\Validation Results\Experiment_2\WS_%%w\NC_%%c\Validation\EventLogs /E
		xcopy Output\ELE\Experiment_2\Test\ClashRoyale Results\Experiment_2\WS_%%w\NC_%%c\Test\EventLogs\ClashRoyale /E
		xcopy Output\ELE\Experiment_2\Test\RocketLeague Results\Experiment_2\WS_%%w\NC_%%c\Test\EventLogs\RocketLeague /E
		
		for /D %%p IN ("Input\PD\EventLogs\*") DO (
			del /s /f /q %%p\*.*
			for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
			rmdir "%%p" /s /q
		)
		xcopy Output\ELE\Experiment_1\Training Input\PD\EventLogs /E
		
		call process_discovery %noise_threshold%
		
		xcopy Output\PD\PetriNets Results\Experiment_1\WS_%%w\NC_%%c\Training\PetriNets /E
		xcopy Output\PD\PetriNets Results\Experiment_2\WS_%%w\NC_%%c\Training\PetriNets /E
		
		for /D %%p IN ("Input\PD\EventLogs\*") DO (
			del /s /f /q %%p\*.*
			for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
			rmdir "%%p" /s /q
		)
		xcopy Output\ELE\Experiment_1\Test Input\PD\EventLogs /E
			
		call process_discovery %noise_threshold%
		
		xcopy Output\PD\PetriNets Results\Experiment_1\WS_%%w\NC_%%c\Test\PetriNets /E
	)	
	
)







