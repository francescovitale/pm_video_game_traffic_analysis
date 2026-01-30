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

::set fe_window_size=2 3 4 5 6 7 8
::set n_clusters=2 3 4 5

set fe_window_size=2 3 4 5 6 7 8
set n_clusters=2 3 4 5
set noise_threshold=0.0
set n_reps=1 2 3 4 5

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

mkdir Results

xcopy Data\ClashRoyale Input\ELE /E
xcopy Data\RocketLeague Input\ELE /E

for %%r in (%n_reps%) do (

	mkdir Results\%%r

	for %%w in (%fe_window_size%) do (

		mkdir Results\%%r\WS_%%w
		
		for %%c in (%n_clusters%) do (
		
			mkdir Results\%%r\WS_%%w\NC_%%c
			mkdir Results\%%r\WS_%%w\NC_%%c\ProcessMining\Training
			mkdir Results\%%r\WS_%%w\NC_%%c\ProcessMining\Training\EventLogs
			mkdir Results\%%r\WS_%%w\NC_%%c\ProcessMining\Training\PetriNets
			mkdir Results\%%r\WS_%%w\NC_%%c\ProcessMining\Validation
			mkdir Results\%%r\WS_%%w\NC_%%c\ProcessMining\Validation\EventLogs
			mkdir Results\%%r\WS_%%w\NC_%%c\ProcessMining\Test
			mkdir Results\%%r\WS_%%w\NC_%%c\ProcessMining\Test\EventLogs
			mkdir Results\%%r\WS_%%w\NC_%%c\ProcessMining\Test\EventLogs\ClashRoyale
			mkdir Results\%%r\WS_%%w\NC_%%c\ProcessMining\Test\EventLogs\RocketLeague
			
			mkdir Results\%%r\WS_%%w\NC_%%c\Other\Training
			mkdir Results\%%r\WS_%%w\NC_%%c\Other\Validation
			mkdir Results\%%r\WS_%%w\NC_%%c\Other\Test
			mkdir Results\%%r\WS_%%w\NC_%%c\Other\Test\ClashRoyale
			mkdir Results\%%r\WS_%%w\NC_%%c\Other\Test\RocketLeague
			
			call event_log_extraction %%w %%c
			
			xcopy Output\ELE\ProcessMining\Training Results\%%r\WS_%%w\NC_%%c\ProcessMining\Training\EventLogs /E
			xcopy Output\ELE\ProcessMining\Validation Results\%%r\WS_%%w\NC_%%c\ProcessMining\Validation\EventLogs /E
			xcopy Output\ELE\ProcessMining\Test\ClashRoyale Results\%%r\WS_%%w\NC_%%c\ProcessMining\Test\EventLogs\ClashRoyale /E
			xcopy Output\ELE\ProcessMining\Test\RocketLeague Results\%%r\WS_%%w\NC_%%c\ProcessMining\Test\EventLogs\RocketLeague /E
			
			copy Output\ELE\Other\Training\* Results\%%r\WS_%%w\NC_%%c\Other\Training\
			copy Output\ELE\Other\Validation\* Results\%%r\WS_%%w\NC_%%c\Other\Validation\
			copy Output\ELE\Other\Test\ClashRoyale\* Results\%%r\WS_%%w\NC_%%c\Other\Test\ClashRoyale
			copy Output\ELE\Other\Test\RocketLeague\* Results\%%r\WS_%%w\NC_%%c\Other\Test\RocketLeague
			
			for /D %%p IN ("Input\PD\EventLogs\*") DO (
				del /s /f /q %%p\*.*
				for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				rmdir "%%p" /s /q
			)
			xcopy Output\ELE\ProcessMining\Training Input\PD\EventLogs /E
			
			call process_discovery %noise_threshold%
			
			xcopy Output\PD\PetriNets Results\%%r\WS_%%w\NC_%%c\ProcessMining\Training\PetriNets /E
		)
		
	)
)	







