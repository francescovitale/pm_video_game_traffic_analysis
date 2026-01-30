:: Useful commands:

:: cd <directory>
:: copy path\to\file destination\path
:: xcopy path\to\dirs destination\path /E
:: rmdir "<dir_name>" /s /q
:: ren path\to\file <name>
:: del /F /Q path\to\file 

set n_reps=1 2 3 4 5
set ws_conf=WS_2 WS_3 WS_4 WS_5 WS_6 WS_7 WS_8
set nc_conf=NC_2 NC_3 NC_4 NC_5
set trace_fraction=0.01

for /D %%p IN ("AnalysisResults\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for %%r in (%n_reps%) do (

	mkdir AnalysisResults\%%r
	mkdir AnalysisResults\%%r\ProcessMining

	for %%w in (%ws_conf%) do (

		for %%c in (%nc_conf%) do (

			for /D %%p IN ("Classification\Input\*") DO (
				del /s /f /q %%p\*.*
				for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
				rmdir "%%p" /s /q
			)
			del /F /Q Classification\Output\*
			
			
			xcopy FrameworkResults\%%r\%%w\%%c\ProcessMining Classification\Input /E
			
			cd Classification
			python evaluation_processmining.py %trace_fraction%
			cd ..
			
			copy Classification\Output\roc_curve.csv AnalysisResults\%%r\ProcessMining
			ren AnalysisResults\%%r\ProcessMining\roc_curve.csv roc_curve_%%w_%%c.csv
			copy Classification\Output\Metrics.txt AnalysisResults\%%r\ProcessMining
			ren AnalysisResults\%%r\ProcessMining\Metrics.txt Metrics_%%w_%%c.txt
		)

	)
)	

	








