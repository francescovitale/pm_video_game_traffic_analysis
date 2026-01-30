:: Useful commands:

:: cd <directory>
:: copy path\to\file destination\path
:: xcopy path\to\dirs destination\path /E
:: rmdir "<dir_name>" /s /q
:: ren path\to\file <name>
:: del /F /Q path\to\file 

set n_reps=1 2 3 4 5
set ws_conf=WS_3
set nc_conf=NC_3
set trace_fraction=0.01
set ml_classifiers=iforest hbos zscore pca copod

for /D %%p IN ("AnalysisResults\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for %%r in (%n_reps%) do (

	mkdir AnalysisResults\%%r
	mkdir AnalysisResults\%%r\Other


	for /D %%p IN ("Classification\Input\*") DO (
		del /s /f /q %%p\*.*
		for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
		rmdir "%%p" /s /q
	)
	del /F /Q Classification\Output\*
			
	REM other
			
	xcopy FrameworkResults\%%r\%ws_conf%\%nc_conf%\Other Classification\Input /E
			
	for %%c in (%ml_classifiers%) do (	
	
		cd Classification
		python evaluation_other.py %%c %trace_fraction%
		cd ..
				
		copy Classification\Output\roc_curve.csv AnalysisResults\%%r\Other
		ren AnalysisResults\%%r\Other\roc_curve.csv roc_curve_%%c.csv
		copy Classification\Output\Metrics.txt AnalysisResults\%%r\Other
		ren AnalysisResults\%%r\Other\Metrics.txt Metrics_%%c.txt
	)	

	
)	

	








