:: Useful commands:

:: cd <directory>
:: copy path\to\file destination\path
:: xcopy path\to\dirs destination\path /E
:: rmdir "<dir_name>" /s /q
:: ren path\to\file <name>
:: del /F /Q path\to\file 

set ws_conf=WS_6 WS_7 WS_8
set nc_conf=NC_2 NC_3 NC_4 NC_5
set trace_fraction=0.01

::del /F /Q AnalysisResults\*

for %%w in (%ws_conf%) do (

	for %%c in (%nc_conf%) do (

		for /D %%p IN ("Classification\Input\*") DO (
			del /s /f /q %%p\*.*
			for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
			rmdir "%%p" /s /q
		)
		del /F /Q Classification\Output\*
		
		xcopy FrameworkResults\%%w\%%c Classification\Input /E
		
		cd Classification
		python evaluation.py %trace_fraction%
		cd ..
		
		
		copy Classification\Output\roc_curve.csv AnalysisResults
		ren AnalysisResults\roc_curve.csv roc_curve_%%w_%%c.csv
		copy Classification\Output\Metrics.txt AnalysisResults
		ren AnalysisResults\Metrics.txt Metrics_%%w_%%c.txt
	)

)

	








