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
:: n_min_variants=<integer>
:: n_max_variants=<integer>
:: n_max_iterations=<integer>

set fe_window_size=%1
set n_clusters=%2
set normalization_type=zscore
set clustering_type=kmeans

for /D %%p IN ("Input\PD\EventLogs\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for /D %%p IN ("Output\ELE\ProcessMining\Training\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
for /D %%p IN ("Output\ELE\ProcessMining\Validation\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
for /D %%p IN ("Output\ELE\ProcessMining\Test\ClashRoyale\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
for /D %%p IN ("Output\ELE\ProcessMining\Test\RocketLeague\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)


del /F /Q Output\ELE\Other\Training\*
del /F /Q Output\ELE\Other\Validation\*
del /F /Q Output\ELE\Other\Test\ClashRoyale\*
del /F /Q Output\ELE\Other\Test\RocketLeague\*




	
python event_log_extraction.py %fe_window_size% %normalization_type% %clustering_type% %n_clusters%
	

	








