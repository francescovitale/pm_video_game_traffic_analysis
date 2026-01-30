:: Useful commands:

:: cd <directory>
:: copy path\to\file destination\path
:: xcopy path\to\dirs destination\path /E
:: rmdir "<dir_name>" /s /q
:: ren path\to\file <name>
:: del /F /Q path\to\file 

:: Options:

:: variant=[im, ilp]
:: noise_threshold=<float>

set variant=im
set noise_threshold=%1

for /D %%p IN ("Output\PD\PetriNets\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

python process_discovery.py %variant% %noise_threshold% 
	







