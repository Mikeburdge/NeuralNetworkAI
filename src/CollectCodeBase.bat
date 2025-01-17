@echo off
REM Change to the directory where the batch file is located
cd /d "%~dp0"

REM Define the output file name
set "OUTPUT_FILE=Full Codebase.txt"

REM If you want to start fresh each time, uncomment the following lines:
REM if exist "%OUTPUT_FILE%" del "%OUTPUT_FILE%"
REM echo Starting to collect codebase... > "%OUTPUT_FILE%"

REM Initialize the output file (optional: remove if not deleting existing file)
echo Starting to collect codebase... > "%OUTPUT_FILE%"

REM Loop through all .cpp and .h files recursively in the directory
for /R %%f in (*.cpp *.h) do (
    REM Add header separators and the filename to the output file
    echo -------------------------------------------------------------------------->> "%OUTPUT_FILE%"
    echo  %%~nxf>> "%OUTPUT_FILE%"
    echo -------------------------------------------------------------------------->> "%OUTPUT_FILE%"
    
    REM Append the contents of the current file to the output file
    type "%%f" >> "%OUTPUT_FILE%"
    
    REM Add a blank line for readability
    echo.>> "%OUTPUT_FILE%"
)

echo.
echo All .cpp and .h files have been successfully appended to "%OUTPUT_FILE%".
echo.
pause
