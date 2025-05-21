@echo off
REM Set the model name
set MODEL=unet
set MAIN_LOG_DIR=logs\%MODEL%

REM Create a main log directory if it doesn't exist
mkdir "%MAIN_LOG_DIR%"

REM Create a subdirectory for this job's logs with a timestamp
set JOB_LOG_DIR=%MAIN_LOG_DIR%\%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
mkdir "%JOB_LOG_DIR%"

REM Activate your conda environment
call conda activate dl

REM Print the current task information
echo Now processing task on %COMPUTERNAME% at %date% %time%

REM Run your python script and redirect output to a file within the job's log directory
python main.py --logdir "%JOB_LOG_DIR%" > "%JOB_LOG_DIR%\output.txt"

REM Print completion message
echo Finished processing task
exit /b
