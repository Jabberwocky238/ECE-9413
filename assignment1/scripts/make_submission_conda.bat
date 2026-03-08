@echo off
setlocal

cd /d "%~dp0\.."

call conda activate ntt_jax

echo Running public tests...
pytest

set OUT=code.zip
if exist "%OUT%" del "%OUT%"

echo Creating %OUT%...
powershell -Command "Compress-Archive -Path student.py -DestinationPath %OUT% -Force"

echo Done: %OUT%
echo Upload code.zip to Brightspace.
