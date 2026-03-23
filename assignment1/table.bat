@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PYTHON_EXE=C:\Users\71766\miniforge3\envs\ntt_jax\python.exe"
set "BATCH=4"

if not exist "%PYTHON_EXE%" (
  echo Python executable not found: %PYTHON_EXE%
  exit /b 1
)

for %%C in (
  yf3005_1_CooleyTukey
  hy3281_1
  qz2866_1
  montgomery
  stockham
) do (
  echo ========================================
  echo CHOICE=%%C
  echo ========================================
  echo.

  set "CHOICE=%%C"

  for %%L in (12 13 14 15) do (
    echo [Benchmark] CHOICE=%%C logn=%%L batch=%BATCH%
    "%PYTHON_EXE%" -m tests.benchmark --bench --logn %%L --batch %BATCH%
    if errorlevel 1 exit /b !errorlevel!
    echo.
  )
)

endlocal
