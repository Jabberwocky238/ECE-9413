@echo off
setlocal enabledelayedexpansion

set ENV_NAME=ntt_jax

conda env list | findstr /B "%ENV_NAME% " >nul 2>&1
if %errorlevel% equ 0 (
    echo Environment %ENV_NAME% already exists. Activating...
    call conda activate %ENV_NAME%
) else (
    echo Creating conda environment: %ENV_NAME%
    call conda create -n %ENV_NAME% python=3.11 -y
    call conda activate %ENV_NAME%
)

set extra=
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader') do set driver=%%i
    for /f "tokens=1 delims=." %%a in ("!driver!") do set major=%%a
    if !major! geq 580 (
        set extra=cuda13
    ) else if !major! geq 525 (
        set extra=cuda12
    )
)

if defined extra (
    echo Installing JAX with %extra%
    pip install "jax[%extra%]" numpy rich pytest sympy
) else (
    echo Installing CPU-only JAX
    pip install "jax[cpu]" numpy rich pytest sympy
)

echo Setup complete. Activate with: conda activate %ENV_NAME%
