@echo off
REM Windows batch script to run Python with TFMOT environment variables
REM Usage: run_with_tfmot.bat script_name.py

REM Set TFMOT environment variables
set TF_USE_LEGACY_KERAS=1
set KERAS_BACKEND=tensorflow
set TF_CPP_MIN_LOG_LEVEL=1

echo ✅ TFMOT environment variables set
echo    TF_USE_LEGACY_KERAS=%TF_USE_LEGACY_KERAS%
echo    KERAS_BACKEND=%KERAS_BACKEND%
echo.

REM Run the Python script with the provided argument
if "%1"=="" (
    echo Usage: run_with_tfmot.bat script_name.py
    echo Example: run_with_tfmot.bat quick_tfmot_test.py
    pause
    exit /b 1
)

echo Running: python %*
python %*

pause 