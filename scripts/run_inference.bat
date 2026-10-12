@echo off
REM Wrapper: activates the sleap conda env, then runs the inference script.
REM Usage: scripts\run_inference.bat --videos-root data\cohort_12
REM        (any run_inference.py option can be added, e.g. --force)

call conda activate sleap
if errorlevel 1 (
    echo Could not activate the 'sleap' conda env. See README.md for setup.
    exit /b 1
)

python "%~dp0run_inference.py" %*
