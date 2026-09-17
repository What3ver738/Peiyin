@echo off
setlocal
cd /d "%~dp0.."
set "PYBIN="
for %%v in (3.12 3.11 3.10) do (
  if not defined PYBIN (
    py -%%v -c "import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))" >nul 2>&1
    if not errorlevel 1 set "PYBIN=py -%%v"
  )
)
if not defined PYBIN (
  python -c "import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] < (3, 13)))" >nul 2>&1
  if not errorlevel 1 set "PYBIN=python"
)
if not defined PYBIN (
  echo Install Python 3.12, then run this launcher again. Supported: 3.10-3.12.
  pause
  exit /b 1
)
if not exist venv (
  %PYBIN% -m venv venv
  if errorlevel 1 exit /b 1
)
venv\Scripts\python.exe -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 'Rename the unsupported venv and run this launcher again.')"
if errorlevel 1 (
  pause
  exit /b 1
)
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  pause
  exit /b 1
)
venv\Scripts\python.exe -m peiyin
pause
