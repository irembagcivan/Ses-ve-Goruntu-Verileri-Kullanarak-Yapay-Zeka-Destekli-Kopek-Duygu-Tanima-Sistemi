@echo off
cd /d "%~dp0"
if defined CANINE_PYTHON (
  "%CANINE_PYTHON%" -m src.demo.launch
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m src.demo.launch
) else (
  py -3.11 -m src.demo.launch
)
if errorlevel 1 pause
