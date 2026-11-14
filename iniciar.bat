@echo off
chcp 65001 >nul
pushd "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "main.py"
) else (
    ".venv\Scripts\python.exe" "main.py"
)
popd

