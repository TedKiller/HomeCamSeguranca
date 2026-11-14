@echo off
chcp 65001 >nul
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual .venv nao encontrado em %~dp0.
    pause
    exit /b 1
)
echo Iniciando HomeCam Seguranca em modo Servidor 24/7 Headless...
echo Portal Web disponivel em: http://localhost:8765/
".venv\Scripts\python.exe" main.py --headless --web-port 8765
if errorlevel 1 (
    echo [ERRO] O servidor headless encerrou com codigo de falha.
    pause
)
popd

