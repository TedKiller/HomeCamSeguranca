<#
.SYNOPSIS
    Registra o HomeCam Seguranca como Tarefa Agendada 24/7 no Windows.
.DESCRIPTION
    Executa main.py --headless em segundo plano (pythonw.exe), iniciando automaticamente
    e reiniciando em caso de falha, sem janela aberta e mesmo com a tela bloqueada (Win+L).
#>
param(
    [int]$WebPort = 8765,
    [switch]$AtStartupWithPassword,
    [switch]$Uninstall
)

$TaskName = "HomeCamSeguranca24h"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$PythonW = Join-Path $ProjectDir ".venv\Scripts\pythonw.exe"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Tarefa '$TaskName' removida com sucesso."
    exit 0
}

if (-not (Test-Path $PythonW)) {
    Write-Error "pythonw.exe nao encontrado em $PythonW"
    exit 1
}

$ArgsLine = "main.py --headless --web-port $WebPort"
$CurrentUser = "$env:USERDOMAIN\$env:USERNAME"

if ($AtStartupWithPassword) {
    # Modo Boot antes mesmo de fazer login no Windows, preservando DPAPI e Drive OAuth do usuario
    schtasks /Create /F /TN $TaskName /SC ONSTART /RU "$CurrentUser" /RP * /RL HIGHEST `
        /TR "`"$PythonW`" $ArgsLine"
    Write-Host "Tarefa '$TaskName' instalada para iniciar no Boot (com credencial do usuario para DPAPI/Drive)."
    exit 0
}

$Action = New-ScheduledTaskAction -Execute $PythonW -Argument $ArgsLine -WorkingDirectory $ProjectDir
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable
$Principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Motor NVR 24/7, IA ALPR/Pessoas e Portal Web do HomeCam Seguranca." `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Tarefa '$TaskName' instalada e iniciada!"
Write-Host "Acesse o Portal Web em: http://localhost:$WebPort/"
Write-Host "Dica: execute .\configurar_energia_24h.ps1 e use Win+L para desligar o monitor sem interromper a gravacao."

