<#
.SYNOPSIS
    Configura o gerenciamento de energia do Windows para operacao 24/7 com tela desligada.
.DESCRIPTION
    Mantem CPU, GPU, disco e placa de rede ativos continuamente na tomada (AC),
    permitindo apenas que o monitor desligue apos MonitorOffMinutes minutos.
#>
param(
    [int]$MonitorOffMinutes = 10,
    [switch]$Revert
)

$ErrorActionPreference = "SilentlyContinue"

if ($Revert) {
    Write-Host "Restaurando politicas padrao de energia (modo Desktop)..."
    powercfg /change standby-timeout-ac 30
    powercfg /change hibernate-timeout-ac 0
    powercfg /change monitor-timeout-ac 15
    powercfg /change disk-timeout-ac 20
    Write-Host "Configuracao padrao restaurada."
    exit 0
}

Write-Host "Aplicando perfil de energia NVR 24/7 (somente monitor desliga apos $MonitorOffMinutes min)..."

# Nao suspender nem hibernar computador conectado na tomada (AC)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change disk-timeout-ac 0
powercfg /change monitor-timeout-ac $MonitorOffMinutes

# Evita que USB (webcam/adaptadores) e PCIe (GPU/NVMe) entrem em economia agressiva na tomada
powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_USB USBSELECTIVE 0
powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0
powercfg /SETACTIVE SCHEME_CURRENT

# Mantem placas de rede ativas (requer PowerShell como Administrador para gravar no driver)
Get-NetAdapter -Physical | Where-Object { $_.Status -eq "Up" } | ForEach-Object {
    Disable-NetAdapterPowerManagement -Name $_.Name -WakeOnMagicPacket:$false -ErrorAction SilentlyContinue
    Set-NetAdapterPowerManagement -Name $_.Name -SelectiveSuspend Disabled -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Resumo atual do esquema de energia:"
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
powercfg /query SCHEME_CURRENT SUB_VIDEO VIDEOIDLE
Write-Host ""
Write-Host "Pronto! Pode bloquear a tela (Win+L) ou desligar o monitor: a gravacao e a IA continuam ativas."

