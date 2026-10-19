# dot-source this file in PowerShell:  . .\env.ps1
$env:PIPELINE_ROOT = $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot;$PSScriptRoot\formfinding;$PSScriptRoot\vares;$PSScriptRoot\abaqus;$PSScriptRoot\figures;$env:PYTHONPATH"
if (-not $env:PIPELINE_DATA) { $env:PIPELINE_DATA = Join-Path $PSScriptRoot "data" }
Write-Host "PIPELINE_ROOT=$env:PIPELINE_ROOT  PIPELINE_DATA=$env:PIPELINE_DATA"
