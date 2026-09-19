<#
.SYNOPSIS
    CARIVIX-AI API & Backend Integration Test Runner Wrapper

.DESCRIPTION
    Launches automated integration tests across Backend Data Service and WebGIS Spatial Service.
    Automatically handles environment verification, port hygiene, and HTML report generation.

.EXAMPLE
    .\run_api_tests.ps1
    .\run_api_tests.ps1 -Suite backend
    .\run_api_tests.ps1 -Suite gis
#>

param (
    [string]$Suite = "all",
    [switch]$CleanPorts,
    [switch]$NoReport
)

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$Runner = Join-Path $PSScriptRoot "run_api_tests.py"

if (-not (Test-Path $VenvPython)) {
    Write-Host "❌ Error: Virtual environment not found at $VenvPython" -ForegroundColor Red
    exit 1
}

$ArgsList = @($Runner, "--suite", $Suite)
if ($CleanPorts) { $ArgsList += "--clean-ports" }
if ($NoReport) { $ArgsList += "--no-report" }

& $VenvPython $ArgsList
