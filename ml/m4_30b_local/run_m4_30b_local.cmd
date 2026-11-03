@echo off
REM Launches the repair run when PowerShell's execution policy blocks .ps1 files.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_m4_30b_local.ps1" %*
