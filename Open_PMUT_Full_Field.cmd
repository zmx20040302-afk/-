@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0PMUT_Workbench.ps1" %*
if errorlevel 1 (
  echo PMUT workbench startup failed.
  if "%~1"=="" pause
  exit /b 1
)
