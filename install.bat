@echo off
setlocal

set SCRIPT_DIR=%~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install.ps1" %*

if errorlevel 1 (
  echo Aramis installation failed.
  exit /b 1
)

echo Aramis installation finished.
