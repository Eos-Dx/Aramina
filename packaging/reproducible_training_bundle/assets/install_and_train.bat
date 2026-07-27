@echo off
setlocal

echo Starting Aramina reproducible training bundle...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_and_train.ps1" %*
if errorlevel 1 (
  echo Aramina reproducibility bundle failed.
  exit /b 1
)

echo Aramina reproducibility bundle completed.
