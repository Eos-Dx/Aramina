@echo off
setlocal

echo Starting Aramis reproducible training bundle...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_and_train.ps1" %*
if errorlevel 1 (
  echo Aramis reproducibility bundle failed.
  exit /b 1
)

echo Aramis reproducibility bundle completed.
