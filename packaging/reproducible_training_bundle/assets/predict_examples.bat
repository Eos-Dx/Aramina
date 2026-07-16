@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0predict_examples.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
pause
exit /b %EXIT_CODE%
