@echo off
rem ==========================================================
rem  AI Vision Tracker & Pan-Tilt Servoing - Windows Launcher
rem ==========================================================
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
