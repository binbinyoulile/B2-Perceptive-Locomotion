@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if exist "D:\Python313\python.exe" (
  "D:\Python313\python.exe" "%~dp0scripts\check_environment.py"
) else (
  python "%~dp0scripts\check_environment.py"
)
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%