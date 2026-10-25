@echo off
cd /d "%~dp0"
if exist "D:\Python313\python.exe" (
  "D:\Python313\python.exe" "%~dp0visualize.py"
) else (
  python "%~dp0visualize.py"
)
