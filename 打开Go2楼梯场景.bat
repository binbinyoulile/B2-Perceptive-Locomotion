@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0simulations\mujoco\go2_stairs_static"
if exist "D:\Python313\python.exe" (
  "D:\Python313\python.exe" "%~dp0simulations\mujoco\go2_stairs_static\visualize.py"
) else (
  python "%~dp0simulations\mujoco\go2_stairs_static\visualize.py"
)
if errorlevel 1 pause