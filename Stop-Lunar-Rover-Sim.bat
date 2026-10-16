@echo off
title Stop Lunar Rover Simulator and Background Servers
color 0c

echo =========================================================================
echo    STOPPING LUNAR ROVER SIMULATOR AND BACKGROUND AI SERVERS
echo =========================================================================
echo.

echo [*] Terminating llama-server.exe (Qwen2-VL)...
taskkill /F /IM llama-server.exe /T 2>nul
if %errorlevel% equ 0 (
    echo [+] llama-server.exe stopped successfully.
) else (
    echo [-] llama-server.exe was not running.
)

echo [*] Terminating Python Web Server on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a 2>nul
    echo [+] Stopped process with PID %%a on port 8000.
)

echo.
echo =========================================================================
echo    [DONE] All Lunar Rover background services have been stopped.
echo =========================================================================
echo.
echo Closing in 3 seconds...
ping 127.0.0.1 -n 4 >nul
exit /b 0
