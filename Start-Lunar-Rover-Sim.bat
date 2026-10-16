@echo off
title 3D Autonomous Lunar Rover Simulator Launcher
color 0b

echo =========================================================================
echo    NAKSHATRA AEROSPACE - AUTONOMOUS LUNAR ROVER MISSION LAUNCHER
echo    RTX 5060 (Qwen2-VL-2B) + Dual-AI Consensus and 3D WebGL Simulator
echo =========================================================================
echo.

REM 1. Check and Launch Local RTX 5060 VLM/LLM Server (Port 8080)
netstat -ano | findstr :8080 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo [+] Local Qwen2-VL Server is ALREADY active on port 8080.
) else (
    echo [*] Starting Local Qwen2-VL-2B Server on RTX 5060 GPU on port 8080...
    start "Qwen2-VL RTX 5060 Server (Port 8080)" /min "C:\Users\ckceg\llama.cpp\llama-server.exe" -m "C:\Users\ckceg\llama.cpp\models\Qwen2-VL-2B-Instruct-Q4_K_M.gguf" --mmproj "C:\Users\ckceg\llama.cpp\models\mmproj-Qwen2-VL-2B-Instruct-f16.gguf" -ngl 99 --port 8080 --host 127.0.0.1 -c 4096
    ping 127.0.0.1 -n 4 >nul
)

REM 2. Check and Launch Lightweight Simulator Web Server (Port 8000)
netstat -ano | findstr :8000 | findstr LISTENING >nul
if %errorlevel% equ 0 (
    echo [+] Simulation Web Server is ALREADY active on port 8000.
) else (
    echo [*] Starting Local Simulator Web Server on port 8000...
    start "Lunar Rover Web Server (Port 8000)" /min python -m http.server --directory "C:\Users\ckceg\.gemini\antigravity\scratch\lunar_rover_sim" 8000
    ping 127.0.0.1 -n 3 >nul
)

REM 3. Launch Default Browser to Interactive 3D Simulator
echo [*] Opening 3D Autonomous Lunar Rover Simulator in default browser...
start "" "http://127.0.0.1:8000/lunar_rover_interactive_sim.html"

echo.
echo =========================================================================
echo    [SUCCESS] Lunar Rover Expedition environment is ACTIVE!
echo    - 3D WebGL Simulator: http://127.0.0.1:8000
echo    - Local VLM (RTX 5060): http://127.0.0.1:8080
echo    - Continuous 2000-Win Auto-Loop and Continual Learning: READY
echo =========================================================================
echo.
echo Launch complete! Window closing in 4 seconds...
ping 127.0.0.1 -n 5 >nul
exit /b 0
