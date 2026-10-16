@echo off
title Download Qwen2-VL-2B Models for Lunar Rover Simulator
color 0b

echo =========================================================================
echo    DOWNLOADING QWEN2-VL-2B MULTIMODAL MODELS (FOR RTX 5060 GPU)
echo =========================================================================
echo.

python download_models.py

echo.
pause
