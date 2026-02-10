@echo off
echo === EZ-FBX2VRM Environment Setup ===
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo === Assimp Library Setup ===
echo.
echo pyassimp requires the Assimp shared library.
echo.
echo Option A (Recommended): Install via vcpkg:
echo   vcpkg install assimp:x64-windows
echo.
echo Option B: Download from https://github.com/assimp/assimp/releases
echo   and place assimp.dll in this directory or in your PATH.
echo.
echo Option C: Install via conda:
echo   conda install -c conda-forge assimp
echo.

echo Setup complete!
pause
