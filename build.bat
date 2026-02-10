@echo off
echo === EZ-FBX2VRM Build Script ===
echo.

echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

echo.
echo Building executable...
pyinstaller build.spec --clean

echo.
echo Build complete! Check the dist/ folder for EZ-FBX2VRM.exe
pause
