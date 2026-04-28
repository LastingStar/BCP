@echo off
setlocal

cd /d "%~dp0"

echo [1/3] Installing packaging dependencies...
python -m pip install --upgrade pyinstaller pywebview streamlit
if errorlevel 1 goto :error

echo [2/3] Building desktop executable via PyInstaller...
python -m PyInstaller --noconfirm --clean pyinstaller_desktop.spec
if errorlevel 1 goto :error

echo [3/3] Build completed.
echo Output: "%cd%\dist\DroneDesktopLauncher\DroneDesktopLauncher.exe"
exit /b 0

:error
echo Build failed with code %errorlevel%.
exit /b %errorlevel%
