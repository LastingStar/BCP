# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the desktop launcher.

Output:
  dist/DroneDesktopLauncher/DroneDesktopLauncher.exe
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = Path(SPECPATH).resolve()

# Keep the original Streamlit script as a real file for streamlit.bootstrap.run().
datas = [
    (str(project_root / "ui" / "drone_ui.py"), "ui"),
    (str(project_root / "Bernese_Oberland_46.6241_8.0413.png"), "."),
]

# Optionally include default stage-3 best model if it exists.
default_model = project_root / "models" / "ppo_drone_stage3_obs31_run1_best" / "best_model.zip"
if default_model.exists():
    datas.append((str(default_model), "models/ppo_drone_stage3_obs31_run1_best"))

# Streamlit uses dynamic assets; collecting data improves runtime stability.
datas += collect_data_files("streamlit")

hiddenimports = []
hiddenimports += collect_submodules("streamlit")
hiddenimports += collect_submodules("webview")
hiddenimports += ["launcher"]


a = Analysis(
    ["desktop_launcher.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DroneDesktopLauncher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DroneDesktopLauncher",
)
