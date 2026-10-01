# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

mediapipe_datas, mediapipe_binaries, mediapipe_hiddenimports = collect_all("mediapipe")


a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=mediapipe_binaries,
    datas=[
        ('models/pose_landmarker_full.task', 'models'),
        *mediapipe_datas,
    ],
    hiddenimports=[
        *mediapipe_hiddenimports,
        'mediapipe.tasks.c',
    ],
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
    name='PostureGuard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PostureGuard',
)