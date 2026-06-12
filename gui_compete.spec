# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['gui_compete.py'],
    pathex=[],
    binaries=[
        ('stockfish/stockfish-windows-x86-64-avx2.exe', 'stockfish'),
    ],
    datas=[
        ('models', 'models'),
        ('syzygy', 'syzygy'),
        ('kits', 'kits'),
    ],
    hiddenimports=[
        'onnxruntime',
        'chess',
        'chess.engine',
        'chess.syzygy',
        'numpy',
        'tkinter',
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
    a.binaries,
    a.datas,
    name='gui_compete',
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
