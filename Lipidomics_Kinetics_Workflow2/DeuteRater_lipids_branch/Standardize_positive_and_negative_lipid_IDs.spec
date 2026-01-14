# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['H:\\dist\\Lipidomics_Kinetics_Workflow2\\Standardize_positive_and_negative_lipid_IDs\\Standardize_positive_and_negative_lipid_IDs.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['jupyter', 'jupyterlab', 'jupyterlab_server', 'jupyter_server', 'jupyter_client', 'jupyter_core', 'notebook', 'nbformat', 'nbconvert', 'ipykernel', 'ipywidgets', 'traitlets', 'tornado', 'qtconsole', 'comm', 'debugpy'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Standardize_positive_and_negative_lipid_IDs',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='Standardize_positive_and_negative_lipid_IDs',
)
