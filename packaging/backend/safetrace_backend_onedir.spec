# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-dir spec for the SafeTrace backend.

The one-dir layout avoids PyInstaller one-file self-extraction into a temporary
_MEI directory. Local models, checkpoints, data, generated media, logs, and
frontend assets remain external package assets beside the runtime.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules


repo_root = Path(SPECPATH).parents[1]
llama_cpp_binaries = collect_dynamic_libs("llama_cpp")
# AutoProcessor resolves SmolVLM's image and video processor classes lazily from
# the model config.  PyInstaller sees the base SmolVLM imports, but not the
# video processor module, which made packaged local VLM workers fail before
# generation.  Collect the model family rather than relying on that runtime
# lookup.
smolvlm_hiddenimports = collect_submodules("transformers.models.smolvlm")

block_cipher = None

a = Analysis(
    [str(repo_root / "src" / "api" / "__main__.py")],
    pathex=[str(repo_root)],
    binaries=llama_cpp_binaries,
    datas=[],
    hiddenimports=[
        "src.api.server",
        "src.lightweight_vlm_worker",
        "src.lightweight_vlm_worker_client",
        "src.mask_encoding",
        "src.mobile_sam_worker",
        "src.mobile_sam_worker_client",
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "llama_cpp",
        "llama_cpp.llama",
        "llama_cpp.llama_cpp",
        "llama_cpp.llama_cpp_ext",
        "mobile_sam",
        "transformers",
        "transformers.models.auto.processing_auto",
        "transformers.models.auto.modeling_auto",
        "transformers.processing_utils",
        "transformers.video_processing_utils",
        "transformers.models.smolvlm.video_processing_smolvlm",
        *smolvlm_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="safetrace-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="safetrace-backend",
)
