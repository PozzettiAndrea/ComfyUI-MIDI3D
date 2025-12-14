#!/usr/bin/env python
# MIT License
# Copyright (c) 2025 ComfyUI-MIDI3D Contributors

"""
Install script for ComfyUI-MIDI3D dependencies.

Automatically detects PyTorch version and CUDA version to install
the correct torch-cluster wheel.

Usage:
    python install.py
"""

import subprocess
import sys


def get_torch_info():
    """Get PyTorch version and CUDA suffix."""
    import torch

    torch_version = torch.__version__.split("+")[0]

    if torch.cuda.is_available() and torch.version.cuda:
        cuda_suffix = "cu" + torch.version.cuda.replace(".", "")
    else:
        cuda_suffix = "cpu"

    return torch_version, cuda_suffix


def install_torch_cluster():
    """Install torch-cluster from PyG wheel index."""
    torch_version, cuda_suffix = get_torch_info()

    wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html"

    print(f"[MIDI-3D] PyTorch {torch_version} with {cuda_suffix}")
    print(f"[MIDI-3D] Installing torch-cluster from {wheel_url}")

    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch-cluster",
        "-f", wheel_url,
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("[MIDI-3D] torch-cluster installed successfully")
    else:
        print("[MIDI-3D] Failed to install torch-cluster")
        sys.exit(1)


def install_nvdiffrast():
    """Install nvdiffrast from sam3dobjects-wheels."""
    import sys as _sys
    torch_version, cuda_suffix = get_torch_info()

    # Get Python version for wheel selection
    py_ver = f"cp{_sys.version_info.major}{_sys.version_info.minor}"

    # Direct wheel URL from sam3dobjects-wheels releases
    wheel_url = (
        f"https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/"
        f"nvdiffrast-{cuda_suffix}/nvdiffrast-0.4.0%2B{cuda_suffix}-{py_ver}-{py_ver}-linux_x86_64.whl"
    )

    print(f"[MIDI-3D] Installing nvdiffrast from {wheel_url}")

    cmd = [
        sys.executable, "-m", "pip", "install",
        wheel_url,
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("[MIDI-3D] nvdiffrast installed successfully")
    else:
        print("[MIDI-3D] Failed to install nvdiffrast from wheels, trying from source...")
        # Fallback to pip install from GitHub with --no-build-isolation
        cmd_fallback = [
            sys.executable, "-m", "pip", "install",
            "--no-build-isolation",
            "git+https://github.com/NVlabs/nvdiffrast.git",
        ]
        result = subprocess.run(cmd_fallback)
        if result.returncode == 0:
            print("[MIDI-3D] nvdiffrast installed from source")
        else:
            print("[MIDI-3D] WARNING: Failed to install nvdiffrast (texturing will not work)")


def install_mvadapter():
    """Install MV-Adapter from GitHub."""
    print(f"[MIDI-3D] Installing mvadapter from GitHub...")

    cmd = [
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/huanngzh/MV-Adapter.git",
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("[MIDI-3D] mvadapter installed successfully")
    else:
        print("[MIDI-3D] WARNING: Failed to install mvadapter (texturing will not work)")


def install_other_deps():
    """Install other required dependencies."""
    deps = [
        "trimesh",
        "scikit-image",
        "peft",
        "einops",
        "omegaconf",
    ]

    print(f"[MIDI-3D] Installing: {', '.join(deps)}")

    cmd = [sys.executable, "-m", "pip", "install"] + deps
    subprocess.run(cmd)


def install_texture_deps():
    """Install dependencies for texturing (optional)."""
    print("[MIDI-3D] Installing texture dependencies...")
    install_nvdiffrast()
    install_mvadapter()


if __name__ == "__main__":
    print("[MIDI-3D] Installing dependencies...")
    install_torch_cluster()
    install_other_deps()

    # Texturing deps (optional but recommended)
    install_texture_deps()

    print("[MIDI-3D] Done!")
