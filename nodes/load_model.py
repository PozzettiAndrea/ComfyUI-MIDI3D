"""DownloadAndLoadMIDI3DModel node."""

import torch
from pathlib import Path
from typing import Any, Dict

from .utils import (
    _MODEL_CACHE,
    get_midi3d_path,
    get_midi3d_models_path,
    setup_midi3d_imports,
    get_device,
)


class DownloadAndLoadMIDI3DModel:
    """
    Download (if needed) and load the MIDI-3D pipeline.

    MIDI-3D generates compositional 3D scenes from a single image
    with instance segmentation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["MIDI-3D"], {
                    "default": "MIDI-3D",
                    "tooltip": "MIDI-3D model for multi-instance 3D scene generation"
                }),
            },
            "optional": {
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision (bfloat16 recommended for RTX 30xx+)"
                }),
            }
        }

    RETURN_TYPES = ("MIDI3D_MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("MIDI-3D pipeline for scene generation",)
    FUNCTION = "load_model"
    CATEGORY = "MIDI3D"
    DESCRIPTION = "Download and load MIDI-3D model for multi-instance 3D scene generation from images."

    def load_model(self, model: str, dtype: str = "bfloat16"):
        """Load the MIDI-3D pipeline."""
        print(f"[MIDI-3D] Loading model: {model}")

        device = get_device()

        # Check CUDA
        if device.type == "cpu":
            print("[MIDI-3D] WARNING: CUDA not available, running on CPU will be very slow!")

        # Get dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # Cache key
        cache_key = f"{model}_{dtype}"

        if cache_key in _MODEL_CACHE:
            print(f"[MIDI-3D] Using cached model")
            return (_MODEL_CACHE[cache_key],)

        # Download if needed
        model_path = self._get_or_download_model()

        # Setup imports
        setup_midi3d_imports()

        # Load pipeline
        from midi.pipelines.pipeline_midi import MIDIPipeline

        print(f"[MIDI-3D] Loading pipeline from {model_path}")
        pipe = MIDIPipeline.from_pretrained(str(model_path)).to(device, torch_dtype)

        # Initialize custom adapter for multi-instance attention
        pipe.init_custom_adapter(
            set_self_attn_module_names=[
                "blocks.8",
                "blocks.9",
                "blocks.10",
                "blocks.11",
                "blocks.12",
            ]
        )

        # Create model wrapper
        model_wrapper = {
            "pipe": pipe,
            "device": device,
            "dtype": torch_dtype,
        }

        _MODEL_CACHE[cache_key] = model_wrapper
        print(f"[MIDI-3D] Model loaded successfully")

        return (model_wrapper,)

    @classmethod
    def _get_or_download_model(cls) -> Path:
        """Get model path, downloading if necessary."""
        models_dir = get_midi3d_models_path()
        model_path = models_dir / "MIDI-3D"

        # Also check in MIDI-3D source folder
        midi3d_path = get_midi3d_path()
        source_model_path = midi3d_path / "pretrained_weights" / "MIDI-3D"

        if source_model_path.exists() and (source_model_path / "config.json").exists():
            print(f"[MIDI-3D] Found model in source: {source_model_path}")
            return source_model_path

        if model_path.exists() and (model_path / "config.json").exists():
            print(f"[MIDI-3D] Found model: {model_path}")
            return model_path

        # Download from HuggingFace
        print(f"[MIDI-3D] Downloading model from HuggingFace...")
        cls._download_model(model_path)

        if not (model_path / "config.json").exists():
            raise RuntimeError(f"Download completed but config.json not found: {model_path}")

        return model_path

    @classmethod
    def _download_model(cls, target_dir: Path):
        """Download model from HuggingFace."""
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download

            repo_id = "VAST-AI/MIDI-3D"

            print(f"[MIDI-3D] Downloading from {repo_id} (this may take a while)...")

            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

            print(f"[MIDI-3D] Download complete")

        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading models. "
                "Please install it: pip install huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}") from e


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadMIDI3DModel": DownloadAndLoadMIDI3DModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadMIDI3DModel": "(Down)Load MIDI-3D Model",
}
