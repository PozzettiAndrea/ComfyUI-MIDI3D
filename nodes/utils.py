"""Utility functions for ComfyUI-MIDI3D nodes."""

import os
import torch
import numpy as np
import trimesh
from PIL import Image, ImageOps
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import folder_paths


# Global model cache
_MODEL_CACHE: Dict[str, Any] = {}


def get_midi3d_path() -> Path:
    """Get the path to the vendored MIDI-3D source code."""
    return Path(__file__).parent.parent / "midi"


def get_midi3d_models_path() -> Path:
    """Get the path to MIDI-3D models directory."""
    models_dir = Path(folder_paths.models_dir) / "midi3d"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_device() -> torch.device:
    """Get the appropriate torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def comfy_image_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI IMAGE tensor to PIL Image.
    ComfyUI IMAGE format: [B, H, W, C], float32, range [0, 1]
    """
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np, mode='RGB')


def comfy_mask_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI MASK tensor to PIL Image (grayscale).
    ComfyUI MASK format: [N, H, W], float32, range [0, 1]
    """
    if len(tensor.shape) == 3:
        tensor = tensor[0]
    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np, mode='L')


def pil_to_comfy_image(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI IMAGE tensor."""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


def preprocess_image_for_midi(rgb_image: Image.Image, seg_image: Image.Image):
    """
    Preprocess images with padding for border objects.
    Adapted from MIDI-3D's preprocess_image function.
    """
    rgb_image = rgb_image.convert("RGB")
    seg_image = seg_image.convert("L")

    width, height = rgb_image.size

    seg_np = np.array(seg_image)
    rows, cols = np.where(seg_np > 0)
    if rows.size == 0 or cols.size == 0:
        return rgb_image, seg_image

    # Compute bounding box of combined instances
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    L = max(
        max(abs(max_row - width // 2), abs(min_row - width // 2)) * 2,
        max(abs(max_col - height // 2), abs(min_col - height // 2)) * 2,
    )

    # Pad the image
    if L > width * 0.8:
        width = int(L / 4 * 5)
    if L > height * 0.8:
        height = int(L / 4 * 5)

    rgb_new = Image.new("RGB", (width, height), (255, 255, 255))
    seg_new = Image.new("L", (width, height), 0)
    x_offset = (width - rgb_image.size[0]) // 2
    y_offset = (height - rgb_image.size[1]) // 2
    rgb_new.paste(rgb_image, (x_offset, y_offset))
    seg_new.paste(seg_image, (x_offset, y_offset))

    # Pad to square
    max_dim = max(width, height)
    rgb_new = ImageOps.expand(
        rgb_new, border=(0, 0, max_dim - width, max_dim - height), fill="white"
    )
    seg_new = ImageOps.expand(
        seg_new, border=(0, 0, max_dim - width, max_dim - height), fill=0
    )

    return rgb_new, seg_new


def split_rgb_mask(rgb_image: Image.Image, seg_image: Image.Image):
    """
    Split RGB and segmentation into per-instance images.
    Adapted from MIDI-3D's split_rgb_mask function.
    """
    rgb_image = rgb_image.convert("RGB")
    seg_image = seg_image.convert("L")

    rgb_array = np.array(rgb_image)
    seg_array = np.array(seg_image)

    label_ids = np.unique(seg_array)
    label_ids = label_ids[label_ids > 0]

    instance_rgbs, instance_masks, scene_rgbs = [], [], []

    for segment_id in sorted(label_ids):
        white_background = np.ones_like(rgb_array) * 255

        mask = np.zeros_like(seg_array, dtype=np.uint8)
        mask[seg_array == segment_id] = 255
        segment_rgb = white_background.copy()
        segment_rgb[mask == 255] = rgb_array[mask == 255]

        segment_rgb_image = Image.fromarray(segment_rgb)
        segment_mask_image = Image.fromarray(mask)
        instance_rgbs.append(segment_rgb_image)
        instance_masks.append(segment_mask_image)
        scene_rgbs.append(rgb_image)

    return instance_rgbs, instance_masks, scene_rgbs


class MIDI3DSceneToTrimesh:
    """
    Convert MIDI3D Scene to single merged Trimesh.

    Takes a Scene object containing multiple meshes and merges them into
    a single Trimesh for export or visualization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": ("MIDI3D_SCENE", {
                    "tooltip": "Scene from MIDI3DProcess"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    OUTPUT_TOOLTIPS = ("Merged trimesh from all scene objects",)
    FUNCTION = "convert"
    CATEGORY = "MIDI3D/Utils"
    DESCRIPTION = "Merge all meshes in a Scene into a single Trimesh."

    def convert(self, scene):
        """Merge scene into single trimesh."""
        if len(scene.geometry) == 1:
            # Single mesh: extract it
            mesh = list(scene.geometry.values())[0]
        else:
            # Multiple meshes: concatenate
            mesh = scene.dump(concatenate=True)

        # Preserve metadata
        if hasattr(scene, 'metadata'):
            mesh.metadata.update(scene.metadata)

        print(f"[MIDI-3D] Merged scene to single mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        return (mesh,)


NODE_CLASS_MAPPINGS = {
    "MIDI3DSceneToTrimesh": MIDI3DSceneToTrimesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MIDI3DSceneToTrimesh": "Scene to Trimesh",
}
