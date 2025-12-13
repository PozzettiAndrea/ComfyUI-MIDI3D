"""MIDI3DPreprocess node for preparing images."""

import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List

from .utils import (
    comfy_image_to_pil,
    comfy_mask_to_pil,
    preprocess_image_for_midi,
    split_rgb_mask,
)


class MIDI3DPreprocess:
    """
    Prepare RGB image and segmentation mask for MIDI-3D processing.

    This node splits the input into per-instance images based on the
    segmentation mask. Each unique non-zero value in the mask represents
    a different object instance.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "RGB image of the scene"
                }),
                "segmentation": ("MASK", {
                    "tooltip": "Segmentation mask (different values = different instances)"
                }),
            },
            "optional": {
                "do_padding": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Pad image for border objects (use if objects touch image edges)"
                }),
            }
        }

    RETURN_TYPES = ("MIDI3D_DATA",)
    RETURN_NAMES = ("preprocessed",)
    OUTPUT_TOOLTIPS = ("Preprocessed instance data for MIDI-3D",)
    FUNCTION = "preprocess"
    CATEGORY = "MIDI3D"
    DESCRIPTION = "Prepare RGB image and segmentation mask for MIDI-3D 3D scene generation."

    def preprocess(
        self,
        image: torch.Tensor,
        segmentation: torch.Tensor,
        do_padding: bool = False,
    ):
        """Preprocess image and segmentation for MIDI-3D."""
        print(f"[MIDI-3D] Preprocessing image...")

        # Convert to PIL
        rgb_image = comfy_image_to_pil(image)
        seg_image = comfy_mask_to_pil(segmentation)

        # Optional padding
        if do_padding:
            print("[MIDI-3D] Applying padding for border objects...")
            rgb_image, seg_image = preprocess_image_for_midi(rgb_image, seg_image)

        # Split into instances
        instance_rgbs, instance_masks, scene_rgbs = split_rgb_mask(rgb_image, seg_image)

        num_instances = len(instance_rgbs)
        print(f"[MIDI-3D] Found {num_instances} instances in segmentation")

        if num_instances == 0:
            raise ValueError("No instances found in segmentation mask. Mask should have non-zero values for objects.")

        preprocessed_data = {
            "instance_rgbs": instance_rgbs,
            "instance_masks": instance_masks,
            "scene_rgbs": scene_rgbs,
            "num_instances": num_instances,
            "original_size": rgb_image.size,
            "do_padding": do_padding,
        }

        return (preprocessed_data,)


class MIDI3DPreprocessFromFiles:
    """
    Load and preprocess RGB image and segmentation mask from file paths.

    Alternative to MIDI3DPreprocess that takes file paths directly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to RGB image file"
                }),
                "seg_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to segmentation mask file"
                }),
            },
            "optional": {
                "do_padding": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Pad image for border objects"
                }),
            }
        }

    RETURN_TYPES = ("MIDI3D_DATA",)
    RETURN_NAMES = ("preprocessed",)
    OUTPUT_TOOLTIPS = ("Preprocessed instance data for MIDI-3D",)
    FUNCTION = "preprocess"
    CATEGORY = "MIDI3D"
    DESCRIPTION = "Load and prepare RGB/segmentation images from file paths for MIDI-3D."

    def preprocess(
        self,
        rgb_path: str,
        seg_path: str,
        do_padding: bool = False,
    ):
        """Load and preprocess images from file paths."""
        print(f"[MIDI-3D] Loading images from files...")

        # Load images
        rgb_image = Image.open(rgb_path).convert("RGB")
        seg_image = Image.open(seg_path).convert("L")

        # Optional padding
        if do_padding:
            print("[MIDI-3D] Applying padding for border objects...")
            rgb_image, seg_image = preprocess_image_for_midi(rgb_image, seg_image)

        # Split into instances
        instance_rgbs, instance_masks, scene_rgbs = split_rgb_mask(rgb_image, seg_image)

        num_instances = len(instance_rgbs)
        print(f"[MIDI-3D] Found {num_instances} instances in segmentation")

        if num_instances == 0:
            raise ValueError("No instances found in segmentation mask.")

        preprocessed_data = {
            "instance_rgbs": instance_rgbs,
            "instance_masks": instance_masks,
            "scene_rgbs": scene_rgbs,
            "num_instances": num_instances,
            "original_size": rgb_image.size,
            "do_padding": do_padding,
        }

        return (preprocessed_data,)


NODE_CLASS_MAPPINGS = {
    "MIDI3DPreprocess": MIDI3DPreprocess,
    "MIDI3DPreprocessFromFiles": MIDI3DPreprocessFromFiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MIDI3DPreprocess": "MIDI-3D Preprocess",
    "MIDI3DPreprocessFromFiles": "MIDI-3D Preprocess (Files)",
}
