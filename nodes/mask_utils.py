"""Mask utility nodes for ComfyUI-MIDI3D."""

import torch
import numpy as np
from PIL import Image


class RGBToInstanceMask:
    """
    Convert an RGB colored mask image to an instance segmentation mask.

    Each unique color in the input becomes a unique instance label (1, 2, 3...).
    Black (0,0,0) is treated as background (label 0).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "RGB image where each object is a different color"
                }),
            },
            "optional": {
                "ignore_black": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Treat black pixels as background (label 0)"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("instance_mask",)
    OUTPUT_TOOLTIPS = ("Grayscale instance mask for MIDI-3D",)
    FUNCTION = "convert"
    CATEGORY = "MIDI3D"
    DESCRIPTION = "Convert RGB colored mask to instance segmentation mask for MIDI-3D."

    def convert(self, image: torch.Tensor, ignore_black: bool = True):
        """Convert RGB mask to instance mask."""
        # Handle batch dimension
        if len(image.shape) == 4:
            image = image[0]

        # Convert to numpy RGB (H, W, 3), values 0-255
        rgb_np = (image.cpu().numpy() * 255).astype(np.uint8)
        h, w = rgb_np.shape[:2]

        # Flatten to find unique colors
        pixels = rgb_np.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)

        # Create instance mask
        instance_mask = np.zeros((h, w), dtype=np.uint8)

        label = 1
        for color in unique_colors:
            # Skip black if ignore_black is True
            if ignore_black and np.all(color == 0):
                continue

            # Find all pixels matching this color
            mask = np.all(rgb_np == color, axis=2)
            instance_mask[mask] = label

            print(f"[MIDI-3D] Instance {label}: color RGB({color[0]}, {color[1]}, {color[2]})")
            label += 1

        num_instances = label - 1
        print(f"[MIDI-3D] Created instance mask with {num_instances} instances")

        # Convert to ComfyUI MASK format: [1, H, W], float32, range [0, 1]
        mask_tensor = torch.from_numpy(instance_mask.astype(np.float32) / 255.0)
        mask_tensor = mask_tensor.unsqueeze(0)

        return (mask_tensor,)


class CombineMasksToInstance:
    """
    Combine multiple binary masks into a single instance segmentation mask.

    Each input mask becomes a unique instance (1, 2, 3...).
    Useful for combining SAM output masks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK", {"tooltip": "First object mask (becomes instance 1)"}),
            },
            "optional": {
                "mask2": ("MASK", {"tooltip": "Second object mask (becomes instance 2)"}),
                "mask3": ("MASK", {"tooltip": "Third object mask (becomes instance 3)"}),
                "mask4": ("MASK", {"tooltip": "Fourth object mask (becomes instance 4)"}),
                "mask5": ("MASK", {"tooltip": "Fifth object mask (becomes instance 5)"}),
                "mask6": ("MASK", {"tooltip": "Sixth object mask (becomes instance 6)"}),
                "mask7": ("MASK", {"tooltip": "Seventh object mask (becomes instance 7)"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("instance_mask",)
    OUTPUT_TOOLTIPS = ("Combined instance segmentation mask",)
    FUNCTION = "combine"
    CATEGORY = "MIDI3D"
    DESCRIPTION = "Combine multiple binary masks into one instance mask for MIDI-3D."

    def combine(self, mask1, mask2=None, mask3=None, mask4=None,
                mask5=None, mask6=None, mask7=None):
        """Combine binary masks into instance mask."""
        masks = [mask1, mask2, mask3, mask4, mask5, mask6, mask7]
        masks = [m for m in masks if m is not None]

        # Get shape from first mask
        if len(mask1.shape) == 3:
            h, w = mask1.shape[1], mask1.shape[2]
        else:
            h, w = mask1.shape[0], mask1.shape[1]

        # Create instance mask
        instance_mask = np.zeros((h, w), dtype=np.uint8)

        for i, mask in enumerate(masks):
            label = i + 1

            # Handle batch dimension
            if len(mask.shape) == 3:
                mask = mask[0]

            # Convert to numpy and threshold
            mask_np = mask.cpu().numpy()
            binary = (mask_np > 0.5).astype(np.uint8)

            # Assign label (later masks overwrite earlier ones in overlapping regions)
            instance_mask[binary == 1] = label

            print(f"[MIDI-3D] Added mask as instance {label}")

        print(f"[MIDI-3D] Combined {len(masks)} masks into instance mask")

        # Convert to ComfyUI MASK format
        mask_tensor = torch.from_numpy(instance_mask.astype(np.float32) / 255.0)
        mask_tensor = mask_tensor.unsqueeze(0)

        return (mask_tensor,)


NODE_CLASS_MAPPINGS = {
    "RGBToInstanceMask": RGBToInstanceMask,
    "CombineMasksToInstance": CombineMasksToInstance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBToInstanceMask": "RGB to Instance Mask (MIDI-3D)",
    "CombineMasksToInstance": "Combine Masks to Instance (MIDI-3D)",
}
