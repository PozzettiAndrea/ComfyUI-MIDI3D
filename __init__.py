"""
ComfyUI-MIDI3D: ComfyUI nodes for MIDI-3D multi-instance 3D scene generation.

MIDI-3D generates compositional 3D scenes from a single image with
instance segmentation, handling multiple objects simultaneously using
multi-instance diffusion.

Nodes:
- (Down)Load MIDI-3D Model: Download and load the MIDI-3D pipeline
- MIDI-3D Preprocess: Prepare RGB image and segmentation mask
- MIDI-3D Preprocess (Files): Load and prepare from file paths
- MIDI-3D Process: Run inference to generate 3D scene

Integrates with GeometryPack's TRIMESH type for seamless mesh pipeline.
"""

# Add vendored midi module to sys.path so diffusers can import it
import sys
from pathlib import Path
_midi_path = str(Path(__file__).parent)
if _midi_path not in sys.path:
    sys.path.insert(0, _midi_path)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
