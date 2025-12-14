"""MIDI3DProcess node for running inference."""

import time
import torch
import trimesh
import numpy as np
from skimage import measure
from typing import Any, Dict



class MIDI3DProcess:
    """
    Run MIDI-3D inference to generate a 3D scene from preprocessed data.

    This node takes the model and preprocessed instance data and generates
    a compositional 3D scene with multiple objects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MIDI3D_MODEL", {
                    "tooltip": "MIDI-3D model from loader node"
                }),
                "preprocessed": ("MIDI3D_DATA", {
                    "tooltip": "Preprocessed data from MIDI-3D Preprocess node"
                }),
            },
            "optional": {
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Number of diffusion steps (more = better quality, slower)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.0,
                    "max": 15.0,
                    "step": 0.5,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": -1,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed (-1 for random)"
                }),
            }
        }

    RETURN_TYPES = ("MIDI3D_SCENE",)
    RETURN_NAMES = ("scene",)
    OUTPUT_TOOLTIPS = ("Generated 3D scene with individual meshes",)
    FUNCTION = "process"
    CATEGORY = "MIDI3D"
    DESCRIPTION = "Generate 3D scene from image using MIDI-3D multi-instance diffusion."

    def process(
        self,
        model: Dict[str, Any],
        preprocessed: Dict[str, Any],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        seed: int = 42,
    ):
        """Run MIDI-3D inference."""
        print(f"[MIDI-3D] Running inference (steps={num_inference_steps}, guidance={guidance_scale})")

        from midi.utils.smoothing import smooth_gpu

        pipe = model["pipe"]
        device = model["device"]

        instance_rgbs = preprocessed["instance_rgbs"]
        instance_masks = preprocessed["instance_masks"]
        scene_rgbs = preprocessed["scene_rgbs"]
        num_instances = preprocessed["num_instances"]

        # Set up generator
        pipe_kwargs = {}
        if seed != -1:
            pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

        # Run inference
        print(f"[MIDI-3D] Generating {num_instances} instances...")
        start_time = time.time()

        with torch.no_grad():
            outputs = pipe(
                image=instance_rgbs,
                mask=instance_masks,
                image_scene=scene_rgbs,
                attention_kwargs={"num_instances": num_instances},
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                decode_progressive=True,
                return_dict=False,
                **pipe_kwargs,
            )

        inference_time = time.time() - start_time
        print(f"[MIDI-3D] Diffusion completed in {inference_time:.2f}s")

        # Marching cubes to extract meshes
        print("[MIDI-3D] Extracting meshes via marching cubes...")
        trimeshes = []

        for idx, (logits_, grid_size, bbox_size, bbox_min, bbox_max) in enumerate(
            zip(*outputs)
        ):
            grid_logits = logits_.view(grid_size)

            # Smooth the SDF
            grid_logits = smooth_gpu(grid_logits, method="gaussian", sigma=1)
            torch.cuda.empty_cache()

            # Marching cubes
            vertices, faces, normals, _ = measure.marching_cubes(
                grid_logits.float().cpu().numpy(), 0, method="lewiner"
            )

            # Scale vertices to world coordinates
            vertices = vertices / grid_size * bbox_size + bbox_min

            # Create trimesh
            mesh = trimesh.Trimesh(
                vertices.astype(np.float32),
                np.ascontiguousarray(faces)
            )
            trimeshes.append(mesh)
            print(f"[MIDI-3D] Instance {idx + 1}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Compose scene - keep individual meshes
        if len(trimeshes) == 1:
            # Single mesh: wrap in Scene for consistency
            output_scene = trimesh.Scene([trimeshes[0]])
        else:
            output_scene = trimesh.Scene(trimeshes)

        # Add metadata to scene
        output_scene.metadata.update({
            'source': 'midi3d',
            'num_instances': num_instances,
            'inference_time': inference_time,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
        })

        # Count total vertices/faces across all meshes
        total_vertices = sum(len(mesh.vertices) for mesh in trimeshes)
        total_faces = sum(len(mesh.faces) for mesh in trimeshes)
        print(f"[MIDI-3D] Output scene: {total_vertices} vertices, {total_faces} faces")

        return (output_scene,)


NODE_CLASS_MAPPINGS = {
    "MIDI3DProcess": MIDI3DProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MIDI3DProcess": "MIDI-3D Process",
}
