"""Texture nodes for ComfyUI-MIDI3D."""

# Compatibility shim for pymeshlab versions
# pymeshlab >= 2023 renamed Percentage to PercentageValue
import pymeshlab
if not hasattr(pymeshlab, 'Percentage'):
    pymeshlab.Percentage = pymeshlab.PercentageValue

# Patch mvadapter to use torch-native backend instead of torch-cuda
# This avoids JIT compilation which requires CUDA_HOME
def _patch_mvadapter_backend():
    """Monkey-patch mvadapter to use torch-native backend for Poisson blending."""
    try:
        # Patch BEFORE pipeline_texture imports CameraProjection
        import mvadapter.utils.mesh_utils.projection as proj_module
        _orig_camera_proj_init = proj_module.CameraProjection.__init__

        def _patched_camera_proj_init(self, pb_backend, bg_remover, device, context_type="gl"):
            # Force torch-native backend, ignore whatever was passed
            _orig_camera_proj_init(self, "torch-native", bg_remover, device, context_type)

        proj_module.CameraProjection.__init__ = _patched_camera_proj_init
    except ImportError:
        # mvadapter not installed yet, will patch when it's imported
        pass

_patch_mvadapter_backend()

import os
import torch
import trimesh
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image
from tqdm import tqdm

from .utils import get_device, get_midi3d_path


# Global texture model cache
_TEXTURE_MODEL_CACHE: Dict[str, Any] = {}


class MIDI3DLoadTextureModels:
    """
    Load models required for texturing MIDI-3D meshes.

    Loads MV-Adapter (for multi-view generation) and TexturePipeline
    (for projecting textures onto meshes).

    Requires ~14GB VRAM for MV-Adapter + additional for texture pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "dtype": (["float16", "bfloat16"], {
                    "default": "float16",
                    "tooltip": "Model precision (float16 recommended for texturing)"
                }),
            }
        }

    RETURN_TYPES = ("MIDI3D_TEXTURE_MODELS",)
    RETURN_NAMES = ("texture_models",)
    OUTPUT_TOOLTIPS = ("Loaded texture models for MIDI3DTexture node",)
    FUNCTION = "load_models"
    CATEGORY = "MIDI3D"
    DESCRIPTION = "Load MV-Adapter and texture pipeline for adding textures to MIDI-3D meshes."

    def load_models(self, dtype: str = "float16"):
        """Load texture generation models."""
        device = get_device()

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        cache_key = f"texture_{dtype}"
        if cache_key in _TEXTURE_MODEL_CACHE:
            print("[MIDI-3D] Using cached texture models")
            return (_TEXTURE_MODEL_CACHE[cache_key],)

        print("[MIDI-3D] Loading texture models...")

        # Load MV-Adapter pipeline
        print("[MIDI-3D] Loading MV-Adapter pipeline...")
        ig2mv_pipe = self._load_ig2mv_pipeline(device, torch_dtype)

        # Load Texture pipeline
        print("[MIDI-3D] Loading Texture pipeline...")
        texture_pipe = self._load_texture_pipeline(device)

        model_wrapper = {
            "ig2mv_pipe": ig2mv_pipe,
            "texture_pipe": texture_pipe,
            "device": device,
            "dtype": torch_dtype,
        }

        _TEXTURE_MODEL_CACHE[cache_key] = model_wrapper
        print("[MIDI-3D] Texture models loaded successfully")

        return (model_wrapper,)

    def _load_ig2mv_pipeline(self, device, dtype):
        """Load MV-Adapter image-to-multiview pipeline."""
        from diffusers import AutoencoderKL
        from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
        from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
        from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        vae_model = "madebyollin/sdxl-vae-fp16-fix"
        adapter_path = "huanngzh/mv-adapter"
        num_views = 6

        # Load VAE (small model, safe to stage in CPU RAM)
        vae = AutoencoderKL.from_pretrained(
            vae_model,
            torch_dtype=dtype,
        ).to(device)

        # Load pipeline with device_map for memory-efficient loading
        # device_map="auto" loads weights directly to GPU, avoiding CPU RAM bottleneck
        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
            base_model,
            vae=vae,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cuda",
        )

        # Setup scheduler
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
        )

        # Init and load adapter
        pipe.init_custom_adapter(
            num_views=num_views,
            self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
        )
        pipe.load_custom_adapter(
            adapter_path,
            weight_name="mvadapter_ig2mv_partial_sdxl.safetensors"
        )

        # NOTE: Do NOT call pipe.to() - device_map already placed components
        # Only move custom components that aren't part of the device_map
        pipe.cond_encoder.to(device=device, dtype=dtype)

        # Move UNet to device to ensure adapter weights are on GPU
        # The custom adapter adds new parameters that aren't in device_map
        pipe.unet.to(device=device, dtype=dtype)
        pipe.enable_vae_slicing()

        return pipe

    def _load_texture_pipeline(self, device):
        """Load texture projection pipeline."""
        from mvadapter.pipelines.pipeline_texture import TexturePipeline

        # Download checkpoint files if needed
        checkpoints_dir = Path(get_midi3d_path()) / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        lama_path = checkpoints_dir / "big-lama.pt"
        esrgan_path = checkpoints_dir / "RealESRGAN_x2plus.pth"

        if not lama_path.exists():
            print("[MIDI-3D] Downloading LaMa inpainting model...")
            os.system(
                f"wget -q https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt -O {lama_path}"
            )

        if not esrgan_path.exists():
            print("[MIDI-3D] Downloading RealESRGAN upscaler...")
            os.system(
                f"wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O {esrgan_path}"
            )

        texture_pipe = TexturePipeline(
            upscaler_ckpt_path=str(esrgan_path),
            inpaint_ckpt_path=str(lama_path),
            device=device,
        )

        return texture_pipe


class MIDI3DTexture:
    """
    Apply textures to MIDI-3D generated meshes.

    Takes the output scene from MIDI3DProcess and the preprocessed data,
    generates multi-view images using MV-Adapter, and projects textures
    onto each mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texture_models": ("MIDI3D_TEXTURE_MODELS", {
                    "tooltip": "Loaded texture models from MIDI3DLoadTextureModels"
                }),
                "scene": ("MIDI3D_SCENE", {
                    "tooltip": "Scene from MIDI3DProcess"
                }),
                "preprocessed": ("MIDI3D_DATA", {
                    "tooltip": "Preprocessed data from MIDI3DPreprocess"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 42,
                    "min": -1,
                    "max": 2147483647,
                    "tooltip": "Random seed for texture generation (-1 for random)"
                }),
                "num_inference_steps": ("INT", {
                    "default": 35,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of diffusion steps for MV-Adapter"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Classifier-free guidance scale"
                }),
            }
        }

    RETURN_TYPES = ("MIDI3D_SCENE",)
    RETURN_NAMES = ("textured_scene",)
    OUTPUT_TOOLTIPS = ("Textured 3D scene with individual meshes",)
    FUNCTION = "apply_texture"
    CATEGORY = "MIDI3D"
    DESCRIPTION = "Apply textures to MIDI-3D meshes using MV-Adapter multi-view generation."

    def apply_texture(
        self,
        texture_models: Dict,
        scene: trimesh.Scene,
        preprocessed: Dict,
        seed: int = 42,
        num_inference_steps: int = 35,
        guidance_scale: float = 3.0,
    ):
        """Apply textures to meshes."""
        ig2mv_pipe = texture_models["ig2mv_pipe"]
        texture_pipe = texture_models["texture_pipe"]
        device = texture_models["device"]

        instance_rgbs = preprocessed["instance_rgbs"]
        instance_masks = preprocessed["instance_masks"]

        print(f"[MIDI-3D] Texturing {len(scene.geometry)} meshes...")

        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as tmp_dir:
            textured_scene = self._texture_scene(
                ig2mv_pipe=ig2mv_pipe,
                texture_pipe=texture_pipe,
                scene=scene,
                instance_rgbs=instance_rgbs,
                instance_masks=instance_masks,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_dir=tmp_dir,
                device=device,
            )

        torch.cuda.empty_cache()
        print("[MIDI-3D] Texturing complete!")

        return (textured_scene,)

    def _texture_scene(
        self,
        ig2mv_pipe,
        texture_pipe,
        scene: trimesh.Scene,
        instance_rgbs: List[Image.Image],
        instance_masks: List[Image.Image],
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        output_dir: str,
        device: str,
    ) -> trimesh.Scene:
        """Apply textures to all meshes in scene."""
        from mvadapter.pipelines.pipeline_texture import ModProcessConfig
        from mvadapter.utils import make_image_grid
        from mvadapter.utils.mesh_utils import (
            NVDiffRastContextWrapper,
            get_orthogonal_camera,
            load_mesh,
            render,
        )
        from mvadapter.utils import tensor_to_image
        from midi.utils.mesh_process import process_raw

        textured_scene = trimesh.Scene()

        for i, (mesh, rgb, mask) in tqdm(
            enumerate(zip(scene.geometry.values(), instance_rgbs, instance_masks)),
            total=len(instance_rgbs),
            desc="Texturing meshes"
        ):
            # Export mesh to temp file
            tmp_mesh_path = os.path.join(output_dir, f"mesh_{i}.glb")
            mesh.export(tmp_mesh_path)

            # Preprocess mesh
            tmp_mesh_preprocessed = os.path.join(output_dir, f"mesh_{i}_preprocessed.glb")
            process_raw(tmp_mesh_path, tmp_mesh_preprocessed, preprocess=True)

            # Create RGBA from RGB + mask
            rgba = rgb.convert("RGBA")
            rgba.putalpha(mask)

            # Run MV-Adapter
            mv_images = self._run_mvadapter(
                pipe=ig2mv_pipe,
                mesh_path=tmp_mesh_preprocessed,
                image=rgba,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                device=device,
            )

            # Save multi-view images
            mv_path = os.path.join(output_dir, f"mv_{i}.png")
            make_image_grid(mv_images, rows=1).save(mv_path)

            # Run texture projection
            texture_out = texture_pipe(
                mesh_path=tmp_mesh_preprocessed,
                save_dir=output_dir,
                save_name=f"mesh_{i}",
                move_to_center=True,
                uv_unwarp=False,
                preprocess_mesh=False,
                uv_size=4096,
                rgb_path=mv_path,
                rgb_process_config=ModProcessConfig(view_upscale=True, inpaint_mode="view"),
                camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            )

            # Load textured mesh
            textured_mesh = trimesh.load(texture_out.shaded_model_save_path, process=False)
            textured_scene.add_geometry(textured_mesh)

            torch.cuda.empty_cache()

        return textured_scene

    def _run_mvadapter(
        self,
        pipe,
        mesh_path: str,
        image: Image.Image,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        device: str,
    ) -> List[Image.Image]:
        """Run MV-Adapter to generate multi-view images."""
        from mvadapter.utils.mesh_utils import (
            NVDiffRastContextWrapper,
            get_orthogonal_camera,
            load_mesh,
            render,
        )
        from mvadapter.utils import tensor_to_image
        import numpy as np

        num_views = 6
        height = 768
        width = 768

        # Prepare cameras
        cameras = get_orthogonal_camera(
            elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
            distance=[1.8] * num_views,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            device=device,
        )

        ctx = NVDiffRastContextWrapper(device=device)

        mesh, offset, scale = load_mesh(
            mesh_path,
            rescale=True,
            move_to_center=True,
            device=device,
            return_transform=True,
        )

        render_out = render(
            ctx,
            mesh,
            cameras,
            height=height,
            width=width,
            render_attr=False,
            normal_background=0.0,
        )

        control_images = (
            torch.cat(
                [
                    (render_out.pos + 0.5).clamp(0, 1),
                    (render_out.normal / 2 + 0.5).clamp(0, 1),
                ],
                dim=-1,
            )
            .permute(0, 3, 1, 2)
            .to(device)
        )

        # Preprocess reference image
        reference_image = self._preprocess_image(image, height, width)

        # Run pipeline
        pipe_kwargs = {}
        if seed != -1:
            pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

        images = pipe(
            "high quality",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            reference_image=reference_image,
            reference_conditioning_scale=0.7,
            negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
            **pipe_kwargs,
        ).images

        return images

    def _preprocess_image(self, image: Image.Image, height: int, width: int) -> Image.Image:
        """Preprocess RGBA image for MV-Adapter."""
        import numpy as np

        image = np.array(image)
        alpha = image[..., 3] > 0
        H, W = alpha.shape

        # Get bounding box of alpha
        y, x = np.where(alpha)
        y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
        x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
        image_center = image[y0:y1, x0:x1]

        # Resize longer side to 90% of target
        H, W, _ = image_center.shape
        if H > W:
            W = int(W * (height * 0.9) / H)
            H = int(height * 0.9)
        else:
            H = int(H * (width * 0.9) / W)
            W = int(width * 0.9)

        image_center = np.array(Image.fromarray(image_center).resize((W, H)))

        # Pad to target size
        start_h = (height - H) // 2
        start_w = (width - W) // 2
        result = np.zeros((height, width, 4), dtype=np.uint8)
        result[start_h:start_h + H, start_w:start_w + W] = image_center

        # Composite on gray background
        result = result.astype(np.float32) / 255.0
        result = result[:, :, :3] * result[:, :, 3:4] + (1 - result[:, :, 3:4]) * 0.5
        result = (result * 255).clip(0, 255).astype(np.uint8)

        return Image.fromarray(result)


NODE_CLASS_MAPPINGS = {
    "MIDI3DLoadTextureModels": MIDI3DLoadTextureModels,
    "MIDI3DTexture": MIDI3DTexture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MIDI3DLoadTextureModels": "(Down)Load MIDI-3D Texture Models",
    "MIDI3DTexture": "MIDI-3D Texture",
}
