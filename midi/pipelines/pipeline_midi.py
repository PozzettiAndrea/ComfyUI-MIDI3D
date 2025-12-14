import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # not sure
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from peft import LoraConfig, get_peft_model_state_dict
from transformers import (
    BitImageProcessor,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    Dinov2Model,
)

from ..inference_utils import generate_dense_grid_points
from ..loaders import CustomAdapterMixin
from ..models.attention_processor import MIAttnProcessor2_0
from ..models.autoencoders import TripoSGVAEModel
from ..models.transformers import TripoSGDiTModel, set_transformer_attn_processor
from .pipeline_triposg_output import TripoSGPipelineOutput
from .pipeline_utils import TransformerDiffusionMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class MIDIPipeline(DiffusionPipeline, TransformerDiffusionMixin, CustomAdapterMixin):
    """
    Pipeline for image-to-scene generation based on pre-trained shape diffusion.
    """

    def __init__(
        self,
        vae: TripoSGVAEModel,
        transformer: TripoSGDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_1: CLIPVisionModelWithProjection,
        image_encoder_2: Dinov2Model,
        feature_extractor_1: CLIPImageProcessor,
        feature_extractor_2: BitImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_1=image_encoder_1,
            image_encoder_2=image_encoder_2,
            feature_extractor_1=feature_extractor_1,
            feature_extractor_2=feature_extractor_2,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image_1(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder_1.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_1(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder_1(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(
            num_images_per_prompt, dim=0
        ).unsqueeze(1)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def encode_image_2(
        self,
        image_one,
        image_two,
        mask,
        device,
        num_images_per_prompt,
    ):
        dtype = next(self.image_encoder_2.parameters()).dtype

        images = [image_one, image_two, mask]
        images_new = []
        for i, image in enumerate(images):
            if not isinstance(image, torch.Tensor):
                if i <= 1:
                    images_new.append(
                        self.feature_extractor_2(
                            image, return_tensors="pt"
                        ).pixel_values
                    )
                else:
                    image = [
                        torch.from_numpy(
                            (np.array(im) / 255.0).astype(np.float32)
                        ).unsqueeze(0)
                        for im in image
                    ]
                    image = torch.stack(image, dim=0)
                    images_new.append(
                        F.interpolate(
                            image, size=images_new[0].shape[-2:], mode="nearest"
                        )
                    )

        image = torch.cat(images_new, dim=1).to(device=device, dtype=dtype)
        image_embeds = self.image_encoder_2(image).last_hidden_state
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
        sampled_points: torch.Tensor,
        decode_progressive: bool = False,
        decode_to_cpu: bool = False,
        # Params for sampling points
        bbox_min: np.ndarray = np.array([-1.005, -1.005, -1.005]),
        bbox_max: np.ndarray = np.array([1.005, 1.005, 1.005]),
        octree_depth: int = 8,
        indexing: str = "ij",
        padding: float = 0.05,
    ):
        device, dtype = latents.device, latents.dtype
        batch_size = latents.shape[0]

        grid_sizes, bbox_sizes, bbox_mins, bbox_maxs = [], [], [], []

        if sampled_points is None:
            sampled_points, grid_size, bbox_size = generate_dense_grid_points(
                bbox_min, bbox_max, octree_depth, indexing
            )
            sampled_points = torch.FloatTensor(sampled_points).to(
                device=device, dtype=dtype
            )
            sampled_points = sampled_points.unsqueeze(0).expand(batch_size, -1, -1)

            grid_sizes.append(grid_size)
            bbox_sizes.append(bbox_size)
            bbox_mins.append(bbox_min)
            bbox_maxs.append(bbox_max)

        self.vae: TripoSGVAEModel
        output = self.vae.decode(
            latents, sampled_points=sampled_points, to_cpu=decode_to_cpu
        ).sample

        if not decode_progressive:
            return (output, grid_sizes, bbox_sizes, bbox_mins, bbox_maxs)

        grid_sizes, bbox_sizes, bbox_mins, bbox_maxs = [], [], [], []
        sampled_points_list = []

        for i in range(batch_size):
            sdf_ = output[i].squeeze(-1)  # [num_points]
            sampled_points_ = sampled_points[i]
            occupied_points = sampled_points_[sdf_ <= 0]  # [num_occupied_points, 3]

            if occupied_points.shape[0] == 0:
                logger.warning(
                    f"No occupied points found in batch {i}. Using original bounding box."
                )
            else:
                bbox_min = occupied_points.min(dim=0).values
                bbox_max = occupied_points.max(dim=0).values
                bbox_min = (bbox_min - padding).float().cpu().numpy()
                bbox_max = (bbox_max + padding).float().cpu().numpy()

            sampled_points_, grid_size, bbox_size = generate_dense_grid_points(
                bbox_min, bbox_max, octree_depth, indexing
            )
            sampled_points_ = torch.FloatTensor(sampled_points_).to(
                device=device, dtype=dtype
            )
            sampled_points_list.append(sampled_points_)

            grid_sizes.append(grid_size)
            bbox_sizes.append(bbox_size)
            bbox_mins.append(bbox_min)
            bbox_maxs.append(bbox_max)

        sampled_points = torch.stack(sampled_points_list, dim=0)

        # Re-decode the new sampled points
        output = self.vae.decode(
            latents, sampled_points=sampled_points, to_cpu=decode_to_cpu
        ).sample

        return (output, grid_sizes, bbox_sizes, bbox_mins, bbox_maxs)

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        mask: PipelineImageInput,
        image_scene: PipelineImageInput,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        sampled_points: Optional[torch.Tensor] = None,
        decode_progressive: bool = False,
        decode_to_cpu: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        output_type: Optional[str] = "mesh_vf",
        return_dict: bool = True,
    ):
        # 1. Check inputs. Raise error if not correct
        # TODO

        self._decode_progressive = decode_progressive
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device

        # 3. Encode condition
        image_embeds_1, negative_image_embeds_1 = self.encode_image_1(
            image, device, num_images_per_prompt
        )
        image_embeds_2, negative_image_embeds_2 = self.encode_image_2(
            image, image_scene, mask, device, num_images_per_prompt
        )

        if self.do_classifier_free_guidance:
            image_embeds_1 = torch.cat([negative_image_embeds_1, image_embeds_1], dim=0)
            image_embeds_2 = torch.cat([negative_image_embeds_2, image_embeds_2], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_tokens = self.transformer.config.width
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds_1.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds_1,
                    encoder_hidden_states_2=image_embeds_2,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    image_embeds_1 = callback_outputs.pop(
                        "image_embeds_1", image_embeds_1
                    )
                    negative_image_embeds_1 = callback_outputs.pop(
                        "negative_image_embeds_1", negative_image_embeds_1
                    )
                    image_embeds_2 = callback_outputs.pop(
                        "image_embeds_2", image_embeds_2
                    )
                    negative_image_embeds_2 = callback_outputs.pop(
                        "negative_image_embeds_2", negative_image_embeds_2
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        grid_sizes, bbox_sizes, bbox_mins, bbox_maxs = None, None, None, None

        if output_type == "latent":
            output = latents
        else:
            output, grid_sizes, bbox_sizes, bbox_mins, bbox_maxs = self.decode_latents(
                latents,
                sampled_points=sampled_points,
                decode_progressive=decode_progressive,
                decode_to_cpu=decode_to_cpu,
            )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, grid_sizes, bbox_sizes, bbox_mins, bbox_maxs)

        return TripoSGPipelineOutput(
            samples=output,
            grid_sizes=grid_sizes,
            bbox_sizes=bbox_sizes,
            bbox_mins=bbox_mins,
            bbox_maxs=bbox_maxs,
        )

    def _init_custom_adapter(
        self,
        # Attention processor
        set_self_attn_module_names: Optional[List[str]] = None,
        # Image encoder 2
        pretrained_image_encoder_2_processor_config: Optional[Dict[str, Any]] = None,
        image_encoder_2_input_channels: int = 7,
        image_encoder_2_init_projection_method: str = "clone",
        # LoRA
        transformer_lora_config: Optional[Dict[str, Any]] = None,
        image_encoder_1_lora_config: Optional[Dict[str, Any]] = None,
        image_encoder_2_lora_config: Optional[Dict[str, Any]] = None,
    ):
        # Modify feature extractor 2 if needed
        if pretrained_image_encoder_2_processor_config is not None:
            self.feature_extractor_2 = BitImageProcessor.from_dict(
                self.feature_extractor_2.to_dict(),
                **(pretrained_image_encoder_2_processor_config or {}),
            )

        # Expand the input channels of the image encoder if needed
        original_num_channels = self.image_encoder_2.config.num_channels
        if original_num_channels < image_encoder_2_input_channels:
            image_encoder_2: Dinov2Model = self.image_encoder_2
            image_encoder_2_config = image_encoder_2.config
            image_encoder_2_config.num_channels = image_encoder_2_input_channels
            image_encoder_2_state_dict = image_encoder_2.state_dict()
            projection_key = "embeddings.patch_embeddings.projection.weight"
            projection_weight = image_encoder_2_state_dict[projection_key]
            new_projection_weight = [projection_weight[:, :3]]
            num_channels_left = image_encoder_2_config.num_channels - 3
            while num_channels_left > 0:
                if image_encoder_2_init_projection_method == "clone":
                    new_projection_weight.append(
                        projection_weight[
                            :, : min(original_num_channels, num_channels_left)
                        ].clone()
                    )
                elif image_encoder_2_init_projection_method == "zero":
                    new_projection_weight.append(
                        torch.zeros_like(
                            projection_weight[
                                :, : min(original_num_channels, num_channels_left)
                            ]
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported image_encoder_2_init_projection_method: {image_encoder_2_init_projection_method}"
                    )
                num_channels_left -= original_num_channels
            new_projection_weight = torch.cat(new_projection_weight, dim=1)
            image_encoder_2_state_dict[projection_key] = new_projection_weight

            self.image_encoder_2 = Dinov2Model(image_encoder_2_config)
            self.image_encoder_2.load_state_dict(image_encoder_2_state_dict)

        self.image_encoder_2 = self.image_encoder_2.to(self.device, self.dtype)

        # Set attention processor
        func_default = lambda name, hs, cad, ap: MIAttnProcessor2_0(use_mi=False)
        set_transformer_attn_processor(  # avoid warning
            self.transformer,
            set_self_attn_proc_func=func_default,
            set_cross_attn_1_proc_func=func_default,
            set_cross_attn_2_proc_func=func_default,
        )
        set_transformer_attn_processor(
            self.transformer,
            set_self_attn_proc_func=lambda name, hs, cad, ap: MIAttnProcessor2_0(),
            set_self_attn_module_names=set_self_attn_module_names,
        )

        # LoRA
        if transformer_lora_config is not None:
            self.transformer.add_adapter(LoraConfig(**transformer_lora_config))
        else:
            self.transformer.requires_grad_(False)
        if image_encoder_1_lora_config is not None:
            self.image_encoder_1.add_adapter(LoraConfig(**image_encoder_1_lora_config))
        else:
            self.image_encoder_1.requires_grad_(False)
        if image_encoder_2_lora_config is not None:
            self.image_encoder_2.add_adapter(LoraConfig(**image_encoder_2_lora_config))
        else:
            self.image_encoder_2.requires_grad_(False)

    def _load_custom_adapter(self, state_dict):
        parse_state_dict = lambda state_dict, prefix: {
            k.replace(prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        transformer_state_dict = parse_state_dict(state_dict, "transformer.")
        image_encoder_1_state_dict = parse_state_dict(state_dict, "image_encoder_1.")
        image_encoder_2_state_dict = parse_state_dict(state_dict, "image_encoder_2.")

        if len(transformer_state_dict) > 0:
            self.transformer.load_state_dict(transformer_state_dict, strict=False)
        if len(image_encoder_1_state_dict) > 0:
            self.image_encoder_1.load_state_dict(
                image_encoder_1_state_dict, strict=False
            )
        if len(image_encoder_2_state_dict) > 0:
            self.image_encoder_2.load_state_dict(
                image_encoder_2_state_dict, strict=False
            )

    def _save_custom_adapter(
        self,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        def include_fn(k):
            is_included = False

            if include_keys is not None:
                is_included = is_included or any([key in k for key in include_keys])
            if exclude_keys is not None:
                is_included = is_included and not any(
                    [key in k for key in exclude_keys]
                )

            return is_included

        parse_state_dict = lambda state_dict, prefix: {
            prefix + k: v for k, v in state_dict.items() if include_fn(k)
        }
        transformer_state_dict = parse_state_dict(
            self.transformer.state_dict(), "transformer."
        )
        image_encoder_1_state_dict = parse_state_dict(
            self.image_encoder_1.state_dict(), "image_encoder_1."
        )
        image_encoder_2_state_dict = parse_state_dict(
            self.image_encoder_2.state_dict(), "image_encoder_2."
        )
        state_dict = {
            **transformer_state_dict,
            **image_encoder_1_state_dict,
            **image_encoder_2_state_dict,
        }

        return state_dict
