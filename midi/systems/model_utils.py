import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..utils.typing import *

CLIP_SIZE = (224, 224)  # height, width
CLIP_INPUT_MEAN = torch.as_tensor(
    [0.48145466, 0.4578275, 0.40821073], dtype=torch.float32
)[None, :, None, None]
CLIP_INPUT_STD = torch.as_tensor(
    [0.26862954, 0.26130258, 0.27577711], dtype=torch.float32
)[None, :, None, None]

DINOv2_SIZE = (224, 224)
DINOv2_INPUT_MEAN = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32)[
    None, :, None, None
]
DINOv2_INPUT_STD = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32)[
    None, :, None, None
]


def preprocess_image_for_clip(
    image: Float[Tensor, "B C H W"], do_resize=True, size: Optional[int] = None
):
    if do_resize:
        size = size if size is not None else DINOv2_SIZE
        image = F.interpolate(image, size=size, mode="bilinear")
    image = (image - CLIP_INPUT_MEAN.to(image)) / CLIP_INPUT_STD.to(image)
    return image


def preprocess_image_for_dinov2(
    image: Float[Tensor, "B C H W"], do_resize=True, size: Optional[int] = None
):
    if do_resize:
        size = size if size is not None else DINOv2_SIZE
        image = F.interpolate(image, size=size, mode="bilinear")
    image = (image - DINOv2_INPUT_MEAN.to(image)) / DINOv2_INPUT_STD.to(image)
    return image


def to_pil_image(image: Float[Tensor, "B C H W"]) -> List[Image.Image]:
    batch_size = image.shape[0]
    to_pil = lambda x: Image.fromarray((x * 255).astype(np.uint8))
    if image.shape[1] == 3:  # rgb
        pil_list = [
            to_pil(image[i].permute(1, 2, 0).cpu().numpy()) for i in range(batch_size)
        ]
    elif image.shape[1] == 1:  # grayscale
        pil_list = [to_pil(image[i, 0].cpu().numpy()) for i in range(batch_size)]
    else:
        raise ValueError(f"Invalid image shape: {image.shape}")

    return pil_list
