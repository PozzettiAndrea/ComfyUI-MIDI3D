from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.utils import BaseOutput

PipelineBoxOutput = Union[
    List[List[int]],  # [[257, 257, 257], ...]
    List[List[float]],  # [[-1.05, -1.05, -1.05], ...]
    List[np.ndarray],
]


@dataclass
class TripoSGPipelineOutput(BaseOutput):
    r"""
    Output class for TripoSG pipelines.
    """

    samples: torch.Tensor
    grid_sizes: Optional[PipelineBoxOutput] = None
    bbox_sizes: Optional[PipelineBoxOutput] = None
    bbox_mins: Optional[PipelineBoxOutput] = None
    bbox_maxs: Optional[PipelineBoxOutput] = None
