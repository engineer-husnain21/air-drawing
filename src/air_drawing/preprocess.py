"""Conversion of drawings into the Quick, Draw! 28x28 tensor format."""

import numpy as np
import torch

from air_drawing import config


def bitmaps_to_tensor(bitmaps: np.ndarray) -> torch.Tensor:
    """Convert uint8 bitmaps (N, 784) or (N, 28, 28) to float (N, 1, 28, 28) in [0, 1].

    This is the single scaling step shared by training and inference.
    """
    arr = np.asarray(bitmaps, dtype=np.float32).reshape(-1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    return torch.from_numpy(arr / config.PIXEL_MAX)
