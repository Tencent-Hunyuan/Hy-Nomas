"""Encode a still image as a fixed segment in DCAE-V latent space."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from hymm.utils.image_resize import fit_image_to_frame


@torch.no_grad()
def encode_reference_image(vae, image_path, *, height, width, num_frames, device, dtype):
    """Repeat the image in pixel space and encode the complete video segment.

    At 24 FPS the five-second reference is 120 identical input frames. Pass
    that full video through DCAE-V so its temporal convolutions and internal
    feature cache determine all 30 latent frames. No extra latent scale is used.
    """
    spatial = int(vae.spatial_compression_ratio)
    temporal = int(vae.time_compression_ratio)
    height, width = height // spatial * spatial, width // spatial * spatial
    if min(height, width) < spatial or num_frames <= 0 or num_frames % temporal:
        raise ValueError("Reference segment dimensions must match DCAE-V compression")
    with Image.open(Path(image_path)) as source:
        image = fit_image_to_frame(source, (width, height))
        pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
    pixels = pixels.to(device=device, dtype=dtype).div_(127.5).sub_(1)
    clip = pixels[None, :, None].repeat(1, 1, num_frames, 1, 1)
    logger.info(f"Encoding image reference with DCAE-V: pixel_frames={num_frames}, size={height}x{width}")
    latent = vae.encode(clip).latent_dist.mode()
    expected = (1, 32, num_frames // temporal, height // spatial, width // spatial)
    if tuple(latent.shape) != expected:
        raise ValueError(f"Unexpected reference latent shape: {tuple(latent.shape)} != {expected}")
    return latent.contiguous()
