"""Incremental noise warping adapter for streaming video generation."""

import cv2
import torch
import numpy as np
from simulation.image23D.noise_warp.noise_warp import NoiseWarper as _NW, mix_new_noise
from config import (
    DEFAULT_HEIGHT, DEFAULT_WIDTH, NOISE_CHANNELS,
    LATENT_H, LATENT_W, LATENT_C, FRAMES_PER_BLOCK, TEMPORAL_FACTOR,
    EVAL_DEGRADATION,
)


def _resize_flow(flow, new_size=(832, 832)):
    """Resize flow field of shape (2, H, W) to (2, new_H, new_W)."""
    resized = np.zeros((2, new_size[0], new_size[1]), dtype=flow.dtype)
    for i in range(2):
        resized[i] = cv2.resize(flow[i], (new_size[1], new_size[0]),
                                interpolation=cv2.INTER_LINEAR)
    return resized


class StreamingNoiseWarper:
    """Incremental noise warping that works frame-by-frame.

    Usage:
        warper = StreamingNoiseWarper(crop_start=200)
        for each pixel frame:
            warper.warp_step(optical_flow_2hw)
        structured, sde = warper.get_block_noise(block_idx)
    """

    def __init__(self, crop_start=176, eval_degradation=EVAL_DEGRADATION,
                 device="cuda"):
        self.height = DEFAULT_HEIGHT  # 480
        self.width = DEFAULT_WIDTH    # 832
        self.noise_channels = NOISE_CHANNELS  # 32
        self.crop_start = crop_start
        self.eval_degradation = eval_degradation
        self.device = device

        self.noise_buffer = []
        self.frame_count = 0
        self._init_warper()

    def _init_warper(self):
        self.nw = _NW(
            c=self.noise_channels,
            h=self.height,
            w=self.width,
            device=self.device,
            scale_factor=1,
        )
        self.noise_buffer = [self.nw.noise.clone()]

    def warp_step(self, flow_2hw):
        """Warp noise by one frame using the given optical flow.

        Args:
            flow_2hw: numpy array of shape (2, 512, 512). Channel 0=dx, 1=dy.
        """
        if isinstance(flow_2hw, torch.Tensor):
            flow_2hw = flow_2hw.cpu().numpy()

        flow_resized = _resize_flow(flow_2hw, new_size=(832, 832))
        flow_resized = flow_resized * (832.0 / 512.0)
        flow_cropped = flow_resized[:, self.crop_start:self.crop_start + self.height, :]

        dx = torch.from_numpy(flow_cropped[0]).to(self.device).float()
        dy = torch.from_numpy(flow_cropped[1]).to(self.device).float()

        self.nw(dx, dy, idx=self.frame_count)
        self.noise_buffer.append(self.nw.noise.clone())
        self.frame_count += 1
        return self

    def get_block_noise(self, block_idx):
        """Get noise tensors for one generation block.

        Returns:
            (structured_noise, sde_noise)
            structured_noise: [1, FRAMES_PER_BLOCK, LATENT_C, LATENT_H, LATENT_W]
            sde_noise: [1, FRAMES_PER_BLOCK, LATENT_C, LATENT_H, LATENT_W] or None
        """
        latent_frames = []
        for lf in range(FRAMES_PER_BLOCK):
            global_latent_idx = block_idx * FRAMES_PER_BLOCK + lf
            pixel_idx = (global_latent_idx + 1) * TEMPORAL_FACTOR - 1
            pixel_idx = min(pixel_idx, len(self.noise_buffer) - 1)
            noise_frame = self.noise_buffer[pixel_idx]
            latent_frames.append(noise_frame)

        noise_stack = torch.stack(latent_frames, dim=0)  # [FRAMES_PER_BLOCK, C, 480, 832]

        downscale_factor = self.height // LATENT_H  # 480 // 60 = 8
        noise_latent = torch.nn.functional.interpolate(
            noise_stack,
            size=(LATENT_H, LATENT_W),
            mode="area",
        ) * downscale_factor

        if self.eval_degradation > 0:
            noise_latent = mix_new_noise(noise_latent, self.eval_degradation)

        structured = noise_latent[:, :LATENT_C]
        if self.noise_channels > LATENT_C:
            sde = noise_latent[:, LATENT_C:2 * LATENT_C]
        else:
            sde = None

        structured = structured.unsqueeze(0)
        if sde is not None:
            sde = sde.unsqueeze(0)

        return structured, sde

    def reset(self):
        self.noise_buffer = []
        self.frame_count = 0
        self._init_warper()
