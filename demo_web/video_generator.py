"""Streaming block-by-block video generation.

Decomposes CausalInferencePipelineSDEdit.inference() into per-block
streaming calls, enabling real-time generation with interactive control.
"""

from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

from vidgen import (
    CausalInferencePipelineSDEdit,
    WanImageEncoder,
    WanVideoVAE,
    WanVideoUnit_ImageEmbedderCLIP,
    WanVideoUnit_ImageEmbedderVAE,
    DynamicSwapInstaller,
    gpu,
    get_cuda_free_memory_gb,
    load_first_frame,
    set_seed,
)
from vidgen.utils import extract_subdim
from gpu_profiler import log_gpu
from config import (
    DEFAULT_HEIGHT, DEFAULT_WIDTH,
    FRAMES_PER_BLOCK, LATENT_C, LATENT_H, LATENT_W,
    DEFAULT_LOCAL_ATTN_SIZE, DEFAULT_TIMESTEP_SHIFT, CONTEXT_NOISE,
)


class StreamingVideoGenerator:
    """Block-by-block video generation with SDEdit support."""

    def __init__(self, checkpoint_path: str, num_pixel_frames: int,
                 denoising_steps: list, device: str = "cuda",
                 use_ema: bool = False, seed: int = 42,
                 mask_dropin_step: int = -1, franka_step: int = -1,
                 enable_taehv: bool = False):
        self.checkpoint_path = checkpoint_path
        self.num_pixel_frames = num_pixel_frames
        self.device = torch.device(device)
        self.use_ema = use_ema
        self.seed = seed
        self.denoising_steps = denoising_steps
        self.mask_dropin_step = mask_dropin_step
        self.franka_step = franka_step
        self.enable_taehv = enable_taehv

        self.pipeline = None
        self.taehv_decoder = None
        self.taehv_cache = None
        self.is_setup = False

    def setup(self):
        """Load models and initialize the pipeline."""
        set_seed(self.seed)
        torch.set_grad_enabled(False)

        low_memory = get_cuda_free_memory_gb(gpu) < 40

        config = OmegaConf.create({
            "independent_first_frame": False,
            "warp_denoising_step": True,
            "context_noise": CONTEXT_NOISE,
            "causal": True,
            "i2v": True,
            "i2v_flow": True,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "num_frames": self.num_pixel_frames,
            "num_frame_per_block": FRAMES_PER_BLOCK,
            "denoising_step_list": self.denoising_steps,
            "mask_dropin_step": self.mask_dropin_step,
            "franka_step": self.franka_step,
            "model_kwargs": {
                "sink_size": 1,
                "local_attn_size": DEFAULT_LOCAL_ATTN_SIZE,
                "timestep_shift": DEFAULT_TIMESTEP_SHIFT,
            },
        })

        log_gpu("before pipeline init")
        self.pipeline = CausalInferencePipelineSDEdit(config, device=self.device,
                                                       use_separate_encode_vae=True)
        log_gpu("after pipeline init (on CPU)")

        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        key = "generator_ema" if self.use_ema else "generator"
        gen_state_dict = state_dict[key]
        try:
            self.pipeline.generator.load_state_dict(gen_state_dict)
        except RuntimeError:
            gen_state_dict = {
                k.replace("._fsdp_wrapped_module", ""): v
                for k, v in gen_state_dict.items()
            }
            self.pipeline.generator.load_state_dict(gen_state_dict)

        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
        log_gpu("after checkpoint load (bf16, CPU)")

        if low_memory:
            DynamicSwapInstaller.install_model(self.pipeline.text_encoder, device=gpu)
        else:
            self.pipeline.text_encoder.to(device=gpu)

        self.pipeline.generator.to(device=gpu)
        self.pipeline.vae.to(device=gpu)
        self.pipeline.encode_vae.to(device=gpu, dtype=torch.bfloat16)

        if self.enable_taehv:
            import os
            import urllib.request
            from taehv import TAEHV

            taehv_path = os.path.join(os.path.dirname(__file__), "checkpoints", "taew2_1.pth")
            taehv_path = os.path.abspath(taehv_path)
            self._ensure_taehv_weights(taehv_path, urllib.request)

            print("Loading TAEHV decoder...")
            self.taehv_decoder = TAEHV(checkpoint_path=taehv_path).to(
                device=gpu, dtype=torch.float16,
            )
            self.taehv_decoder.eval()
            self.taehv_decoder.requires_grad_(False)

        self.pipeline.processor_dtype = torch.float32
        self.pipeline.processor_device = gpu
        self.pipeline.processor_vae = WanVideoVAE().to(device=gpu, dtype=torch.float32)
        self.pipeline.processor_ienc = WanImageEncoder().to(device=gpu, dtype=torch.float32)

        self.pipeline.processor_vae.requires_grad_(False)
        self.pipeline.processor_ienc.requires_grad_(False)

        for p in self.pipeline.processor_vae.parameters():
            p.data = p.data.to(dtype=torch.float32)
        for b in self.pipeline.processor_vae.buffers():
            b.data = b.data.to(dtype=torch.float32)

        self.pipeline.processors = [
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
        ]

        self.is_setup = True
        log_gpu("setup complete")
        print("StreamingVideoGenerator setup complete")

    def _ensure_taehv_weights(self, taehv_path: str, urllib_request):
        """Download TAEHV weights if needed and validate they are loadable.

        Network failures on shared servers often produce HTML or truncated
        files at the target path. Validate eagerly so setup fails with a
        clear message instead of an opaque `torch.load` OSError later.
        """
        path = Path(taehv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/madebyollin/taehv/main/taew2_1.pth"

        def _is_valid_checkpoint(candidate: Path) -> bool:
            if not candidate.exists() or candidate.stat().st_size < 1024 * 1024:
                return False
            try:
                torch.load(candidate, map_location="cpu", weights_only=True)
                return True
            except Exception:
                return False

        if not _is_valid_checkpoint(path):
            if path.exists():
                print(f"Existing TAEHV file looks invalid, removing: {path}")
                path.unlink()

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()

            print(f"Downloading TAEHV weights from {url} ...")
            urllib_request.urlretrieve(url, tmp_path.as_posix())

            if not _is_valid_checkpoint(tmp_path):
                size = tmp_path.stat().st_size if tmp_path.exists() else 0
                raise RuntimeError(
                    "Downloaded TAEHV checkpoint is invalid. "
                    f"path={tmp_path} size={size} bytes. "
                    "This usually means the server downloaded an HTML error page "
                    "or a truncated file. Delete the file and retry, or download "
                    "the checkpoint manually from a stable network."
                )

            tmp_path.replace(path)

    def precompute_first_frame(self, first_frame_path: str, default_prompt: str = ""):
        """Pre-compute all user-input-independent work at startup."""
        assert self.is_setup, "Call setup() first"
        device = self.device
        pipeline = self.pipeline

        log_gpu("precompute_first_frame: start")

        input_image = load_first_frame(first_frame_path, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH)
        batch = {
            "input_image": input_image.unsqueeze(0),
            "end_image": None,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "num_frames": self.num_pixel_frames,
        }
        self.i2v_conditional = {}
        for unit in pipeline.processors:
            input_data = {"device": device, "torch_dtype": pipeline.processor_dtype}
            for key in unit.input_params:
                input_data[key] = batch.get(key)
            for key in unit.onload_model_names:
                if key == "image_encoder":
                    input_data["image_encoder"] = pipeline.processor_ienc
                if key == "vae":
                    input_data["vae"] = pipeline.processor_vae
            unit_output = unit.process(**input_data)
            for k, v in unit_output.items():
                self.i2v_conditional[k] = (
                    v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) else v
                )

        if hasattr(pipeline, 'processor_vae') and pipeline.processor_vae is not None:
            pipeline.processor_vae.cpu()
            del pipeline.processor_vae
            pipeline.processor_vae = None
        if hasattr(pipeline, 'processor_ienc') and pipeline.processor_ienc is not None:
            pipeline.processor_ienc.cpu()
            del pipeline.processor_ienc
            pipeline.processor_ienc = None
        torch.cuda.empty_cache()

        dtype = torch.bfloat16
        pipeline._initialize_kv_cache(batch_size=1, dtype=dtype, device=device)
        pipeline._initialize_crossattn_cache(batch_size=1, dtype=dtype, device=device)

        self.full_y = (
            self.i2v_conditional["y"].permute(0, 2, 1, 3, 4)
            if "y" in self.i2v_conditional else None
        )

        pipeline.vae.model.clear_cache()
        pipeline.encode_vae.model.clear_cache()

        self.default_prompt = default_prompt
        if default_prompt:
            self.default_text_features = pipeline.text_encoder(text_prompts=[default_prompt])
        else:
            self.default_text_features = None

        log_gpu("precompute_first_frame: done")
        print("First frame pre-computation complete")

    def prepare_generation(self, prompt: str):
        """Lightweight per-request preparation: text encode + reset caches."""
        assert self.is_setup, "Call setup() first"
        pipeline = self.pipeline

        if (prompt == self.default_prompt and self.default_text_features is not None):
            text_features = self.default_text_features
        else:
            text_features = pipeline.text_encoder(text_prompts=[prompt])

        self.conditional_dict = dict(text_features)
        for k, v in self.i2v_conditional.items():
            self.conditional_dict[k] = v

        if prompt != getattr(self, "_last_prepared_prompt", None):
            for block_index in range(pipeline.num_transformer_blocks):
                pipeline.crossattn_cache[block_index]["is_init"] = False
            self._last_prepared_prompt = prompt
        for block_index in range(len(pipeline.kv_cache1)):
            pipeline.kv_cache1[block_index]["global_end_index"].fill_(0)
            pipeline.kv_cache1[block_index]["local_end_index"].fill_(0)

        self.current_start_frame = 0
        self.taehv_cache = None

        pipeline.vae.model.clear_cache()
        pipeline.encode_vae.model.clear_cache()

        print("Generation prepared")

    def generate_block(self, block_idx: int, structured_noise: torch.Tensor,
                       sim_latent: torch.Tensor,
                       sde_noise: Optional[torch.Tensor] = None,
                       sim_mask: Optional[torch.Tensor] = None,
                       sim_franka_mask: Optional[torch.Tensor] = None) -> List[np.ndarray]:
        """Generate and decode one block of video."""
        pipeline = self.pipeline
        device = self.device
        num_frames = FRAMES_PER_BLOCK

        structured_noise = structured_noise.to(device=device, dtype=torch.bfloat16)
        sim_latent = sim_latent.to(device=device, dtype=torch.bfloat16)
        if sde_noise is not None:
            sde_noise = sde_noise.to(device=device, dtype=torch.bfloat16)
        if sim_mask is not None:
            sim_mask = sim_mask.to(device=device)
        if sim_franka_mask is not None:
            sim_franka_mask = sim_franka_mask.to(device=device)

        sdedit_step = pipeline.denoising_step_list[0]
        noisy_input = pipeline.scheduler.add_noise(
            sim_latent.flatten(0, 1),
            structured_noise.flatten(0, 1),
            sdedit_step * torch.ones([num_frames], device=device, dtype=torch.long),
        ).unflatten(0, (1, num_frames))

        bg_noisy_input = None
        if pipeline.mask_dropin_step > 0 and sim_mask is not None:
            mask_step = pipeline.denoising_step_list[pipeline.mask_dropin_step]
            bg_noisy_input = pipeline.scheduler.add_noise(
                sim_latent.flatten(0, 1),
                noisy_input.flatten(0, 1),
                mask_step * torch.ones([num_frames], device=device, dtype=torch.long),
            ).unflatten(0, (1, num_frames))

        bg_noisy_franka = None
        use_franka = (
            pipeline.franka_step >= 0
            and sim_franka_mask is not None
            and sim_franka_mask.any()
        )
        if use_franka:
            franka_step = pipeline.denoising_step_list[pipeline.franka_step]
            bg_noisy_franka = pipeline.scheduler.add_noise(
                sim_latent.flatten(0, 1),
                noisy_input.flatten(0, 1),
                franka_step * torch.ones([num_frames], device=device, dtype=torch.long),
            ).unflatten(0, (1, num_frames))

        curr_y = None
        if self.full_y is not None:
            start = self.current_start_frame
            curr_y = self.full_y[:, start:start + num_frames]

        for index, current_timestep in enumerate(pipeline.denoising_step_list):
            timestep = torch.ones(
                [1, num_frames], device=device, dtype=torch.int64
            ) * current_timestep

            if (pipeline.mask_dropin_step > 0
                    and pipeline.mask_dropin_step == index
                    and sim_mask is not None
                    and bg_noisy_input is not None):
                noisy_input = torch.where(
                    sim_mask.unsqueeze(2),
                    noisy_input, bg_noisy_input,
                )

            if (use_franka and pipeline.franka_step == index and bg_noisy_franka is not None):
                noisy_input = torch.where(
                    sim_franka_mask.unsqueeze(2),
                    bg_noisy_franka, noisy_input,
                )

            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=self.conditional_dict,
                curr_y=curr_y,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=self.current_start_frame * pipeline.frame_seq_length,
            )

            if index < len(pipeline.denoising_step_list) - 1:
                next_step = pipeline.denoising_step_list[index + 1]
                if sde_noise is not None:
                    sde_n = extract_subdim(sde_noise, LATENT_C, return_complement=False, channel_dim=2)
                else:
                    sde_n = torch.randn_like(noisy_input)
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    sde_n.flatten(0, 1),
                    next_step * torch.ones([num_frames], device=device, dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])

        context_timestep = torch.ones_like(timestep) * pipeline.args.context_noise
        pipeline.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=self.conditional_dict,
            curr_y=curr_y,
            timestep=context_timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=self.current_start_frame * pipeline.frame_seq_length,
        )

        self.current_start_frame += num_frames

        if self.enable_taehv:
            if self.taehv_cache is None:
                decode_input = denoised_pred
                self.taehv_cache = denoised_pred
            else:
                decode_input = torch.cat([self.taehv_cache, denoised_pred], dim=1)
                self.taehv_cache = decode_input[:, -3:, :, :, :]
            video = self.taehv_decoder.decode_video(
                decode_input.to(dtype=torch.float16), parallel=True,
            )
            if block_idx == 0:
                video = video[:, 3:]
            else:
                video = video[:, 12:]
            video = video.clamp(0, 1)
        else:
            video = pipeline.vae.decode_to_pixel(denoised_pred, use_cache=True)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        video = rearrange(video, "b t c h w -> b t h w c").cpu()
        frames = (255.0 * video[0]).to(torch.uint8).numpy()
        return [frames[i] for i in range(frames.shape[0])]

    def reset(self):
        """Reset generation state, preserving KV cache allocations."""
        if self.pipeline is not None:
            pipeline = self.pipeline
            if pipeline.kv_cache1 is not None:
                for block_index in range(len(pipeline.kv_cache1)):
                    pipeline.kv_cache1[block_index]["global_end_index"].fill_(0)
                    pipeline.kv_cache1[block_index]["local_end_index"].fill_(0)
            pipeline.vae.model.clear_cache()
            pipeline.encode_vae.model.clear_cache()
        self.current_start_frame = 0
        self.conditional_dict = None
        self.taehv_cache = None
