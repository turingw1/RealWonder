"""Default configuration constants for the RealWonder interactive demo."""

# Video dimensions
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832

# Latent dimensions (after VAE encoding)
LATENT_H = 60
LATENT_W = 104
LATENT_C = 16

# VAE temporal downsampling factor
TEMPORAL_FACTOR = 4

# Causal generation blocks (model architecture constants)
FRAMES_PER_BLOCK = 3  # latent frames per block
FRAMES_PER_BLOCK_PIXEL = FRAMES_PER_BLOCK * TEMPORAL_FACTOR  # pixel frames per block
FRAMES_FIRST_BLOCK_PIXEL = (FRAMES_PER_BLOCK - 1) * TEMPORAL_FACTOR + 1  # pixel frames for first block

# Playback
FPS = 8

# Simulation parameters are read from each case's config.yaml at runtime
# (dt, substeps, frame_steps) — see InteractiveSimulator.__init__

# Noise warping
NOISE_CHANNELS = 32

# SDEdit
EVAL_DEGRADATION = 0.5

# Model defaults
DEFAULT_LOCAL_ATTN_SIZE = 21
DEFAULT_TIMESTEP_SHIFT = 5.0
CONTEXT_NOISE = 0


def load_case_sdedit_config(case_config: dict) -> dict:
    """Extract SDEdit parameters from a case config.yaml dict.

    Reads num_output_frames, denoising_step_list, mask_dropin_step from the
    case config and computes all derived frame/block counts.

    Returns a dict with keys:
        num_latent_frames, num_pixel_frames, num_blocks,
        denoising_step_list, mask_dropin_step
    """
    num_latent_frames = case_config["num_output_frames"]
    assert num_latent_frames % FRAMES_PER_BLOCK == 0, (
        f"num_output_frames ({num_latent_frames}) must be divisible by "
        f"FRAMES_PER_BLOCK ({FRAMES_PER_BLOCK})"
    )
    return {
        "num_latent_frames": num_latent_frames,
        "num_pixel_frames": (num_latent_frames - 1) * TEMPORAL_FACTOR + 1,
        "num_blocks": num_latent_frames // FRAMES_PER_BLOCK,
        "denoising_step_list": case_config["denoising_step_list"],
        "mask_dropin_step": case_config.get("mask_dropin_step", -1),
        "franka_step": case_config.get("franka_step", -1),
    }
