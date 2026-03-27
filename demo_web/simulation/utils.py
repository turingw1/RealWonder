import torch
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from typing import List, Optional, Tuple
from torchvision.io import write_video
from torchvision.transforms.functional import pil_to_tensor
import os
from glob import glob
import imageio.v2 as imageio

PRESET_Z_VALUE = 0.0


def pt3d_to_gs(xyz, no_z_offset=False):
    z_offset = 0.0 if no_z_offset else PRESET_Z_VALUE
    if isinstance(xyz, torch.Tensor):
        xyz_new = xyz.clone()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    elif isinstance(xyz, np.ndarray):
        xyz_new = xyz.copy()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    else:
        raise ValueError(f"Input type {type(xyz)} is not supported")
    xyz_new[..., 2] += z_offset
    return xyz_new


def gs_to_pt3d(xyz, no_z_offset=False):
    z_offset = 0.0 if no_z_offset else PRESET_Z_VALUE
    if isinstance(xyz, torch.Tensor):
        xyz_new = xyz.clone()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    elif isinstance(xyz, np.ndarray):
        xyz_new = xyz.copy()
        xyz_new[..., 0] = -1 * xyz[..., 0]
        xyz_new[..., 1] = xyz[..., 2]
        xyz_new[..., 2] = xyz[..., 1]
    else:
        raise ValueError(f"Input type {type(xyz)} is not supported")
    xyz_new[..., 1] -= z_offset
    return xyz_new


def pose_to_transform_matrix(pos, quat):
    """Convert position and quaternion to 4x4 transformation matrix for Genesis."""
    if hasattr(pos, 'cpu'):
        pos = pos.cpu().numpy()
    if hasattr(quat, 'cpu'):
        quat = quat.cpu().numpy()
    pos = np.array(pos, dtype=np.float64)
    quat = np.array(quat, dtype=np.float64)
    quat_scipy = quat[[1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]
    quat_scipy = quat_scipy / np.linalg.norm(quat_scipy)
    rot = Rotation.from_quat(quat_scipy)
    rot_matrix = rot.as_matrix()
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[:3, 3] = pos
    return transform_matrix


def resize_and_crop_pil(image: Image.Image, start_y=None) -> Image.Image:
    width, height = image.size
    assert width == 512 and height == 512, f"Expected 512x512 image, got {width}x{height}"
    resized_image = image.resize((832, 832), resample=Image.BILINEAR)
    crop_width = 832
    crop_height = 480
    start_x = 0
    if start_y is None:
        start_y = (832 - crop_height) // 2
    cropped_image = resized_image.crop((start_x, start_y, start_x + crop_width, start_y + crop_height))
    return cropped_image


def save_video_from_pil(
    frames: List[Image.Image],
    out_path: str,
    fps: int = 16,
    size: Optional[Tuple[int, int]] = None,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    yuv420p: bool = True,
) -> None:
    if not frames:
        raise ValueError("frames is empty")
    if size is None:
        size = (frames[0].width, frames[0].height)
    W, H = size
    tensor_frames = []
    for im in frames:
        im = im.convert("RGB")
        if (im.width, im.height) != (W, H):
            im = im.resize((W, H), Image.BICUBIC)
        t = pil_to_tensor(im).permute(1, 2, 0).contiguous()
        tensor_frames.append(t)
    video = torch.stack(tensor_frames, dim=0)
    options = {"crf": str(crf), "preset": preset}
    if yuv420p:
        options["pix_fmt"] = "yuv420p"
    write_video(filename=out_path, video_array=video, fps=fps, video_codec=codec, options=options)


def save_gif_from_image_folder(input_folder, gif_path, duration=0.1):
    image_exts = ('*.png', '*.jpg', '*.jpeg')
    input_images = []
    for ext in image_exts:
        input_images.extend(glob(os.path.join(input_folder, ext)))
    input_images = sorted(input_images)
    if not input_images:
        print("No images found in input folder.")
        return
    frames = []
    for img_path in input_images:
        try:
            img = imageio.imread(img_path)
            frames.append(img)
        except Exception as e:
            print(f"[ERROR] Skipping {img_path}: {e}")
    if frames:
        imageio.mimsave(gif_path, frames, duration=duration, loop=0)
        print(f"GIF saved to {gif_path}")
    else:
        print("No valid images to save as GIF.")
