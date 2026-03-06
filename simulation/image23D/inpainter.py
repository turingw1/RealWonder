import torch
from diffusers.utils import load_image, check_min_version
from submodules.flux_controlnet_inpainting.controlnet_flux import FluxControlNetModel
from submodules.flux_controlnet_inpainting.transformer_flux import FluxTransformer2DModel
from submodules.flux_controlnet_inpainting.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
from simulation.utils import dilate_binary_mask, smooth_segmentation_mask_255
import sys
import os
sys.path.append(os.path.abspath("submodules/flux_controlnet_inpainting"))

check_min_version("0.30.2")

class FluxInpainter:
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipe = None
        self.load_model()
        
    def load_model(self):
        """Load the FLUX ControlNet inpainting model and pipeline"""
        # Load ControlNet
        controlnet = FluxControlNetModel.from_pretrained(
            "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", 
            torch_dtype=self.torch_dtype
        )
        
        # Load Transformer
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            subfolder='transformer', 
            torch_dtype=self.torch_dtype
        )
        
        # Build pipeline
        self.pipe = FluxControlNetInpaintingPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=self.torch_dtype
        ).to(self.device)
        
        # Ensure components are in correct dtype
        self.pipe.transformer.to(self.torch_dtype)
        self.pipe.controlnet.to(self.torch_dtype)
        
        print("Model loaded successfully")
        
    def __call__(self, image, mask, prompt="", size=(512, 512), 
                      num_inference_steps=24, controlnet_conditioning_scale=0.9,
                      guidance_scale=3.5, negative_prompt="", true_guidance_scale=3.5,
                      seed=42):
        """Run inpainting with the given parameters"""
        if self.pipe is None:
            raise ValueError("Model not loaded. Please call load_model() first.")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        mask = dilate_binary_mask(mask, size=size, kernel_size=50, iterations=1)
        mask = smooth_segmentation_mask_255(mask, blur_kernel_size=51, blur_sigma=5.0, threshold=60, binary_output=True, morph_close=True, morph_kernel_size=7, return_pil=True)

        image_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        mask_np = np.array(mask)

        mask_3c = np.repeat(mask_np[:, :, None] == 0, 3, axis=2)
        masked_image = np.where(mask_3c, image_np, 255)

        masked_image_pil = Image.fromarray(masked_image)
        # masked_image_pil.save('debug/masked_image.png')

        # Run inpainting
        result = self.pipe(
            prompt=prompt,
            height=size[1],
            width=size[0],
            control_image=masked_image_pil,
            control_mask=mask,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            true_guidance_scale=true_guidance_scale
        ).images[0]
        return result
