from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import torch
import sys
import os
import cv2
from pathlib import Path
from repvit_sam import SamAutomaticMaskGenerator, sam_model_registry
import urllib.request
import PIL
from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pyplot as plt
import sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2ImagePredictor


DEBUG_ROOT = Path("debug")


def ensure_debug_dir(*parts: str) -> Path:
    path = DEBUG_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, save_prefix, point_coords=None, box_coords=None, input_labels=None, borders=True):
    debug_dir = ensure_debug_dir("sam2")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(debug_dir / f"{save_prefix}_masks_{i:02d}.png")
        plt.close()


class OneFormerSegmenter:
    def __init__(self, device="cuda"):
        self.device = device
        self.segment_processor = None
        self.segment_model = None
        self.load_model()
        
    def load_model(self):
        """Load the OneFormer model and processor"""
        self.segment_processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large"
        )
        self.segment_model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large"
        ).to(self.device)
        
        print("OneFormer model loaded successfully")
        
    def __call__(self, image):
        """Run semantic segmentation on the given image"""
        if self.segment_processor is None or self.segment_model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        # Check if input_image is a tensor and convert to PIL Image if needed
        if torch.is_tensor(image):
            # Ensure tensor is in correct format [B, C, H, W] or [C, H, W]
            if image.dim() == 4:
                image = image.squeeze(0)  # Remove batch dimension if present
            
            # Convert tensor to PIL Image using ToPILImage()
            if image.dim() == 3:
                image = ToPILImage()(image)
            else:
                raise ValueError(f"Unexpected tensor dimensions: {image.shape}")

        segmenter_input = self.segment_processor(
            image, ["semantic"], return_tensors="pt"
        )
        segmenter_input = {
            name: tensor.to(self.device) for name, tensor in segmenter_input.items()
        }
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
            segment_output, target_sizes=[image.size[::-1]]
        )[0]
        return pred_semantic_map
    
class RepViTSegmenter:
    def __init__(self, device="cuda"):
        self.device = device
        self.load_model()

    def load_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(current_dir, "repvit_sam.pt")
        
        if not os.path.exists(ckpt_path):
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            print(f"Downloading RepViT-SAM checkpoint to {ckpt_path}...")
            urllib.request.urlretrieve(
                "https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt",
                ckpt_path
            )
            print("Checkpoint downloaded successfully")
        model_type = "repvit"
        repvit_sam = sam_model_registry[model_type](checkpoint=ckpt_path)
        repvit_sam = repvit_sam.to(self.device)
        repvit_sam.eval()

        self.repvit_segmenter = SamAutomaticMaskGenerator(
            model=repvit_sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.9,
            # min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

    def __call__(self, image, target_class=[0], merge_mask=False):
        """
        Run RepViT-SAM segmentation to generate instance masks.
        
        Args:
            image: Input image as tensor [B, C, H, W] or [C, H, W] or PIL Image
            target_class: Unused parameter for compatibility
            
        Returns:
            List of dictionaries, each containing:
            - 'segmentation': boolean numpy array of shape (H, W) indicating mask
            - 'area': int, number of pixels in the mask
            - 'bbox': list [x, y, w, h] bounding box coordinates
            - 'predicted_iou': float, predicted IoU score
            - 'point_coords': list of [x, y] coordinates used for prediction
            - 'stability_score': float, stability score of the mask
            - 'crop_box': list [x0, y0, x1, y1] crop box used for prediction
        """
        assert isinstance(image, PIL.Image.Image), f"Image must be a PIL Image, but got {type(image)}"
        image_np = np.array(image)

        output = self.repvit_segmenter.generate(image_np)
        sam_debug_dir = ensure_debug_dir("sam")

        # for debug
        sam_masks_np = []
        for sid, sam_mask in enumerate(output):
            if sam_mask['area'] < 100:
                continue
            sam_masks_np.append(sam_mask['segmentation'])   # (512, 512) bool numpy array
            sam_mask = sam_mask['segmentation'] * 255
            sam_mask = sam_mask.astype(np.uint8)
            cv2.imwrite((sam_debug_dir / f"sam_mask_{sid:02d}.png").as_posix(), sam_mask)
            cv2.imwrite(
                (sam_debug_dir / f"sam_mask_{sid:02d}_rgb.png").as_posix(),
                (sam_mask[:, :, None] / 255).astype(np.uint8) * image_np[:, :, [2, 1, 0]],
            )
        
        # # Dilate the mask(s) using cv2.dilate
        # kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size as needed
        # dilated_sam_masks = []
        # for sid, sam_mask in enumerate(sam_masks_np):
        #     dilated = cv2.dilate(sam_mask.astype(np.uint8), kernel, iterations=1)
        #     dilated_bool = dilated.astype(bool)
        #     dilated_sam_masks.append(dilated_bool)
        #     # Save dilated mask for debug if needed
        #     cv2.imwrite(f"debug/sam_dilated/sam_mask_{sid:02d}_dilated.png", dilated * 255)
        # # Optionally, you may want to replace sam_masks_np with dilated_sam_masks for later processing:
        # sam_masks_np = dilated_sam_masks

        # Dilate the mask(s) before returning
        # kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size as needed
        target_masks_np = []
        for target_id in target_class:
            if merge_mask:
                # If merge_mask is True, combine all masks corresponding to any part_id in target_id
                merged_mask = np.zeros_like(output[0]['segmentation'], dtype=bool)
                for part_id in target_id:
                    if part_id >= 0:
                        merged_mask = np.logical_or(merged_mask, output[part_id]['segmentation'])
                    else:
                        real_part_id = -(part_id + 1)
                        merged_mask = np.logical_or(merged_mask, np.logical_not(output[real_part_id]['segmentation']))
                # Dilate merged_mask
                # dilated_mask = cv2.dilate(merged_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                # target_masks_np.append(dilated_mask)
                target_masks_np.append(merged_mask)
            else:
                # If merge_mask is False, take mask corresponding to target_id (single index)
                if target_id >= 0:
                    mask = output[target_id]['segmentation']
                else:
                    real_part_id = -(target_id + 1)
                    mask = np.logical_not(output[real_part_id]['segmentation'])
                # dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                # target_masks_np.append(dilated_mask)
                target_masks_np.append(mask)

        # import pdb; pdb.set_trace()

        # Optionally, for debugging, you can save the final dilated masks:
        # for tid, dilated_mask in enumerate(target_masks_np):
        #     cv2.imwrite(f"debug/sam_dilated/final_mask_{tid:02d}_dilated.png", dilated_mask.astype(np.uint8) * 255)
        
        return target_masks_np


class SegmentAnythingSegmenter:
    def __init__(self, config, device="cuda"):
        self.device = device
        repo_root = Path(__file__).resolve().parents[2]
        self.sam2_checkpoint = (
            repo_root / "submodules" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
        ).as_posix()
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.config = config

    def __call__(self, image):
        image = np.array(image)
        sam2_debug_dir = ensure_debug_dir("sam2")
        sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)

        all_object_points = self.config['all_object_points']
        all_object_masks_idx = self.config['all_object_masks_idx']

        output_masks = []
        for object_idx, object_points in enumerate(all_object_points):
            # for each object
            object_points = np.array(object_points)
            object_points_xy = object_points[:, :2].copy()
            object_point_labels = object_points[:, 2].copy()

            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_points(object_points_xy, object_point_labels, plt.gca())
            plt.axis('on')
            plt.savefig(sam2_debug_dir / f"input_points_{object_idx:02d}.png")
            plt.close()
        
            masks, scores, logits = predictor.predict(
                point_coords=object_points_xy,
                point_labels=object_point_labels,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            show_masks(image, masks, scores, f"object_{object_idx:02d}", point_coords=object_points_xy, input_labels=object_point_labels, borders=True)
            output_masks.append(masks[all_object_masks_idx[object_idx]])
        
        return output_masks









        


