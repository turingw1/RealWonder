import torch
from PIL import Image
import math
import os
import numpy as np
from einops import rearrange
import trimesh
import cv2
from moge.model.v1 import MoGeModel
from typing import NamedTuple, Sequence, Union

from torchvision import utils as torchvision_utils
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend, hard_rgb_blend
import torch.nn as nn
import matplotlib.pyplot as plt

from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.renderer import (
    PointsRenderer, PointsRasterizer, PointsRasterizationSettings, AlphaCompositor,
    MeshRenderer, MeshRasterizer, RasterizationSettings, SoftPhongShader,
    PerspectiveCameras, BlendParams, PointLights, TexturesVertex, mesh, HardFlatShader, Textures, NormWeightedCompositor
)
from kornia.geometry import PinholeCamera
from pathlib import Path

from simulation.image23D.segmenter import RepViTSegmenter, SegmentAnythingSegmenter
from simulation.image23D.mesh_generator import Sam3DMeshGenerator
from simulation.image23D.inpainter import FluxInpainter

from pytorch3d.renderer.mesh.textures import TexturesVertex

from simulation.utils import (
    soft_stitching,
    dilate_binary_mask,
    extract_foreground_depth_torch,
    save_point_cloud_as_ply,
    save_depth_map,
    save_mask_kps,
    remove_isolated_areas,
    render_mesh_with_occlusion_detection,
    create_occluded_submesh,
    # sample_mesh_surface,
    # match_color_style,
)

class HardShader(nn.Module):
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = (
            blend_params if blend_params is not None else MyBlendParams()
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"
            raise ValueError(msg)
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        # images = softmax_rgb_blend(texels, fragments, blend_params)
        images = hard_rgb_blend(texels, fragments, blend_params)

        return images

class MyBlendParams(NamedTuple):
    """
    Data class to store blending params with defaults

    Members:
        sigma (float): For SoftmaxPhong, controls the width of the sigmoid
            function used to calculate the 2D distance based probability. Determines
            the sharpness of the edges of the shape. Higher => faces have less defined
            edges. For SplatterPhong, this is the standard deviation of the Gaussian
            kernel. Higher => splats have a stronger effect and the rendered image is
            more blurry.
        gamma (float): Controls the scaling of the exponential function used
            to set the opacity of the color.
            Higher => faces are more transparent.
        background_color: RGB values for the background color as a tuple or
            as a tensor of three floats.
    """

    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Union[torch.Tensor, Sequence[float]] = (0.0, 0.0, 0.0)


# pytorch3d space
class SingleViewReconstructor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_size = (512, 512)
        self.config = config
        self.device = config['device']
        
        self.input_image_pil = Image.open(os.path.join(config['data_path'], 'input.png')).convert('RGB')
        self.input_image = ToTensor()(self.input_image_pil).to(self.device)
        self.output_folder = Path(config['output_folder']) / 'render'
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder_frames = self.output_folder / 'frames'
        self.output_folder_frames.mkdir(parents=True, exist_ok=True)
        self.output_folder_masks = self.output_folder / 'masks'
        self.output_folder_masks.mkdir(parents=True, exist_ok=True)
        self.output_folder_optical_flow = self.output_folder / 'optical_flow'
        self.output_folder_optical_flow.mkdir(parents=True, exist_ok=True)

        self.previous_frame_data = None
        self.optical_flow = np.array([])

        self.franka_mesh = None
        self.merge_mask = True if 'merge_mask' in self.config and self.config['merge_mask'] else False

        self.fg_objects = []
        self.cache_bg = None

    def reconstruct(self):
        if 'segmenter' not in self.config or self.config['segmenter'] == "repvit":
            self.object_id = self.config['object_id']
            self.segmenter = RepViTSegmenter(self.device)
            target_masks = self.segmenter(self.input_image_pil, target_class=self.object_id, merge_mask=self.merge_mask)
        elif self.config['segmenter'] == "sam2":
            self.segmenter = SegmentAnythingSegmenter(self.config, self.device)
            target_masks = self.segmenter(self.input_image_pil)
        else:
            raise ValueError(f"Invalid segmenter: {self.config['segmenter']}")

        self.object_masks = [torch.from_numpy(mask).to(self.device) for mask in target_masks]

        inpainted_image_path = os.path.join(self.config['data_path'], 'inpainted.png')
        if os.path.exists(inpainted_image_path):
            self.inpainted_image = ToTensor()(Image.open(inpainted_image_path).convert('RGB')).to(self.device)
        else:
            inpainter = FluxInpainter(device=self.device)
            all_objects_masks = torch.zeros_like(self.object_masks[0], dtype=torch.bool)
            for mask in self.object_masks:
                all_objects_masks = all_objects_masks | mask.bool()
            # convert all_objects_masks to PIL image
            # all_objects_masks = ToPILImage()(all_objects_masks.cpu().numpy().astype(np.uint8) * 255)
            if self.config.get('debug', False):
                # all_objects_masks.save(os.path.join(self.config['output_folder'], 'inpainter_masks.png'))
                torchvision_utils.save_image(all_objects_masks.float(), os.path.join(self.config['output_folder'], 'inpainter_masks.png'))
                
            self.inpainted_image_pil = inpainter(self.input_image, all_objects_masks, prompt=self.config['inpainting_prompt'], negative_prompt=self.config['inpainting_negative_prompt'])
            self.inpainted_image = ToTensor()(self.inpainted_image_pil).to(self.device)
            if self.config.get('debug', False):
                self.inpainted_image_pil.save(os.path.join(self.config['output_folder'], 'inpainted_image.png'))
            
            del inpainter
            torch.cuda.empty_cache()

        if 'stitched_inpainting' in self.config and self.config['stitched_inpainting']:
            # all_dilated_masks = [torch.from_numpy(dilate_binary_mask(per_mask, size=(512, 512), kernel_size=3, iterations=1)).unsqueeze(0).unsqueeze(0).to(self.device) for per_mask in self.object_masks]
            self.inpainted_image = soft_stitching(self.inpainted_image.unsqueeze(0), self.input_image.unsqueeze(0), [per_mask.unsqueeze(0).unsqueeze(0) for per_mask in self.object_masks]).squeeze(0)
            # self.inpainted_image = soft_stitching(self.inpainted_image.unsqueeze(0), self.input_image.unsqueeze(0), all_dilated_masks).squeeze(0)
        
        if self.config.get('debug', False):
            torchvision_utils.save_image(self.inpainted_image, os.path.join(self.config['output_folder'], 'stitched_inpainted_image.png'))

        self.mesh_generator = Sam3DMeshGenerator(self.config)
        self.fg_meshes = []
        self.fg_pcs = []
        for idx, per_mask in enumerate(self.object_masks):

            if 'refine_mask' in self.config and self.config['refine_mask']:
                min_size = self.config['min_size'] if 'min_size' in self.config else 100
                per_mask = torch.from_numpy(remove_isolated_areas(per_mask.cpu().numpy(), min_size=min_size)).to(self.device)
                if self.config.get('debug', False):
                    torchvision_utils.save_image(
                        per_mask.float(),
                        os.path.join(self.config['output_folder'], f"refined_mask_{idx:02d}.png")
                    )
            else:
                print(f"Refine mask is disabled, using original mask")

            original_mesh, simplified_mesh, fx_pixels, fx_deg, _ = self.mesh_generator(np.array(self.input_image_pil), per_mask.cpu().numpy(), mesh_resize_factor=self.config['mesh_resize_factor'], target_faces=self.config['target_faces'])


            if idx == 0:
                self.init_focal_length = fx_pixels
                self.config['fov_x_input'] = fx_deg.item()
                self.current_camera = self.get_camera_at_origin()

                # background point cloud
                moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)
                moge_model.eval()
                with torch.no_grad():
                    depth_inpainted = moge_model.infer(self.inpainted_image, fov_x=fx_deg)['depth']
                    depth_input = moge_model.infer(self.input_image, fov_x=fx_deg)['depth']
                    mask_noninf_inpainted = ~torch.isinf(depth_inpainted)
                    mask_noninf_input = ~torch.isinf(depth_input)

                    if mask_noninf_inpainted.any():
                        max_val_inpainted = depth_inpainted[mask_noninf_inpainted].max()
                        depth_inpainted = depth_inpainted.clone()
                        depth_inpainted[~mask_noninf_inpainted] = max_val_inpainted
                    if mask_noninf_input.any():
                        max_val_input = depth_input[mask_noninf_input].max()
                        depth_input = depth_input.clone()
                        depth_input[~mask_noninf_input] = max_val_input
                    if 'remap_depth' in self.config:
                        depth_inpainted = self.remap_depth(depth_inpainted, self.config['remap_depth'], mask_noninf_inpainted)
                        depth_input = self.remap_depth(depth_input, self.config['remap_depth'], mask_noninf_input)
                
                if self.config.get('debug', False):
                    save_depth_map(depth_inpainted.cpu().numpy(), os.path.join(self.config['output_folder'], f"depth_inpainted_{idx:02d}.png"))
                    save_depth_map(depth_input.cpu().numpy(), os.path.join(self.config['output_folder'], f"depth_input_{idx:02d}.png"))
                
                self.bg_points, self.bg_points_colors = self.depth2pc(depth_inpainted, self.inpainted_image)
                self.input_image_points, self.input_image_colors = self.depth2pc(depth_input, self.input_image)


            if 'obj_kp_matching' in self.config and self.config['obj_kp_matching']:
                # scale = 1.0
                # translation = np.array([-0.01, 0.08, 0.0])
                # simplified_mesh.vertices = simplified_mesh.vertices * scale + translation
                # original_mesh.vertices = original_mesh.vertices * scale + translation

                scale, translation = self.obj_kp_matching(per_mask, torch.from_numpy(original_mesh.vertices).to(self.device).float(), torch.from_numpy(original_mesh.faces).to(self.device).long(), idx)
                simplified_mesh.vertices = simplified_mesh.vertices * scale.item() + translation.cpu().numpy()
                original_mesh.vertices = original_mesh.vertices * scale.item() + translation.cpu().numpy()
            

            per_mask_from_mesh, depth_map, occluded_vertices_mask = render_mesh_with_occlusion_detection(torch.from_numpy(original_mesh.vertices).to(self.device).float(), torch.from_numpy(original_mesh.faces).to(self.device).long(), torch.from_numpy(original_mesh.visual.vertex_colors).to(self.device).float()[:,:3]/255.0, self.current_camera)
            occluded_submesh_vertices, occluded_submesh_faces, occluded_submesh_colors = create_occluded_submesh(torch.from_numpy(original_mesh.vertices).to(self.device).float(), torch.from_numpy(original_mesh.faces).to(self.device).long(), torch.from_numpy(original_mesh.visual.vertex_colors).to(self.device).float()[:,:3]/255.0, occluded_vertices_mask)
            per_points, per_colors = self.depth2pc(depth_map, self.input_image, per_mask_from_mesh)

            # occluded_submesh_colors[:] = torch.tensor([216/255.0, 190/255.0, 150/255.0], device=occluded_submesh_colors.device).unsqueeze(0).expand_as(occluded_submesh_colors)

            if self.config.get('use_rgb_frontside', True):
                merged_per_points = torch.cat([per_points, occluded_submesh_vertices], dim=0)
                merged_per_colors = torch.cat([per_colors, occluded_submesh_colors], dim=0)
            else:
                merged_per_points = torch.from_numpy(original_mesh.vertices).to(self.device).float()
                merged_per_colors = torch.from_numpy(original_mesh.visual.vertex_colors).to(self.device).float()[:,:3]/255.0

            self.fg_meshes.append(
                {
                    'vertices': torch.from_numpy(simplified_mesh.vertices).to(self.device).float(),
                    'faces': torch.from_numpy(simplified_mesh.faces).to(self.device).long(),
                    'colors': torch.from_numpy(simplified_mesh.visual.vertex_colors).to(self.device).float()[:,:3]/255.0
                }
            )

            self.fg_pcs.append(
                # {
                #     'points': torch.from_numpy(original_mesh.vertices).to(self.device).float(),
                #     'colors': torch.from_numpy(original_mesh.visual.vertex_colors).to(self.device).float()[:,:3]/255.0
                # }
                {
                    'points': merged_per_points,
                    'colors': merged_per_colors
                }
            )

            if self.config.get('debug', False):
                save_point_cloud_as_ply(
                    merged_per_points.cpu(),
                    merged_per_colors.cpu(),
                    os.path.join(self.config['output_folder'], f"merged_per_points_{idx:02d}.ply")
                )

            if self.config.get('debug', False):
                original_mesh.export(os.path.join(self.config['output_folder'], f"sam3d_mesh_{idx:02d}.obj"))
                simplified_mesh.export(os.path.join(self.config['output_folder'], f"sam3d_mesh_{idx:02d}_simplified.obj"))

        if self.config.get('debug', False):
            save_point_cloud_as_ply(self.bg_points, self.bg_points_colors, os.path.join(self.config['output_folder'], 'projected_bg_points.ply'))

        # self.render(render_bg=True, render_obj=True, render_mesh=True, frame_id=0, save=True, mask=True)
        # import pdb; pdb.set_trace()

        self.num_fg_objects = len(self.fg_pcs)
        
        self.ground_plane_normal = None

        if 'estimate_plane' in self.config and self.config['estimate_plane']:
            self.ground_plane_normal = self.estimate_plane_normal_simple(self.fg_pcs[-1]['points'].cpu().numpy())
            if self.ground_plane_normal[1] < 0:
                self.ground_plane_normal = -self.ground_plane_normal
            self.fg_pcs = self.fg_pcs[:-1]
            self.fg_meshes = self.fg_meshes[:-1]

        return self.fg_pcs, self.fg_meshes, self.ground_plane_normal, self.config
        
    def depth2pc(self, depth_map, image, mask=None):
        # initialize the point cloud for background
        kf_camera = self.convert_pytorch3d_kornia(self.current_camera, self.init_focal_length)
        point_depth = rearrange(depth_map.unsqueeze(0), "c h w -> (w h) c")
        # Set all inf values in point_depth to 6
        # point_depth[point_depth == float('inf')] = 6

        x = torch.arange(self.target_size[0]).float() + 0.5
        y = torch.arange(self.target_size[1]).float() + 0.5

        points_cloud = torch.stack(torch.meshgrid(x, y, indexing="ij"), -1)
        points_cloud = rearrange(points_cloud, "h w c -> (h w) c").to(self.device)

        unprojected_points = kf_camera.unproject(points_cloud, point_depth)
        points_colors = rearrange(image, "c h w -> (w h) c")

        if mask is not None:
            mask = rearrange(mask, "h w -> (w h)")
            unprojected_points = unprojected_points[mask]
            points_colors = points_colors[mask]

        return unprojected_points, points_colors

    @torch.no_grad()
    def get_camera_at_origin(self):
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = self.init_focal_length
        K[0, 1, 1] = self.init_focal_length
        K[0, 0, 2] = 256
        K[0, 1, 2] = 256
        K[0, 3, 2] = 1
        K[0, 2, 3] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        camera = PerspectiveCameras(
            K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device
        )
        return camera
    
    def convert_pytorch3d_kornia(self, camera, focal_length, size=512, update_intrinsics_parameters=None, new_size=512):
        transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
        transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)

        pt3d_to_kornia = torch.diag(torch.tensor([-1.0, -1, 1, 1], device=camera.device))
        transform_matrix_w2c_kornia = pt3d_to_kornia @ transform_matrix_w2c_pt3d

        extrinsics = transform_matrix_w2c_kornia.unsqueeze(0)
        h = torch.tensor([size], device="cuda")
        w = torch.tensor([size], device="cuda")
        K = torch.eye(4)[None].to("cuda")
        K[0, 0, 2] = size // 2
        K[0, 1, 2] = size // 2
        K[0, 0, 0] = focal_length
        K[0, 1, 1] = focal_length
        if update_intrinsics_parameters is not None:
            u0, v0, w_crop, h_crop, p_left, p_right, p_up, p_down, scale = (
                update_intrinsics_parameters
            )
            new_cx = (K[0, 0, 2] - u0 + p_left) * scale
            new_cy = (K[0, 1, 2] - v0 + p_up) * scale
            new_fx = K[0, 0, 0] * scale
            new_fy = K[0, 1, 1] * scale
            K[0, 0, 2] = new_cx
            K[0, 1, 2] = new_cy
            K[0, 0, 0] = new_fx
            K[0, 1, 1] = new_fy
            new_h = torch.tensor([new_size], device="cuda")
            new_w = torch.tensor([new_size], device="cuda")
            return PinholeCamera(K, extrinsics, new_h, new_w)

        return PinholeCamera(K, extrinsics, h, w)

    def render(self, render_bg=True, render_obj=True, render_mesh=True, frame_id=0, save=True, mask=True, 
            compute_optical_flow=True):
        """
        Render function with optical flow support based on the original Gaussian splatting logic.
        
        Args:
            render_bg: Whether to render background
            render_obj: Whether to render foreground objects  
            render_mesh: Whether to render mesh
            frame_id: Current frame ID
            save: Whether to save outputs
            mask: Whether to save masks
            compute_optical_flow: Whether to compute optical flow
            prev_frame_data: Dictionary containing previous frame's point positions and camera
                            Format: {
                                'fg_points': previous frame foreground points,
                                'camera': previous frame camera,
                                'bg_points': previous frame background points (optional)
                            }
        
        Returns:
            image_pil: Rendered image
            fg_points_mask: Foreground points mask
            mesh_mask: Mesh mask  
            optical_flow: Optical flow (H, W, 3) if compute_optical_flow=True, else None (third channel is 0 for foreground)
        """
        cameras = self.current_camera
        image_size = self.target_size[0]
        optical_flow = None

        ### 1. Render background point cloud
        if render_bg and self.cache_bg is None:
            bg_pc = Pointclouds(
                points=[self.bg_points],
                features=[self.bg_points_colors]
            )
            bg_raster_settings = PointsRasterizationSettings(
                image_size=image_size,
                radius= 0.0001 if 'bg_points_render_radius' not in self.config else self.config['bg_points_render_radius'],
                points_per_pixel=30
            )
            bg_renderer = PointsRenderer(
                rasterizer=PointsRasterizer(cameras=cameras, raster_settings=bg_raster_settings),
                compositor=AlphaCompositor()
            )
            bg_image = bg_renderer(bg_pc)
            self.cache_bg = bg_image
        elif render_bg and self.cache_bg is not None:
            bg_image = self.cache_bg
        else:
            bg_image = torch.zeros(1, image_size, image_size, 3, device=self.device)

        base_rgb = bg_image[0].clone()
        final_rgb = base_rgb.clone()

        ### 2. Render foreground point clouds
        all_fg_points = []
        all_fg_colors = []
        
        for pc_info in self.fg_pcs:
            points = pc_info['points']
            colors = pc_info['colors']
            
            all_fg_points.append(points)
            all_fg_colors.append(colors)
        
        combined_fg_points = torch.cat(all_fg_points, dim=0)
        combined_fg_colors = torch.cat(all_fg_colors, dim=0)

        flow_rendered_points = combined_fg_points.clone()
        
        alpha = 1.0
        combined_rgba = torch.cat([
            combined_fg_colors,
            alpha * torch.ones_like(combined_fg_colors[..., :1])
        ], dim=-1)
        
        fg_pc = Pointclouds(
            points=[combined_fg_points],
            features=[combined_rgba]
        )
        
        fg_raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=0.01 if 'fg_points_render_radius' not in self.config else self.config['fg_points_render_radius'],
            points_per_pixel=30,
            max_points_per_bin = 20000,
            bin_size=0,
        )
        
        fg_rasterizer = PointsRasterizer(cameras=cameras, raster_settings=fg_raster_settings)
        fg_renderer = PointsRenderer(
            rasterizer=fg_rasterizer,
            compositor=AlphaCompositor()
        )
        
        fg_image = fg_renderer(fg_pc)
        fg_rgb = fg_image[0, ..., :3]
        fg_alpha = fg_image[0, ..., 3:4]
        
        fragments = fg_rasterizer(fg_pc)
        fg_depth = fragments.zbuf[0, ..., 0]
        
        fg_points_mask = torch.where(fg_alpha.squeeze(-1) > self.config['alpha_threshold'], 1.0, 0.0).unsqueeze(-1)
        
        fg_mask_2d = fg_points_mask.squeeze(-1)
        final_rgb = fg_rgb * fg_mask_2d.unsqueeze(-1) + final_rgb * (1.0 - fg_mask_2d.unsqueeze(-1))

        ### 4. Render mesh
        mesh_mask = torch.zeros(image_size, image_size, 1, dtype=torch.float32, device=self.device)
        
        if render_mesh and self.franka_mesh is not None:
            from pytorch3d.renderer import (
                MeshRenderer, MeshRasterizer, SoftPhongShader,
                RasterizationSettings, BlendParams
            )
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer.mesh.textures import TexturesVertex

            vertices = self.franka_mesh['vertices']
            faces = self.franka_mesh['faces']
            colors = self.franka_mesh['colors']

            flow_rendered_points = torch.cat([flow_rendered_points, vertices], dim=0)
            
            if not isinstance(vertices, torch.Tensor):
                vertices = torch.tensor(vertices, dtype=torch.float32, device=self.device)
            if not isinstance(faces, torch.Tensor):
                faces = torch.tensor(faces, dtype=torch.long, device=self.device)
            if not isinstance(colors, torch.Tensor):
                colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
            
            vertices = vertices.to(self.device)
            faces = faces.to(self.device)
            colors = colors.to(self.device)
            
            textures = TexturesVertex(verts_features=[colors])
            combined_mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
                
            mesh_raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=10,
            )
            
            mesh_rasterizer = MeshRasterizer(cameras=cameras, raster_settings=mesh_raster_settings)
            mesh_renderer = MeshRenderer(
                rasterizer=mesh_rasterizer,
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=cameras,
                    blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))
                )
            )
            
            mesh_image = mesh_renderer(combined_mesh)
            mesh_rgb = mesh_image[0, ..., :3]
            mesh_alpha = mesh_image[0, ..., 3:4]
            
            mesh_fragments = mesh_rasterizer(combined_mesh)
            mesh_depth = mesh_fragments.zbuf[0, ..., 0]
            
            mesh_mask_2d = torch.where(mesh_alpha.squeeze(-1) > 0.01, 1.0, 0.0)
            
            fg_depth_valid = torch.where(fg_mask_2d > 0, fg_depth, torch.tensor(float('inf'), device=self.device))
            mesh_depth_valid = torch.where(mesh_mask_2d > 0, mesh_depth, torch.tensor(float('inf'), device=self.device))
            
            mesh_closer_bool = (mesh_depth_valid < fg_depth_valid) & (mesh_mask_2d > 0)
            mesh_closer_float = mesh_closer_bool.float()
            mesh_mask = mesh_closer_float.unsqueeze(-1)
            
            mesh_closer_3d = mesh_closer_float.unsqueeze(-1)
            final_rgb = mesh_rgb * mesh_closer_3d + final_rgb * (1.0 - mesh_closer_3d)
            
            fg_points_mask = torch.where(mesh_closer_bool.unsqueeze(-1), 
                                    torch.zeros_like(fg_points_mask), 
                                    fg_points_mask)


        # 3. Compute optical flow if requested (following original logic)
        if compute_optical_flow and self.previous_frame_data is not None:
            
            optical_flow = self._compute_optical_flow_pytorch3d_style(
                current_fg_points=flow_rendered_points,
                prev_fg_points=self.previous_frame_data['flow_rendered_points'],
                current_camera=cameras,
                prev_camera=self.previous_frame_data['camera'],
                image_size=image_size,
                frame_id=frame_id
            )
            
            if self.optical_flow.size == 0:
                self.optical_flow = np.expand_dims(optical_flow.cpu().numpy(), 0)
            else:
                self.optical_flow = np.concatenate([self.optical_flow, np.expand_dims(optical_flow.cpu().numpy(), 0)])

        ### 5. Save outputs
        if mask and save:
            
            points_mask_path = self.output_folder_masks / f"points_mask_{frame_id:04d}.png"
            points_mask_to_save = fg_points_mask.squeeze(2) if fg_points_mask.dim() == 3 else fg_points_mask
            ToPILImage()(points_mask_to_save.unsqueeze(0).clamp(0, 1).cpu()).save(points_mask_path.as_posix())
            
            mesh_mask_path = self.output_folder_masks / f"mesh_mask_{frame_id:04d}.png"
            mesh_mask_to_save = mesh_mask.squeeze(2) if mesh_mask.dim() == 3 else mesh_mask
            ToPILImage()(mesh_mask_to_save.unsqueeze(0).clamp(0, 1).cpu()).save(mesh_mask_path.as_posix())
        
        # if save and compute_optical_flow and optical_flow is not None:
        #     self._save_optical_flow(optical_flow, frame_id)
        
        image_pil = ToPILImage()(final_rgb.permute(2, 0, 1).clamp(0, 1).cpu())
        if save:
            image_path = self.output_folder_frames / f"frame_{frame_id:04d}.png"
            image_pil.save(image_path.as_posix())
        
        self.previous_frame_data = {
            'camera': cameras,
            'bg_points': self.bg_points,
            'flow_rendered_points': flow_rendered_points
        }

        return image_pil, fg_points_mask, mesh_mask


    def save_optical_flow(self, optical_flow, valid_mask, frame_id):

        # Extract flow components
        flow_x = optical_flow[:, :, 0].cpu().numpy()
        flow_y = optical_flow[:, :, 1].cpu().numpy()
        valid_mask_np = valid_mask.cpu().numpy()
        
        # Convert flow to HSV color representation
        angle = np.arctan2(-flow_y, flow_x)
        
        # Create HSV image
        hsv = np.zeros((optical_flow.shape[0], optical_flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 179
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        # magnitude = np.sqrt(flow_x**2 + flow_y**2)
        # hsv[..., 2] = np.clip(magnitude * 255 / np.max(magnitude), 0, 255).astype(np.uint8)
        
        # Apply valid mask
        hsv[~valid_mask_np] = 0
        
        # Convert HSV to RGB
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Create color wheel
        def create_color_wheel(size=256):
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, y)
            
            magnitude = np.sqrt(X**2 + Y**2)
            angle = np.arctan2(-Y, X)
            
            mask = magnitude <= 1.0
            
            hsv_wheel = np.zeros((size, size, 3), dtype=np.uint8)
            hsv_wheel[mask, 0] = ((angle[mask] + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
            hsv_wheel[mask, 1] = 255
            hsv_wheel[mask, 2] = 255
            
            rgb_wheel = cv2.cvtColor(hsv_wheel, cv2.COLOR_HSV2RGB)
            return rgb_wheel
        
        color_wheel = create_color_wheel()
        
        # Save visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Flow visualization
        ax1.imshow(flow_rgb)
        ax1.set_title(f'Optical Flow Direction - Frame {frame_id}')
        ax1.axis('off')
        
        # Color wheel
        ax2.imshow(color_wheel)
        ax2.set_title('Flow Direction Color Wheel')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_folder_optical_flow}/optical_flow_frame_{frame_id:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _compute_optical_flow_pytorch3d_style(self, current_fg_points, prev_fg_points, 
                                        current_camera, prev_camera, image_size=512, frame_id=0):
        
        if current_fg_points.shape[0] > prev_fg_points.shape[0]:
            current_fg_points = current_fg_points[:prev_fg_points.shape[0]]
        elif prev_fg_points.shape[0] > current_fg_points.shape[0]:
            prev_more = prev_fg_points[-(prev_fg_points.shape[0] - current_fg_points.shape[0]):]
            current_fg_points = torch.cat([current_fg_points, prev_more], dim=0)
        
        current_uv = self._proj_uv(current_fg_points, current_camera, image_size)
        prev_uv = self._proj_uv(prev_fg_points, prev_camera, image_size)
        
        delta_uv = current_uv - prev_uv

        delta_uv_3d = torch.cat([delta_uv, torch.zeros_like(delta_uv[:, :1])], dim=-1)
        

        flow_colors = delta_uv_3d.clone()
        xy_flow = flow_colors[:, :2]

        magnitude = torch.sqrt(xy_flow[:, 0]**2 + xy_flow[:, 1]**2)
        zero_flow_mask = magnitude < 1e-4

        min_val = xy_flow.min()
        max_val = xy_flow.max()

        if max_val - min_val > 1e-4:
            flow_colors[:, :2] = 0.1 + (xy_flow - min_val) / (max_val - min_val) * 0.8
            flow_colors[zero_flow_mask, :2] = 0.0
        else:
            flow_colors[:, :2] = 0.5
        
        flow_colors = torch.clamp(flow_colors, 0, 1)

        alpha = 1.0
        flow_rgba = torch.cat([
            flow_colors,
            alpha * torch.ones_like(flow_colors[..., :1])
        ], dim=-1)

        point_cloud = Pointclouds(
            points=[prev_fg_points],
            features=[flow_rgba]
        )
        
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=0.01 if 'fg_points_render_radius' not in self.config else self.config['fg_points_render_radius'],
            points_per_pixel=50,
        )
        
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=current_camera, raster_settings=raster_settings),
            compositor=AlphaCompositor()
        )
        
        flow_image = renderer(point_cloud)
        
        flow_alpha = flow_image[0, :, :, 3]
        valid_mask = flow_alpha > self.config['alpha_threshold']

        optical_flow = torch.zeros(image_size, image_size, 3, device=self.device)

        if valid_mask.sum() > 0 and max_val - min_val > 1e-4:
            rendered_flow = flow_image[0, :, :, :2][valid_mask]
            
            zero_pixels = torch.all(rendered_flow < 0.05, dim=-1)
            normal_pixels = ~zero_pixels
            
            full_flow = torch.zeros_like(rendered_flow)
            
            if normal_pixels.sum() > 0:
                full_flow[normal_pixels] = (rendered_flow[normal_pixels] - 0.1) / 0.8 * (max_val - min_val) + min_val
            
            if zero_pixels.sum() > 0:
                full_flow[zero_pixels] = 0.0
            
            optical_flow[:, :, :2][valid_mask] = full_flow

            meaningful_mask = valid_mask.clone()
            valid_coords = torch.where(valid_mask)
            zero_coords_in_valid = zero_pixels
            meaningful_mask[valid_coords[0][zero_coords_in_valid], valid_coords[1][zero_coords_in_valid]] = False
            
            if self.config.get('debug', False):
                self.save_optical_flow(optical_flow, meaningful_mask, frame_id)

        return optical_flow


    def _proj_uv(self, xyz, camera, image_size):
        device = xyz.device
        
        K_4x4 = camera.K[0]
        intr = K_4x4[:3, :3].clone()
        
        w2c = torch.eye(4).float().to(device)
        R_w2c = camera.R[0]
        T_w2c = camera.T[0]
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = T_w2c

        intr[2, 2] = 1.0
        
        intr = intr.to(device)
        
        c_xyz = (w2c[:3, :3] @ xyz.T).T + w2c[:3, 3]
        i_xyz = (intr @ c_xyz.T).T
        uv = i_xyz[:, :2] / i_xyz[:, -1:].clip(1e-3)

        uv = image_size - uv
        
        return uv

    def obj_kp_matching(self, mask, mesh_vertices, mesh_faces, idx):

        gt_kp_h, gt_kp_w = self.kps_from_quants(mask, idx)
        if self.config.get('debug', False):
            gt_kp_save_path = (self.output_folder / "gt_kps.png").as_posix()
            save_mask_kps(mask, gt_kp_h, gt_kp_w, gt_kp_save_path)

        verts_min = mesh_vertices.min(dim=0)[0].unsqueeze(0).unsqueeze(0)
        verts_max = mesh_vertices.max(dim=0)[0].unsqueeze(0).unsqueeze(0)

        proxy_colors = ((mesh_vertices.clone() - verts_min) / (
            verts_max - verts_min
        )).squeeze(0)
        
        z_translation = torch.tensor([0, 0, 0.5], device=self.device)
        mesh_vertices += z_translation

        def render_mesh(mesh_vertices, mesh_faces, mesh_colors):
            textures = Textures(verts_rgb=mesh_colors.unsqueeze(0))
            obj_mesh = Meshes(
                verts=[mesh_vertices],
                faces=[mesh_faces],
                textures=textures
            )
            obj_raster_settings = RasterizationSettings(
                image_size=self.target_size,
                blur_radius=0.0,
                faces_per_pixel=1
            )

            obj_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=self.current_camera, raster_settings=obj_raster_settings),
                shader=HardShader(device=self.device, cameras=self.current_camera),
            )
            rendered_images = obj_renderer(obj_mesh)
            rendered_rgb = rendered_images[0, ..., :3]
            rendered_mask = rendered_images[0, ..., -1]
            rendered_mask = (rendered_mask > 0).float()
            rendered_rgb = rendered_rgb.permute(2, 0, 1).clamp(0, 1)

            return rendered_rgb, rendered_mask


        fg_render, fg_mask = render_mesh(mesh_vertices, mesh_faces, proxy_colors)
        if self.config.get('debug', False):
            torchvision_utils.save_image(fg_render, self.output_folder / "mesh_init_render_proxy_color.png")

        mesh_kps_h, mesh_kps_w = self.kps_from_quants(fg_mask, idx)
        if self.config.get('debug', False):
            mesh_kps_save_path = (self.output_folder / "mesh_kps.png").as_posix()
            save_mask_kps(fg_mask, mesh_kps_h, mesh_kps_w, mesh_kps_save_path)
        
        input_unprojected_points = rearrange(
            self.input_image_points, "(w h) c -> c h w", h=self.target_size[0], w=self.target_size[1]
        )

        gt_kps = input_unprojected_points[:, gt_kp_h, gt_kp_w].permute(1, 0)
        mesh_kps = fg_render[:, mesh_kps_h, mesh_kps_w].permute(1, 0)
        mesh_kps = mesh_kps * (verts_max[0] - verts_min[0]) + verts_min[0]

        A = mesh_kps
        B = gt_kps.flatten().unsqueeze(-1)
        A_compact = torch.cat(
            [
                A.unsqueeze(-1),
                torch.eye(3)
                .unsqueeze(0)
                .repeat(mesh_kps.shape[0], 1, 1)
                .to(device=self.device),
            ],
            dim=-1,
        )
        A_compact_final = torch.cat([i for i in A_compact], dim=0)

        solution = torch.linalg.lstsq(A_compact_final, B).solution
        scale = solution[0]
        translation = solution[1:, 0]
        mesh_vertices -= z_translation

        return scale, translation


    def kps_from_quants(self, mask, idx):

        if 'obj_kp' in self.config:
            quant = torch.tensor(self.config['obj_kp'][idx][0]).float().to(self.device)
            per_quant = torch.tensor(self.config['obj_kp'][idx][1]).float().to(self.device)
        else:
            quant = torch.tensor([0.1, 0.9]).float().to(self.device)
            per_quant = torch.tensor([0.2, 0.8]).float().to(self.device)

        mask_h, mask_w = torch.where(mask != 0)
        mask_w_min, mask_w_max = mask_w.min(), mask_w.max()
        
        quant_index = quant * (mask_w_max - mask_w_min) + mask_w_min
        quant_index = quant_index.long()

        select_hs = []
        select_ws = []
        for id in range(quant_index.shape[0]):
            valid_h = torch.where(mask[:,quant_index[id]] != 0)[0]
            valid_h, _ = torch.sort(valid_h)

            for jd in range(per_quant.shape[0]):
                select_h = valid_h[(valid_h.shape[0] * per_quant[jd]).long()]
                select_hs.append(select_h)
                select_ws.append(quant_index[id])
        
        select_hs = torch.stack(select_hs, dim=0)
        select_ws = torch.stack(select_ws, dim=0)
        return select_hs, select_ws
            
    def update_fg_obj_info(self, all_obj_points):
        for idx, per_obj_point in enumerate(all_obj_points):
            self.fg_pcs[idx]['points'] = per_obj_point.clone()
    

    def estimate_plane_normal_simple(self, vertices):
        """
        Simple version - estimate plane normal vector
        
        Parameters:
        -----------
        vertices : np.ndarray, shape (N, 3)
            vertex coordinates
        
        Returns:
        --------
        normal : np.ndarray, shape (3,)
            unit normal vector [x, y, z]
        """
        centroid = np.mean(vertices, axis=0)
        centered = vertices - centroid
        
        cov_matrix = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        normal = eigenvecs[:, 0]
        
        return normal
    

    def remap_depth(self, depth_map, remap_depth, valid_mask=None, percentile_clip=95):
        depth_map = depth_map.clone()
        
        valid_depths = depth_map[valid_mask]
        
        clip_max = torch.quantile(valid_depths, percentile_clip / 100.0)
        
        min_val = valid_depths.min()
        max_val = clip_max
        
        if max_val - min_val < 1e-8:
            return depth_map
        
        normalized = torch.zeros_like(depth_map)
        clipped_depths = torch.clamp(depth_map[valid_mask], max=clip_max)
        normalized[valid_mask] = (clipped_depths - min_val) / (max_val - min_val)
        
        remapped = normalized * (remap_depth[1] - remap_depth[0]) + remap_depth[0]
        
        remapped[~valid_mask] = torch.max(remapped[valid_mask])
        
        return remapped