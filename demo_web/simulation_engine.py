"""Interactive Genesis simulation wrapper for the demo.

Loads pre-computed 3D reconstruction results and runs physics simulation
with user-controlled forces, rendering frames and computing optical flow.
"""

import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import genesis as gs

from omegaconf import OmegaConf

from simulation.utils import pt3d_to_gs, gs_to_pt3d, pose_to_transform_matrix
from simulation.case_simulation.case_handler import get_case_handler

from pytorch3d.renderer import PerspectiveCameras
from PIL import Image


class InteractiveSimulator:
    """Wraps Genesis simulation for interactive force control."""

    def __init__(self, demo_data_path: str, device: str = "cuda",
                 config_overrides: dict | None = None):
        self.demo_data_path = Path(demo_data_path)
        self.device = torch.device(device)

        self.config = OmegaConf.to_container(
            OmegaConf.load(self.demo_data_path / "config.yaml"), resolve=True
        )
        self.config["device"] = device
        self.config["output_folder"] = str(self.demo_data_path / "sim_tmp")
        os.makedirs(self.config["output_folder"], exist_ok=True)
        self.config.setdefault("debug", False)
        if config_overrides:
            self.config.update(config_overrides)

        self.dt = self.config.get("dt", 0.01)
        self.substeps = self.config.get("substeps", 10)
        self.frame_steps = self.config.get("frame_steps", 5)
        self.material_type = self.config["material_type"]
        self.crop_start = self.config.get("crop_start", 176)

        self.object_masks_b64 = self._load_object_masks()
        self.demo_case_handler = None

        self._setup_scene()

    def _setup_scene(self):
        """Load pre-computed data and build Genesis scene."""
        meshes_dir = self.demo_data_path / "fg_meshes"
        pcs_dir = self.demo_data_path / "fg_pcs"

        mesh_files = sorted(meshes_dir.glob("mesh_*.obj"))
        pc_files = sorted(pcs_dir.glob("pc_*.pt"))

        self.fg_meshes = []
        for mf in mesh_files:
            mesh = trimesh.load(str(mf), process=False)
            self.fg_meshes.append({
                "vertices": torch.from_numpy(mesh.vertices).to(self.device).float(),
                "faces": torch.from_numpy(mesh.faces).to(self.device).long(),
                "colors": torch.from_numpy(
                    np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
                ).to(self.device).float(),
            })

        self.fg_pcs_pt3d = []
        self.fg_pcs_gs = []
        for pf in pc_files:
            data = torch.load(pf, map_location=self.device)
            self.fg_pcs_pt3d.append({
                "points": data["points"].to(self.device),
                "colors": data["colors"].to(self.device),
            })
            self.fg_pcs_gs.append({
                "points": pt3d_to_gs(data["points"].clone().to(self.device)),
                "colors": data["colors"].to(self.device),
            })

        for mesh_info in self.fg_meshes:
            mesh_info["vertices"] = pt3d_to_gs(mesh_info["vertices"])

        cam_data = torch.load(self.demo_data_path / "camera.pt", map_location=self.device)
        bg_data = torch.load(self.demo_data_path / "bg_points.pt", map_location=self.device)

        gn_path = self.demo_data_path / "ground_plane_normal.npy"
        self.ground_plane_normal = None
        if gn_path.exists():
            self.ground_plane_normal = pt3d_to_gs(np.load(gn_path))
            if self.ground_plane_normal[2] < 0:
                self.ground_plane_normal = -self.ground_plane_normal

        self._setup_renderer(cam_data, bg_data)
        self._setup_genesis()

    def _setup_renderer(self, cam_data, bg_data):
        camera = PerspectiveCameras(
            K=cam_data["K"].to(self.device),
            R=cam_data["R"].to(self.device),
            T=cam_data["T"].to(self.device),
            in_ndc=False,
            image_size=((512, 512),),
            device=self.device,
        )

        self.svr = _MinimalSVR(
            config=self.config,
            camera=camera,
            focal_length=cam_data["focal_length"],
            bg_points=bg_data["points"].to(self.device),
            bg_points_colors=bg_data["colors"].to(self.device),
            fg_pcs=[{
                "points": pc["points"].clone(),
                "colors": pc["colors"].clone(),
            } for pc in self.fg_pcs_pt3d],
            device=self.device,
        )

    def _setup_genesis(self):
        all_obj_info = []
        all_lower = torch.tensor([float("inf")] * 3, device=self.device)
        all_upper = torch.tensor([float("-inf")] * 3, device=self.device)

        for idx, mesh_info in enumerate(self.fg_meshes):
            vmin = mesh_info["vertices"].min(0).values
            vmax = mesh_info["vertices"].max(0).values
            center = mesh_info["vertices"].mean(0)
            size = vmax - vmin

            mesh_info["vertices"] -= center

            mesh_path = os.path.join(self.config["output_folder"], f"fg_mesh_{idx:02d}.obj")
            t = trimesh.Trimesh(
                vertices=mesh_info["vertices"].cpu().numpy(),
                faces=mesh_info["faces"].cpu().numpy(),
                vertex_colors=mesh_info["colors"].cpu().numpy(),
            )
            t.export(mesh_path)

            all_obj_info.append({
                "min": vmin, "max": vmax, "center": center, "size": size,
                "mesh_path": mesh_path,
                "vertices": mesh_info["vertices"] + center,
            })
            all_lower = torch.minimum(all_lower, vmin)
            all_upper = torch.maximum(all_upper, vmax)

        self.all_obj_info = all_obj_info

        self.case_handler = get_case_handler(
            self.config["example_name"], self.config, all_obj_info, self.device
        )
        self.case_handler.set_simulation_bounds(all_lower, all_upper)
        sim_lower, sim_upper = self.case_handler.get_simulation_bounds()

        gravity_dir = (
            self.ground_plane_normal.copy()
            if self.ground_plane_normal is not None
            else np.array([0, 0, 1])
        )
        if "gravity" in self.config:
            if isinstance(self.config["gravity"], (int, float)):
                gravity = tuple(self.config["gravity"] * gravity_dir)
            else:
                gravity = tuple(pt3d_to_gs(np.array(self.config["gravity"])))
        else:
            gravity = tuple(-9.8 * gravity_dir)

        pbd_gravity = None
        if "pbd_gravity" in self.config:
            if isinstance(self.config["pbd_gravity"], (int, float)):
                pbd_gravity = tuple(self.config["pbd_gravity"] * gravity_dir)
            else:
                pbd_gravity = tuple(pt3d_to_gs(np.array(self.config["pbd_gravity"])))

        mpm_gravity = None
        if "mpm_gravity" in self.config:
            if isinstance(self.config["mpm_gravity"], (int, float)):
                mpm_gravity = tuple(self.config["mpm_gravity"] * gravity_dir)
            else:
                mpm_gravity = tuple(pt3d_to_gs(np.array(self.config["mpm_gravity"])))

        gs.init(seed=self.config.get("seed", 0), precision="32",
                backend=gs.gpu, logging_level="warning")

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt, gravity=gravity, substeps=self.substeps,
            ),
            show_viewer=False,
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=False,
                ambient_light=(0.5, 0.5, 0.5),
                lights=[{
                    "type": "directional",
                    "dir": (0, 0, 1),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 2.0,
                }],
            ),
            renderer=gs.renderers.Rasterizer(),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                enable_collision=True,
                enable_self_collision=False,
                constraint_timeconst=0.02,
            ),
            pbd_options=gs.options.PBDOptions(
                lower_bound=tuple(sim_lower),
                upper_bound=tuple(sim_upper),
                particle_size=self.config.get("particle_size", 0.01),
                gravity=pbd_gravity,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=tuple(sim_lower),
                upper_bound=tuple(sim_upper),
                grid_density=self.config.get("MPM_grid_density", 64),
                particle_size=self.config.get("particle_size", 0.01),
                gravity=mpm_gravity,
            ),
            coupler_options=gs.options.LegacyCouplerOptions(
                rigid_pbd=True, rigid_mpm=True,
            ),
        )

        obj_materials = []
        obj_vis_modes = []
        for mt in self.material_type:
            mat, vis = self._get_material(mt)
            obj_materials.append(mat)
            obj_vis_modes.append(vis)

        self.objs = self.case_handler.add_entities_to_scene(
            self.scene, obj_materials, obj_vis_modes
        )

        self.case_handler.before_scene_building(
            self.scene, self.objs, self.ground_plane_normal
        )

        self.debug_cam = None
        self._debug_cam_failed = False
        if self.config.get("debug", False):
            self.debug_cam = self.scene.add_camera(
                res=(512, 512),
                pos=(0, -1, 0),
                lookat=(0, 1, 0),
                fov=self.config.get("fov_x_input", 60),
                GUI=False,
            )
            self._debug_output = Path(self.config["output_folder"])
            self._debug_gs_frames = self._debug_output / "gs_frames"
            self._debug_gs_frames.mkdir(parents=True, exist_ok=True)

        self.scene.build()
        self.case_handler.after_scene_building()

        for _ in range(3):
            self.scene.step()
        self.scene.reset()
        self.case_handler.fix_particles()

        self.initial_transform_matrix = {}
        self.closest_indices = {}
        for obj_idx, mt in enumerate(self.material_type):
            if mt == "rigid":
                self.objs[obj_idx].solver.update_vgeoms_render_T()
                rigid_T = self.objs[obj_idx].solver._vgeoms_render_T
                rigid_idx = self.objs[obj_idx].idx
                self.initial_transform_matrix[obj_idx] = (
                    torch.tensor(rigid_T[rigid_idx, 0]).to(self.device).float()
                )
            elif mt in ("pbd_liquid", "pbd_cloth", "mpm_sand", "mpm_liquid",
                        "mpm_elastic", "mpm_snow", "mpm_elastic2plastic",
                        "pbd_elastic", "pbd_particle"):
                self.closest_indices[obj_idx] = self._map_pc_to_particles(obj_idx)

        self._init_particles_gpu = {
            obj_idx: torch.tensor(
                self.objs[obj_idx].init_particles,
                device=self.device, dtype=torch.float32,
            )
            for obj_idx in self.closest_indices
        }

        self.step_count = 0
        print("Genesis scene construction finished")

    def set_demo_case_handler(self, handler):
        self.demo_case_handler = handler
        # Warmup WITH forces to trigger Taichi/CUDA JIT compilation upfront,
        # so block 0 of the actual simulation doesn't pay the compile cost.
        print("Warming up force kernels ...")
        for i in range(3):
            self.demo_case_handler.apply_forces(self, i)
            self.scene.step()
        self.scene.reset()
        self.case_handler.fix_particles()
        self.step_count = 0
        print("Force kernel warmup done")

    def _load_object_masks(self):
        masks_dir = self.demo_data_path / "fg_masks"
        if not masks_dir.exists():
            return []
        mask_files = sorted(masks_dir.glob("mask_*.png"))
        masks_b64 = []
        for mf in mask_files:
            with open(mf, "rb") as f:
                masks_b64.append(base64.b64encode(f.read()).decode("ascii"))
        return masks_b64

    def step(self, extract_points=True):
        """Run one simulation step with interactive force applied."""
        if self.demo_case_handler is not None:
            self.demo_case_handler.apply_forces(self, self.step_count)

        if self.debug_cam is not None and not self._debug_cam_failed:
            try:
                self.debug_cam.start_recording()
            except Exception:
                self._debug_cam_failed = True

        self.scene.step()

        if self.debug_cam is not None and not self._debug_cam_failed:
            try:
                render_out = self.debug_cam.render()
                cv2.imwrite(
                    str(self._debug_gs_frames / f"{self.step_count:04d}.png"),
                    render_out[0],
                )
            except Exception:
                self._debug_cam_failed = True

        self.step_count += 1

        if not extract_points:
            return None

        updated_all_obj_points = []
        for obj_idx, mt in enumerate(self.material_type):
            if mt == "rigid":
                pos = self.objs[obj_idx].get_pos().cpu().numpy()
                quat = self.objs[obj_idx].get_quat().cpu().numpy()
                T = torch.from_numpy(
                    pose_to_transform_matrix(pos, quat)
                ).to(self.device).float()
                T_inv = torch.linalg.inv(self.initial_transform_matrix[obj_idx])
                real_T = T @ T_inv
                pts_h = torch.cat([
                    self.fg_pcs_gs[obj_idx]["points"],
                    torch.ones(self.fg_pcs_gs[obj_idx]["points"].shape[0], 1, device=self.device),
                ], dim=1)
                updated = (real_T.unsqueeze(0) @ pts_h.unsqueeze(-1)).squeeze(-1)[:, :3]
                updated_all_obj_points.append(gs_to_pt3d(updated))
            else:
                p_start = self.objs[obj_idx].particle_start
                p_end = self.objs[obj_idx].particle_end
                state = self.objs[obj_idx].solver.get_state(0)
                particles_now = state.pos[0, p_start:p_end].float()

                init_particles_gpu = self._init_particles_gpu.get(obj_idx)
                if init_particles_gpu is None:
                    init_particles_gpu = torch.tensor(
                        self.objs[obj_idx].init_particles,
                        device=self.device, dtype=torch.float32,
                    )
                delta = particles_now - init_particles_gpu
                pc_delta = delta[self.closest_indices[obj_idx]].mean(dim=1)
                updated = self.fg_pcs_gs[obj_idx]["points"] + pc_delta
                updated_all_obj_points.append(gs_to_pt3d(updated))

        return updated_all_obj_points

    def render_preview(self):
        frame_pil, _, _ = self.svr.render(frame_id=0, save=False, mask=False)
        return frame_pil

    def render_and_flow(self, updated_points, frame_id=None):
        """Render the current frame and compute optical flow."""
        self.svr.update_fg_obj_info(updated_points)
        if frame_id is None:
            frame_id = self.step_count
        save_debug = self.config.get("debug", False)
        frame_pil, fg_mask, mesh_mask = self.svr.render(
            frame_id=frame_id, save=save_debug, mask=True,
        )

        if self.svr._last_optical_flow is not None:
            flow_hw3 = self.svr._last_optical_flow
            flow_2hw = flow_hw3[..., :2].transpose(2, 0, 1)
        else:
            flow_2hw = np.zeros((2, 512, 512), dtype=np.float32)

        return frame_pil, flow_2hw, fg_mask, mesh_mask

    def save_debug_outputs(self, sim_frames=None):
        if not self.config.get("debug", False):
            return

        from simulation.utils import save_gif_from_image_folder, save_video_from_pil

        output = self._debug_output
        render_dir = self.svr.output_folder

        if self.debug_cam is not None and not self._debug_cam_failed:
            try:
                self.debug_cam.stop_recording(
                    save_to_filename=str(output / "render_gs.mp4"), fps=10
                )
            except Exception as e:
                print(f"[debug] cam.stop_recording failed: {e}")

        if hasattr(self, '_debug_gs_frames') and self._debug_gs_frames.exists():
            save_gif_from_image_folder(
                str(self._debug_gs_frames), str(output / "simulated_frames_gs.gif")
            )

        svr_frames_dir = render_dir / "frames"
        if svr_frames_dir.exists():
            save_gif_from_image_folder(
                str(svr_frames_dir), str(output / "simulated_frames.gif")
            )

        svr_flow_dir = render_dir / "optical_flow"
        if svr_flow_dir.exists():
            save_gif_from_image_folder(
                str(svr_flow_dir), str(output / "flow_image.gif")
            )

        if sim_frames:
            save_video_from_pil(
                sim_frames, str(output / "simulated_frames.mp4"), fps=10
            )

    def reset(self):
        self.step_count = 0
        if self.demo_case_handler is not None:
            self.demo_case_handler.reset_forces()
        self.scene.reset()
        self.case_handler.fix_particles()
        self.svr.previous_frame_data = None
        self.svr.optical_flow = np.array([])
        self.svr._last_optical_flow = None
        self.svr.cache_bg = None
        self.svr._prev_fg_frags_idx = None
        self.svr._prev_fg_frags_dists = None

    def _map_pc_to_particles(self, obj_idx):
        sim_particles = torch.tensor(
            self.objs[obj_idx].init_particles, device=self.device
        )
        K = 256
        num_closest = self.config.get("closest_points_num", 5)
        chunks = torch.split(self.fg_pcs_gs[obj_idx]["points"], K)
        indices = []
        for chunk in chunks:
            dists = torch.norm(
                chunk.unsqueeze(1) - sim_particles.unsqueeze(0), dim=2
            )
            indices.append(
                torch.topk(dists, k=num_closest, dim=1, largest=False)[1]
            )
            del dists
        return torch.cat(indices)

    def _get_material(self, mt):
        c = self.config
        if mt == "rigid":
            return gs.materials.Rigid(
                rho=c.get("rigid_rho", 1000.0),
                friction=c.get("rigid_friction", 5.0),
                coup_friction=c.get("rigid_coup_friction", 5),
                coup_softness=c.get("rigid_coup_softness", 0.002),
            ), "visual"
        elif mt == "pbd_cloth":
            return gs.materials.PBD.Cloth(
                rho=c.get("pbd_rho", 4.0),
                static_friction=c.get("pbd_static_friction", 0.6),
                kinetic_friction=c.get("pbd_kinetic_friction", 0.35),
                stretch_compliance=c.get("pbd_stretch_compliance", 1e-7),
                bending_compliance=c.get("pbd_bending_compliance", 1e-5),
                stretch_relaxation=c.get("pbd_stretch_relaxation", 0.7),
                bending_relaxation=c.get("pbd_bending_relaxation", 0.1),
                air_resistance=c.get("pbd_air_resistance", 5e-3),
            ), "particle"
        elif mt == "pbd_elastic":
            return gs.materials.PBD.Elastic(
                rho=c.get("pbd_elastic_rho", 300.0),
                static_friction=c.get("pbd_elastic_static_friction", 0.15),
                kinetic_friction=c.get("pbd_elastic_kinetic_friction", 0.0),
                stretch_compliance=c.get("pbd_elastic_stretch_compliance", 0.0),
                bending_compliance=c.get("pbd_elastic_bending_compliance", 0.0),
                volume_compliance=c.get("pbd_elastic_volume_compliance", 0.0),
                stretch_relaxation=c.get("pbd_elastic_stretch_relaxation", 0.1),
                bending_relaxation=c.get("pbd_elastic_bending_relaxation", 0.1),
                volume_relaxation=c.get("pbd_elastic_volume_relaxation", 0.1),
            ), "particle"
        elif mt == "mpm_sand":
            return gs.materials.MPM.Sand(
                E=c.get("MPM_E", 1e6), nu=c.get("MPM_nu", 0.2),
                rho=c.get("MPM_rho", 1000.0),
                friction_angle=c.get("MPM_friction_angle", 45),
            ), "particle"
        elif mt == "mpm_elastic":
            return gs.materials.MPM.Elastic(
                E=c.get("MPM_E", 1e6), nu=c.get("MPM_nu", 0.2),
                rho=c.get("MPM_rho", 1000.0),
            ), "particle"
        elif mt == "mpm_liquid":
            return gs.materials.MPM.Liquid(
                E=c.get("MPM_E", 1e6), nu=c.get("MPM_nu", 0.2),
                rho=c.get("MPM_rho", 1000.0),
            ), "particle"
        elif mt == "mpm_snow":
            return gs.materials.MPM.Snow(
                E=c.get("MPM_E", 1e6), nu=c.get("MPM_nu", 0.2),
                rho=c.get("MPM_rho", 1000.0),
            ), "particle"
        elif mt == "pbd_liquid":
            return gs.materials.PBD.Liquid(
                rho=c.get("pbd_rho", 1000.0),
                density_relaxation=c.get("pbd_density_relaxation", 0.2),
                viscosity_relaxation=c.get("pbd_viscosity_relaxation", 0.1),
            ), "particle"
        elif mt == "pbd_particle":
            return gs.materials.PBD.Particle(), "particle"
        else:
            raise NotImplementedError(f"Material {mt} not supported")


class _MinimalSVR:
    """Minimal point-cloud renderer with optical flow computation.

    Provides render() and update_fg_obj_info() with pre-loaded data.
    _proj_uv and save_optical_flow are inlined here to avoid importing
    SingleViewReconstructor (which pulls in SAM3D, MoGe, FluxInpainter).
    """

    def __init__(self, config, camera, focal_length, bg_points,
                 bg_points_colors, fg_pcs, device):
        self.config = config
        self.current_camera = camera
        self.init_focal_length = focal_length
        self.bg_points = bg_points
        self.bg_points_colors = bg_points_colors
        self.fg_pcs = fg_pcs
        self.device = device
        self.target_size = (512, 512)

        self.previous_frame_data = None
        self.optical_flow = np.array([])
        self._last_optical_flow = None
        self._prev_fg_frags_idx = None
        self._prev_fg_frags_dists = None
        self.franka_mesh = None
        self.merge_mask = config.get("merge_mask", False)
        self.cache_bg = None
        self.fg_objects = []

        self.output_folder = Path(config.get("output_folder", "/tmp/svr_render"))
        self.output_folder_frames = self.output_folder / "frames"
        self.output_folder_masks = self.output_folder / "masks"
        self.output_folder_optical_flow = self.output_folder / "optical_flow"
        if config.get("debug", False):
            for d in [self.output_folder_frames, self.output_folder_masks,
                      self.output_folder_optical_flow]:
                d.mkdir(parents=True, exist_ok=True)

        self._build_cached_renderers()

    def _build_cached_renderers(self):
        from pytorch3d.renderer import (
            PointsRenderer, PointsRasterizer, PointsRasterizationSettings,
            AlphaCompositor,
        )
        cameras = self.current_camera
        image_size = self.target_size[0]

        fg_raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=self.config.get('fg_points_render_radius', 0.01),
            points_per_pixel=30,
            max_points_per_bin=20000,
            bin_size=0,
        )
        self._fg_rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=fg_raster_settings,
        )
        self._fg_renderer = PointsRenderer(
            rasterizer=self._fg_rasterizer,
            compositor=AlphaCompositor(),
        )

        flow_raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=self.config.get('fg_points_render_radius', 0.01),
            points_per_pixel=30,
            max_points_per_bin=20000,
            bin_size=0,
        )
        self._flow_rasterizer = PointsRasterizer(
            cameras=cameras, raster_settings=flow_raster_settings,
        )
        self._flow_renderer = PointsRenderer(
            rasterizer=self._flow_rasterizer,
            compositor=AlphaCompositor(),
        )

    def update_fg_obj_info(self, all_obj_points):
        for idx, pts in enumerate(all_obj_points):
            self.fg_pcs[idx]["points"] = pts.clone()

    def _proj_uv(self, xyz, camera, image_size):
        """Project 3D points to 2D UV coordinates."""
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

    def save_optical_flow(self, optical_flow, valid_mask, frame_id):
        """Save optical flow visualization to disk (debug mode only)."""
        flow_x = optical_flow[:, :, 0].cpu().numpy()
        flow_y = optical_flow[:, :, 1].cpu().numpy()
        valid_mask_np = valid_mask.cpu().numpy()

        angle = np.arctan2(-flow_y, flow_x)
        hsv = np.zeros((optical_flow.shape[0], optical_flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 179
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        hsv[~valid_mask_np] = 0
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(flow_rgb)
        ax1.set_title(f'Optical Flow Direction - Frame {frame_id}')
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(
            f'{self.output_folder_optical_flow}/optical_flow_frame_{frame_id:04d}.png',
            dpi=150, bbox_inches='tight',
        )
        plt.close()

    def render(self, render_bg=True, render_obj=True, render_mesh=True,
               frame_id=0, save=False, mask=True, compute_optical_flow=True):
        from pytorch3d.structures import Pointclouds
        from torchvision.transforms import ToPILImage

        cameras = self.current_camera
        image_size = self.target_size[0]

        # Background (cached after first render)
        if render_bg and self.cache_bg is None:
            from pytorch3d.renderer import (
                PointsRenderer, PointsRasterizer, PointsRasterizationSettings,
                AlphaCompositor,
            )
            bg_pc = Pointclouds(
                points=[self.bg_points], features=[self.bg_points_colors],
            )
            bg_raster_settings = PointsRasterizationSettings(
                image_size=image_size,
                radius=self.config.get('bg_points_render_radius', 0.0001),
                points_per_pixel=30,
            )
            bg_renderer = PointsRenderer(
                rasterizer=PointsRasterizer(
                    cameras=cameras, raster_settings=bg_raster_settings,
                ),
                compositor=AlphaCompositor(),
            )
            self.cache_bg = bg_renderer(bg_pc)

        if render_bg and self.cache_bg is not None:
            bg_image = self.cache_bg
        else:
            bg_image = torch.zeros(1, image_size, image_size, 3, device=self.device)

        base_rgb = bg_image[0].clone()
        final_rgb = base_rgb.clone()

        # Foreground
        all_fg_points = []
        all_fg_colors = []
        for pc_info in self.fg_pcs:
            all_fg_points.append(pc_info['points'])
            all_fg_colors.append(pc_info['colors'])

        combined_fg_points = torch.cat(all_fg_points, dim=0)
        combined_fg_colors = torch.cat(all_fg_colors, dim=0)
        flow_rendered_points = combined_fg_points.clone()

        combined_rgba = torch.cat([
            combined_fg_colors,
            torch.ones_like(combined_fg_colors[..., :1]),
        ], dim=-1)

        fg_pc = Pointclouds(points=[combined_fg_points], features=[combined_rgba])

        fragments = self._fg_rasterizer(fg_pc)
        r = self._fg_rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        fg_image = self._fg_renderer.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            fg_pc.features_packed().permute(1, 0),
        )
        fg_image = fg_image.permute(0, 2, 3, 1)
        fg_rgb = fg_image[0, ..., :3]
        fg_alpha = fg_image[0, ..., 3:4]
        fg_depth = fragments.zbuf[0, ..., 0]

        fg_points_mask = torch.where(
            fg_alpha.squeeze(-1) > self.config['alpha_threshold'], 1.0, 0.0,
        ).unsqueeze(-1)
        fg_mask_2d = fg_points_mask.squeeze(-1)
        final_rgb = fg_rgb * fg_mask_2d.unsqueeze(-1) + final_rgb * (1.0 - fg_mask_2d.unsqueeze(-1))

        # Mesh
        mesh_mask = torch.zeros(image_size, image_size, 1, dtype=torch.float32, device=self.device)

        if render_mesh and self.franka_mesh is not None:
            from pytorch3d.renderer import (
                MeshRenderer, MeshRasterizer, SoftPhongShader,
                RasterizationSettings, BlendParams,
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
                image_size=image_size, blur_radius=0.0, faces_per_pixel=10,
            )
            mesh_rasterizer = MeshRasterizer(cameras=cameras, raster_settings=mesh_raster_settings)
            mesh_renderer = MeshRenderer(
                rasterizer=mesh_rasterizer,
                shader=SoftPhongShader(
                    device=self.device, cameras=cameras,
                    blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
                ),
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

            fg_points_mask = torch.where(
                mesh_closer_bool.unsqueeze(-1),
                torch.zeros_like(fg_points_mask), fg_points_mask,
            )

        # Optical flow
        if compute_optical_flow and self.previous_frame_data is not None:
            optical_flow = self._compute_optical_flow_pytorch3d_style(
                current_fg_points=flow_rendered_points,
                prev_fg_points=self.previous_frame_data['flow_rendered_points'],
                current_camera=cameras,
                prev_camera=self.previous_frame_data['camera'],
                image_size=image_size,
                frame_id=frame_id,
                prev_frags_idx=self._prev_fg_frags_idx,
                prev_frags_dists=self._prev_fg_frags_dists,
            )
            flow_np = optical_flow.cpu().numpy()
            self._last_optical_flow = flow_np
            if self.config.get('debug', False):
                if self.optical_flow.size == 0:
                    self.optical_flow = np.expand_dims(flow_np, 0)
                else:
                    self.optical_flow = np.concatenate([
                        self.optical_flow, np.expand_dims(flow_np, 0),
                    ])

        if self.franka_mesh is None:
            self._prev_fg_frags_idx = fragments.idx
            self._prev_fg_frags_dists = fragments.dists
        else:
            self._prev_fg_frags_idx = None
            self._prev_fg_frags_dists = None

        if save:
            if mask:
                points_mask_path = self.output_folder_masks / f"points_mask_{frame_id:04d}.png"
                points_mask_to_save = fg_points_mask.squeeze(2) if fg_points_mask.dim() == 3 else fg_points_mask
                ToPILImage()(points_mask_to_save.unsqueeze(0).clamp(0, 1).cpu()).save(points_mask_path.as_posix())

                mesh_mask_path = self.output_folder_masks / f"mesh_mask_{frame_id:04d}.png"
                mesh_mask_to_save = mesh_mask.squeeze(2) if mesh_mask.dim() == 3 else mesh_mask
                ToPILImage()(mesh_mask_to_save.unsqueeze(0).clamp(0, 1).cpu()).save(mesh_mask_path.as_posix())

            image_pil = ToPILImage()(final_rgb.permute(2, 0, 1).clamp(0, 1).cpu())
            image_path = self.output_folder_frames / f"frame_{frame_id:04d}.png"
            image_pil.save(image_path.as_posix())
        else:
            image_pil = ToPILImage()(final_rgb.permute(2, 0, 1).clamp(0, 1).cpu())

        self.previous_frame_data = {
            'camera': cameras,
            'bg_points': self.bg_points,
            'flow_rendered_points': flow_rendered_points,
        }

        return image_pil, fg_points_mask, mesh_mask

    def _compute_optical_flow_pytorch3d_style(self, current_fg_points, prev_fg_points,
                                              current_camera, prev_camera,
                                              image_size=512, frame_id=0,
                                              prev_frags_idx=None,
                                              prev_frags_dists=None):
        from pytorch3d.structures import Pointclouds

        if current_fg_points.shape[0] > prev_fg_points.shape[0]:
            current_fg_points = current_fg_points[:prev_fg_points.shape[0]]
        elif prev_fg_points.shape[0] > current_fg_points.shape[0]:
            prev_more = prev_fg_points[-(prev_fg_points.shape[0] - current_fg_points.shape[0]):]
            current_fg_points = torch.cat([current_fg_points, prev_more], dim=0)

        current_uv = self._proj_uv(current_fg_points, current_camera, image_size)
        prev_uv = self._proj_uv(prev_fg_points, prev_camera, image_size)
        delta_uv = current_uv - prev_uv

        flow_colors = torch.cat([delta_uv, torch.zeros_like(delta_uv[:, :1])], dim=-1)
        xy_flow = flow_colors[:, :2]

        magnitude = torch.sqrt(xy_flow[:, 0] ** 2 + xy_flow[:, 1] ** 2)
        zero_flow_mask = magnitude < 1e-4

        min_val = xy_flow.min()
        max_val = xy_flow.max()

        if max_val - min_val > 1e-4:
            flow_colors[:, :2] = 0.1 + (xy_flow - min_val) / (max_val - min_val) * 0.8
            flow_colors[zero_flow_mask, :2] = 0.0
        else:
            flow_colors[:, :2] = 0.5

        flow_colors = torch.clamp(flow_colors, 0, 1)
        flow_rgba = torch.cat([flow_colors, torch.ones_like(flow_colors[..., :1])], dim=-1)

        if prev_frags_idx is not None and prev_frags_dists is not None:
            r = self._fg_rasterizer.raster_settings.radius
            dists2 = prev_frags_dists.permute(0, 3, 1, 2)
            prev_weights = 1 - dists2 / (r * r)
            flow_image_raw = self._fg_renderer.compositor(
                prev_frags_idx.long().permute(0, 3, 1, 2),
                prev_weights,
                flow_rgba.permute(1, 0),
            )
            flow_image = flow_image_raw.permute(0, 2, 3, 1)
        else:
            point_cloud = Pointclouds(points=[prev_fg_points], features=[flow_rgba])
            flow_image = self._flow_renderer(point_cloud)

        flow_alpha = flow_image[0, :, :, 3]
        valid_mask = flow_alpha > self.config['alpha_threshold']

        optical_flow = torch.zeros(image_size, image_size, 3, device=self.device)

        if valid_mask.sum() > 0 and max_val - min_val > 1e-4:
            rendered_flow = flow_image[0, :, :, :2][valid_mask]
            zero_pixels = torch.all(rendered_flow < 0.05, dim=-1)
            normal_pixels = ~zero_pixels

            full_flow = torch.zeros_like(rendered_flow)
            if normal_pixels.sum() > 0:
                full_flow[normal_pixels] = (
                    (rendered_flow[normal_pixels] - 0.1) / 0.8 * (max_val - min_val) + min_val
                )

            optical_flow[:, :, :2][valid_mask] = full_flow

            if self.config.get('debug', False):
                meaningful_mask = valid_mask.clone()
                valid_coords = torch.where(valid_mask)
                meaningful_mask[valid_coords[0][zero_pixels], valid_coords[1][zero_pixels]] = False
                self.save_optical_flow(optical_flow, meaningful_mask, frame_id)

        return optical_flow

    @property
    def num_fg_objects(self):
        return len(self.fg_pcs)
