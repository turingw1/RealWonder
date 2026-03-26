import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import trimesh
import cv2
import os
import genesis as gs
from pathlib import Path
from simulation.image23D.single_view_reconstructor import SingleViewReconstructor
from simulation.utils import (
    pt3d_to_gs,
    gs_to_pt3d,
    save_gif_from_image_folder,
    save_video_from_pil,
    pose_to_transform_matrix,
)
from simulation.case_simulation.case_handler import get_case_handler
import time
from simulation.utils import save_gif_from_image_folder

class DiffSim(nn.Module):
    def __init__(self, config, exp_logger=None): 
        super().__init__()
        self.config = config
        self.exp_logger = exp_logger
        self.device = self.config['device']
        self.output_folder = Path(self.config['output_folder']) / 'simulation'

        self.genesis_frames = self.output_folder / "gs_frames"
        self.genesis_frames.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.dt = self.config["dt"]
        self.substeps = self.config["substeps"]
        self.simulated_frames_num = self.config["simulated_frames_num"]
        self.frame_steps = self.config["frame_steps"]
        self.simulation_steps = self.simulated_frames_num * self.frame_steps
        self.material_type = self.config['material_type']

        self.svr = SingleViewReconstructor(config, exp_logger=self.exp_logger)
        reconstruct_start = time.perf_counter()
        self.fg_pcs_from_3d, self.fg_meshes, self.ground_plane_normal, self.config = self.svr.reconstruct()
        if self.exp_logger is not None:
            self.exp_logger.log_event(
                "diffsim.reconstruct_total",
                time.perf_counter() - reconstruct_start,
                object_count=len(self.fg_meshes),
            )

        # initialize the proxy primitived for foreground object
        if self.ground_plane_normal is not None:
            self.ground_plane_normal = pt3d_to_gs(self.ground_plane_normal)
            if self.ground_plane_normal[2] < 0:
                self.ground_plane_normal = -self.ground_plane_normal
        self.fg_pcs = []
        for idx, per_obj_pc in enumerate(self.fg_pcs_from_3d):
            self.fg_pcs.append({
                'points': pt3d_to_gs(per_obj_pc['points'].clone()),
                'colors': pt3d_to_gs(per_obj_pc['colors'].clone()),
            })
        
        # pytorch to genesis coordinates
        for idx, per_obj_mesh in enumerate(self.fg_meshes):
            per_obj_mesh['vertices'] = pt3d_to_gs(per_obj_mesh['vertices'])

        gs.init(
            seed=self.config['seed'],
            precision="32",
            backend=gs.gpu,
            logging_level="warning"
        )

        # get the global bounding box for all foreground objects
        self.all_obj_info = []
        self.all_obj_occupied_lower_bound = torch.tensor([float('inf'), float('inf'), float('inf')]).to(self.device)
        self.all_obj_occupied_upper_bound = torch.tensor([float('-inf'), float('-inf'), float('-inf')]).to(self.device)


        for idx, per_mesh_bounds in enumerate(self.fg_meshes):
            per_mesh_min = self.fg_meshes[idx]['vertices'].min(0).values
            per_mesh_max = self.fg_meshes[idx]['vertices'].max(0).values
            per_mesh_center = self.fg_meshes[idx]['vertices'].mean(0)
            per_mesh_size = per_mesh_max - per_mesh_min

            self.fg_meshes[idx]['vertices'] -= per_mesh_center
            # self.fg_pcs[idx]['points'] -= per_mesh_center
            per_obj_mesh_path = os.path.join(self.config['output_folder'], f'fg_mesh_{idx:02d}.obj')
            
            per_trimesh = trimesh.Trimesh(
                vertices=self.fg_meshes[idx]['vertices'].cpu().numpy(), 
                faces=self.fg_meshes[idx]['faces'].cpu().numpy(), 
                vertex_colors=self.fg_meshes[idx]['colors'].cpu().numpy()
            )
            
            per_trimesh.export(per_obj_mesh_path)

            self.all_obj_info.append({
                'min': per_mesh_min,
                'max': per_mesh_max,
                'center': per_mesh_center,
                'size': per_mesh_size,
                'mesh_path': per_obj_mesh_path,
                'vertices': self.fg_meshes[idx]['vertices'] + per_mesh_center,
            })
            

            self.all_obj_occupied_lower_bound = torch.minimum(self.all_obj_occupied_lower_bound, per_mesh_min)
            self.all_obj_occupied_upper_bound = torch.maximum(self.all_obj_occupied_upper_bound, per_mesh_max)

        self.case_handler = get_case_handler(self.config['example_name'], self.config, self.all_obj_info, self.device)
        self.case_handler.set_simulation_bounds(self.all_obj_occupied_lower_bound, self.all_obj_occupied_upper_bound)
        self.simulation_lower_bound, self.simulation_upper_bound = self.case_handler.get_simulation_bounds()

        if self.ground_plane_normal is not None:
            gravity_dir = self.ground_plane_normal.copy()
        else:
            gravity_dir = np.array([0, 0, 1])

        if 'mpm_gravity' in self.config:
            if isinstance(self.config['mpm_gravity'], (int, float)):
                mpm_gravity = tuple(self.config['mpm_gravity'] * np.array(gravity_dir))
            else:
                mpm_gravity = tuple(pt3d_to_gs(np.array(self.config['mpm_gravity'])))
        else:
            mpm_gravity = None

        if 'pbd_gravity' in self.config:
            if isinstance(self.config['pbd_gravity'], (int, float)):
                pbd_gravity = tuple(self.config['pbd_gravity'] * np.array(gravity_dir))
            else:
                pbd_gravity = tuple(pt3d_to_gs(np.array(self.config['pbd_gravity'])))
        else:
            pbd_gravity = None

        if 'gravity' in self.config:
            if isinstance(self.config['gravity'], (int, float)):
                gravity = tuple(self.config['gravity'] * np.array(gravity_dir))
            else:
                gravity = tuple(pt3d_to_gs(np.array(self.config['gravity'])))
        else:
            gravity = tuple(-9.8 * np.array(gravity_dir))
        
        # initialize the genesis scene
        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(
                dt=self.dt,
                gravity=gravity,
                substeps=self.substeps,
            ),
            show_viewer=False,
            vis_options = gs.options.VisOptions(
                show_world_frame = False,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = False,
                plane_reflection = False,
                ambient_light    = (0.5, 0.5, 0.5),
                lights = [{
                    'type': 'directional',
                    'dir': (0, 0, 1),
                    'color': (1.0, 1.0, 1.0),
                    'intensity': 2.0
                }]
            ),
            renderer = gs.renderers.Rasterizer(),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                enable_collision=True,
                enable_self_collision=False,
                constraint_timeconst = 0.02,
            ),
            pbd_options = gs.options.PBDOptions(
                lower_bound = tuple(self.simulation_lower_bound),
                upper_bound = tuple(self.simulation_upper_bound),
                particle_size = 0.01 if 'particle_size' not in self.config else self.config['particle_size'],
                gravity = pbd_gravity,
            ),
            mpm_options = gs.options.MPMOptions(
                lower_bound = tuple(self.simulation_lower_bound),
                upper_bound = tuple(self.simulation_upper_bound),
                grid_density = 64 if 'MPM_grid_density' not in self.config else self.config['MPM_grid_density'],
                particle_size = 0.01 if 'particle_size' not in self.config else self.config['particle_size'],
                gravity = mpm_gravity,
            ),
            coupler_options=gs.options.LegacyCouplerOptions(
                rigid_pbd=True,
                rigid_mpm=True,
            )
        )

        # get materials for each object
        self.obj_materials= []
        self.obj_vis_modes = []
        for idx, per_material_type in enumerate(self.material_type):
            obj_material, obj_vis_mode = self.get_material_for_each(per_material_type)
            self.obj_materials.append(obj_material)
            self.obj_vis_modes.append(obj_vis_mode)

        self.objs = self.case_handler.add_entities_to_scene(self.scene, self.obj_materials, self.obj_vis_modes)
        
        self.case_handler.before_scene_building(self.scene, self.objs, self.ground_plane_normal)

        self.cam = self.scene.add_camera(
            res = (512, 512),
            pos = (0, -1, 0),
            lookat = (0, 1, 0),
            fov = self.config['fov_x_input'],
            GUI = False,
        )
        build_start = time.perf_counter()
        self.scene.build()
        if self.exp_logger is not None:
            self.exp_logger.log_event("diffsim.scene_build", time.perf_counter() - build_start)

        self.case_handler.after_scene_building()

        # transform and binding
        # self.original_transform_matrix = {}
        self.closest_indices = {}
        self.initial_transform_matrix = {}

        for obj_idx, per_material_type in enumerate(self.material_type):
            if per_material_type == 'rigid':
                # for debug purpose
                # per_object_pos = self.objs[obj_idx].get_pos().cpu().numpy()
                # per_object_quat = self.objs[obj_idx].get_quat().cpu().numpy()
                # per_object_transform_matrix = torch.from_numpy(pose_to_transform_matrix(per_object_pos, per_object_quat)).to(self.device).float()
                # self.initial_transform_matrix[obj_idx] = per_object_transform_matrix
                self.objs[obj_idx].solver.update_vgeoms_render_T()
                rigid_T = self.objs[obj_idx].solver._vgeoms_render_T
                rigid_idx = self.objs[obj_idx].idx
                transform_matrix = torch.tensor(rigid_T[rigid_idx, 0]).to(self.device).float()
                self.initial_transform_matrix[obj_idx] = transform_matrix

            elif per_material_type in ['pbd_liquid', 'pbd_cloth', 'mpm_sand', 'mpm_liquid', 'mpm_elastic', 'mpm_snow', 'mpm_elastic2plastic', 'pbd_elastic', 'pbd_particle']:
                self.closest_indices[obj_idx] = self.map_pc_to_particles(obj_idx)
            else:
                raise NotImplementedError("The current material is not supported for now")

        print("genesis scene construction finished")
        
    def simulate_step(self, sid, output_folder):

        if self.config.get('debug', False):
            self.cam.start_recording()
        self.case_handler.custom_simulation(sid)
        self.scene.step()
        if self.config.get('debug', False):
            render_out = self.cam.render()
        updated_all_obj_points = []

        for obj_idx, per_material_type in enumerate(self.material_type):

            if per_material_type == 'rigid':
                obj_inertial_pos = self.objs[obj_idx].get_pos().cpu().numpy()
                obj_inertial_quat = self.objs[obj_idx].get_quat().cpu().numpy()
                transform_matrix = torch.from_numpy(pose_to_transform_matrix(obj_inertial_pos, obj_inertial_quat)).to(self.device).float()
                # Inverse the initial transform matrix
                initial_transform_matrix_inv = torch.linalg.inv(self.initial_transform_matrix[obj_idx])
                real_transform_matrix = transform_matrix @ initial_transform_matrix_inv
                points_homo = torch.cat([self.fg_pcs[obj_idx]['points'], torch.ones(self.fg_pcs[obj_idx]['points'].shape[0], 1).to(self.device)], dim=1)
                updated_points = torch.matmul(real_transform_matrix.unsqueeze(0), points_homo.unsqueeze(-1)).squeeze(-1)[:, :3]
                updated_points = gs_to_pt3d(updated_points)
                updated_all_obj_points.append(updated_points)

                # self.objs[obj_idx].solver.update_vgeoms_render_T() # trigger update
                # rigid_T = self.objs[obj_idx].solver._vgeoms_render_T
                # rigid_idx = self.objs[obj_idx].idx
                # transform_matrix = torch.tensor(rigid_T[rigid_idx, 0]).to(self.device).float() # 0 for env index
                # real_transform_matrix = transform_matrix @ torch.linalg.inv(self.initial_transform_matrix[obj_idx])
                # points_homo = torch.cat([self.fg_pcs[obj_idx]['points'], torch.ones(self.fg_pcs[obj_idx]['points'].shape[0], 1).to(self.device)], dim=1)
                # updated_points = torch.matmul(real_transform_matrix.unsqueeze(0), points_homo.unsqueeze(-1)).squeeze(-1)[:, :3]
                # updated_points = gs_to_pt3d(updated_points)
                # updated_all_obj_points.append(updated_points)

            elif per_material_type in ['pbd_liquid', 'pbd_cloth', 'mpm_sand', 'mpm_liquid', 'mpm_elastic', 'mpm_snow', 'mpm_elastic2plastic', 'pbd_elastic', 'pbd_particle']:
                particles_now_pos_in_gs = self.objs[obj_idx].solver.particles.pos.to_numpy()
                if len(particles_now_pos_in_gs.shape) == 4:
                    particles_now_pos_in_gs = particles_now_pos_in_gs[0, self.objs[obj_idx].particle_start:self.objs[obj_idx].particle_end, 0]
                else:
                    particles_now_pos_in_gs = particles_now_pos_in_gs[self.objs[obj_idx].particle_start:self.objs[obj_idx].particle_end, 0]
                
                particles_start_pos_in_gs = self.objs[obj_idx].init_particles

                particles_now_pos_in_gs = torch.tensor(particles_now_pos_in_gs).to(self.device)
                particles_start_pos_in_gs = torch.tensor(particles_start_pos_in_gs).to(self.device)

                particles_change_pos_in_gs = particles_now_pos_in_gs - particles_start_pos_in_gs
                points_change_pos_in_gs = particles_change_pos_in_gs[self.closest_indices[obj_idx]]
                points_change_pos_in_gs = points_change_pos_in_gs.mean(dim=1)
                updated_points = self.fg_pcs[obj_idx]['points'] + points_change_pos_in_gs
                updated_points = gs_to_pt3d(updated_points)
                updated_all_obj_points.append(updated_points)
        
        self.case_handler.after_simulation_step(self.svr)

        # if "robot_arm" in self.config['example_name']:
        #     franka_verts, franka_faces, franka_vertex_colors = self.extract_franka_mesh_data_combined(self.case_handler.current_target_franka)
        #     franka_verts = gs_to_pt3d(franka_verts)
        #     self.svr.franka_mesh = {
        #         'vertices': franka_verts,
        #         'faces': franka_faces,
        #         'colors': franka_vertex_colors
        #     }

        if self.config.get('debug', False):
            cv2.imwrite((output_folder / "gs_frames" / f"{sid:04d}.png").as_posix(), render_out[0])

        if self.config.get('debug', False) and sid == self.simulation_steps - 1:
            self.cam.stop_recording(save_to_filename=(output_folder / "render_gs.mp4").as_posix(), fps=10)
            # self.cam.stop_recording()
        
        return updated_all_obj_points

    def simulation_pc_render(self):
        self.simulated_frames = []
        self.simualted_masks = []
        self.simualted_mesh_masks = []
        start_time = time.time()
        step_total = 0.0
        render_total = 0.0
        for sid in tqdm(range(self.simulation_steps)):
            step_start = time.perf_counter()
            all_obj_points = self.simulate_step(sid, self.output_folder)
            step_total += time.perf_counter() - step_start
            if sid % self.frame_steps == 0:
                render_start = time.perf_counter()
                self.svr.update_fg_obj_info(all_obj_points)
                frame_id = sid // self.frame_steps
                current_frame, current_points_mask, current_mesh_mask = self.svr.render(frame_id = frame_id, save = self.config.get('debug', False), mask = True)
                self.simulated_frames.append(current_frame)
                self.simualted_masks.append(current_points_mask)
                self.simualted_mesh_masks.append(current_mesh_mask)
                render_total += time.perf_counter() - render_start
        end_time = time.time()
        print(f"Simulation + rendering time: {end_time - start_time} seconds")
        if self.exp_logger is not None:
            self.exp_logger.log_event(
                "diffsim.simulation_pc_render",
                end_time - start_time,
                simulation_steps=self.simulation_steps,
                rendered_frames=len(self.simulated_frames),
                simulate_step_total_sec=step_total,
                render_total_sec=render_total,
                avg_simulate_step_sec=step_total / max(self.simulation_steps, 1),
                avg_rendered_frame_sec=render_total / max(len(self.simulated_frames), 1),
            )
        # save the gif of the simualated frames
        if self.config.get('debug', False):
            save_gif_from_image_folder(self.output_folder / "render" / "frames", self.output_folder / "simulated_frames.gif")
            save_gif_from_image_folder(self.output_folder / "gs_frames", self.output_folder / "simulated_frames_gs.gif")
            save_video_from_pil(self.simulated_frames, self.output_folder / "simulated_frames.mp4", fps=10)
            save_gif_from_image_folder(self.output_folder / "render" / "flow_image", self.output_folder / "flow_image.gif")
        return self.simulated_frames, self.simualted_masks, self.simualted_mesh_masks

    def map_pc_to_particles(self, obj_idx):
        sim_particles = torch.tensor(self.objs[obj_idx].init_particles).to(self.device)
        print(f"number of sim_particles: {sim_particles.shape[0]}")
        K = 256
        num_closest = 5 if 'closest_points_num' not in self.config else self.config['closest_points_num']
        point_chunks = torch.split(self.fg_pcs[obj_idx]['points'], K)
        closest_indices = []

        for chunk in tqdm(point_chunks):
            # Calculate pairwise distances between chunk and all particles
            # Using broadcasting to avoid memory issues
            # Shape: [K, 1, 3] - [1, N, 3] -> [K, N, 3] -> [K, N]
            distances = torch.norm(
                chunk.unsqueeze(1) - sim_particles.unsqueeze(0),
                dim=2
            )
            # Get top num_closest indices of closest particles for this chunk
            chunk_closest = torch.topk(distances, k=num_closest, dim=1, largest=False)[1]
            del distances
            closest_indices.append(chunk_closest)

        closest_indices = torch.cat(closest_indices)
        return closest_indices

    def get_material_for_each(self, per_material_type):
        if per_material_type == "rigid":
            obj_material = gs.materials.Rigid(
                rho = 1000.0 if 'rigid_rho' not in self.config else self.config['rigid_rho'],
                friction = 5.0 if 'rigid_friction' not in self.config else self.config['rigid_friction'],
                coup_friction = 5 if 'rigid_coup_friction' not in self.config else self.config['rigid_coup_friction'],
                coup_softness = 0.002 if 'rigid_coup_softness' not in self.config else self.config['rigid_coup_softness'],
            )
            obj_vis_mode = "visual"
        elif per_material_type == 'pbd_liquid':
            obj_material = gs.materials.PBD.Liquid(
                rho = 1000.0 if 'pbd_rho' not in self.config else self.config['pbd_rho'],
                density_relaxation = 0.2 if 'pbd_density_relaxation' not in self.config else self.config['pbd_density_relaxation'],
                viscosity_relaxation = 0.1 if 'pbd_viscosity_relaxation' not in self.config else self.config['pbd_viscosity_relaxation'],
            )
            obj_vis_mode = "particle"

        elif per_material_type == "pbd_cloth":
            obj_material = gs.materials.PBD.Cloth(
                rho=4.0 if 'pbd_rho' not in self.config else self.config['pbd_rho'],
                static_friction=0.6 if 'pbd_static_friction' not in self.config else self.config['pbd_static_friction'],
                kinetic_friction=0.35 if 'pbd_kinetic_friction' not in self.config else self.config['pbd_kinetic_friction'],
                stretch_compliance=1e-7 if 'pbd_stretch_compliance' not in self.config else self.config['pbd_stretch_compliance'],
                bending_compliance=1e-5 if 'pbd_bending_compliance' not in self.config else self.config['pbd_bending_compliance'],
                stretch_relaxation=0.7 if 'pbd_stretch_relaxation' not in self.config else self.config['pbd_stretch_relaxation'],
                bending_relaxation=0.1 if 'pbd_bending_relaxation' not in self.config else self.config['pbd_bending_relaxation'],
                air_resistance=5e-3 if 'pbd_air_resistance' not in self.config else self.config['pbd_air_resistance'],

            )
            obj_vis_mode = "particle"
        elif per_material_type == "pbd_elastic":
            obj_material = gs.materials.PBD.Elastic(
                rho=300.0 if 'pbd_elastic_rho' not in self.config else self.config['pbd_elastic_rho'],
                static_friction=0.15 if 'pbd_elastic_static_friction' not in self.config else self.config['pbd_elastic_static_friction'],
                kinetic_friction=0.0 if 'pbd_elastic_kinetic_friction' not in self.config else self.config['pbd_elastic_kinetic_friction'],
                stretch_compliance=0.0 if 'pbd_elastic_stretch_compliance' not in self.config else self.config['pbd_elastic_stretch_compliance'],
                bending_compliance=0.0 if 'pbd_elastic_bending_compliance' not in self.config else self.config['pbd_elastic_bending_compliance'],
                volume_compliance=0.0 if 'pbd_elastic_volume_compliance' not in self.config else self.config['pbd_elastic_volume_compliance'],
                stretch_relaxation=0.1 if 'pbd_elastic_stretch_relaxation' not in self.config else self.config['pbd_elastic_stretch_relaxation'],
                bending_relaxation=0.1 if 'pbd_elastic_bending_relaxation' not in self.config else self.config['pbd_elastic_bending_relaxation'],
                volume_relaxation=0.1 if 'pbd_elastic_volume_relaxation' not in self.config else self.config['pbd_elastic_volume_relaxation'],
            )
            obj_vis_mode = "particle"
        elif per_material_type == "pbd_particle":
            obj_material = gs.materials.PBD.Particle()
            obj_vis_mode = "particle"
        elif per_material_type == "mpm_sand":
            obj_material = gs.materials.MPM.Sand(
                E = 1e6 if 'MPM_E' not in self.config else self.config['MPM_E'],
                nu = 0.2 if 'MPM_nu' not in self.config else self.config['MPM_nu'],
                rho = 1000.0 if 'MPM_rho' not in self.config else self.config['MPM_rho'],
                friction_angle = 45 if 'MPM_friction_angle' not in self.config else self.config['MPM_friction_angle'],
            )
            obj_vis_mode = "particle"
        elif per_material_type == "mpm_elastic":
            obj_material = gs.materials.MPM.Elastic(
                E = 1e6 if 'MPM_E' not in self.config else self.config['MPM_E'],
                nu = 0.2 if 'MPM_nu' not in self.config else self.config['MPM_nu'],
                rho = 1000.0 if 'MPM_rho' not in self.config else self.config['MPM_rho'],
            )
            obj_vis_mode = "particle"
        elif per_material_type == "mpm_liquid":
            obj_material = gs.materials.MPM.Liquid(
                E = 1e6 if 'MPM_E' not in self.config else self.config['MPM_E'],
                nu = 0.2 if 'MPM_nu' not in self.config else self.config['MPM_nu'],
                rho = 1000.0 if 'MPM_rho' not in self.config else self.config['MPM_rho'],
            )
            obj_vis_mode = "particle"
        elif per_material_type == "mpm_snow":
            obj_material = gs.materials.MPM.Snow(
                E = 1e6 if 'MPM_E' not in self.config else self.config['MPM_E'],
                nu = 0.2 if 'MPM_nu' not in self.config else self.config['MPM_nu'],
                rho = 1000.0 if 'MPM_rho' not in self.config else self.config['MPM_rho'],
            )
            obj_vis_mode = "particle"
        elif per_material_type == "mpm_elastic2plastic":
            obj_material = gs.materials.MPM.ElastoPlastic(
                E = 1e6 if 'MPM_E' not in self.config else self.config['MPM_E'],
                nu = 0.2 if 'MPM_nu' not in self.config else self.config['MPM_nu'],
                rho = 1000.0 if 'MPM_rho' not in self.config else self.config['MPM_rho'],
            )
            obj_vis_mode = "particle"
        else:
            raise NotImplementedError(f"The current material {per_material_type} is not supported for now")
        return obj_material, obj_vis_mode

    def extract_franka_mesh_data_combined(self, target_franka):
        """
        Extract and combine all mesh data into single arrays with transformations applied.
        
        Returns:
            vertices: torch tensor of all transformed vertices
            faces: torch tensor of all faces (with proper indexing)
            colors: torch tensor of per-vertex colors
        """
        
        all_vertices = []
        all_faces = []
        all_colors = []
        
        vertex_offset = 0
        sim_vgeoms_render_T = target_franka.solver._vgeoms_render_T
        
        for vgeom in target_franka.vgeoms:
            verts = vgeom.vmesh.verts
            faces = vgeom.vmesh.faces
            
            # Get transformation matrix for this vgeom
            cur_render_T = sim_vgeoms_render_T[vgeom.idx][0]
            
            # Apply transformation to vertices
            # Convert vertices to homogeneous coordinates (N, 4)
            verts_homogeneous = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
            
            # Apply transformation: (N, 4) @ (4, 4)^T = (N, 4)
            verts_transformed = verts_homogeneous @ cur_render_T.T
            
            # Convert back to 3D coordinates (N, 3)
            verts_transformed = verts_transformed[:, :3]
            
            # Get color from surface
            surface = vgeom.vmesh.surface
            if hasattr(surface, 'diffuse_texture') and surface.diffuse_texture is not None:
                color = surface.diffuse_texture.color
            elif surface.color is not None:
                color = surface.color
            else:
                color = (0.5, 0.5, 0.5)
            
            # Offset faces by current vertex count
            faces_offset = faces + vertex_offset
            
            # Create per-vertex colors
            vertex_colors = np.tile(color, (len(verts), 1))
            
            all_vertices.append(verts_transformed)
            all_faces.append(faces_offset)
            all_colors.append(vertex_colors)
            
            vertex_offset += len(verts)
        
        vertices = torch.from_numpy(np.vstack(all_vertices)).to(self.device, dtype=torch.float32)
        faces = torch.from_numpy(np.vstack(all_faces)).to(self.device, dtype=torch.int32)
        colors = torch.from_numpy(np.vstack(all_colors)).to(self.device, dtype=torch.float32)
        
        return vertices, faces, colors
