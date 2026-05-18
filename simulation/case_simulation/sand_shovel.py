import os

import numpy as np
import torch
import genesis as gs

from simulation.utils import gs_to_pt3d
from .case_handler import CaseHandler, register_case


@register_case("sand_shovel")
class SandShovel(CaseHandler):
    """Kinematic shovel tool for an MPM sand surface."""

    def custom_setup(self):
        obj = self.all_obj_info[0]
        self.sand_center = obj["center"].detach().cpu().numpy()
        self.sand_top_z = float(self.all_obj_occupied_upper_bound.detach().cpu().numpy()[2])

        self.blade_size = np.array(self.config.get("shovel_blade_size", [0.34, 0.11, 0.035]), dtype=np.float64)
        self.blade_color = tuple(self.config.get("shovel_blade_color", [0.55, 0.58, 0.60, 1.0]))
        self.blade_mesh_path = os.path.join(self.config["data_path"], "shovel_blade.obj")

        start_pos, start_quat = self._pose_from_config("shovel_start_offset", "shovel_start_pitch_deg")
        self.current_pos = start_pos
        self.current_quat = start_quat

        self.shovel = self.scene.add_entity(
            material=gs.materials.Tool(
                friction=float(self.config.get("shovel_friction", 8.0)),
                coup_softness=float(self.config.get("shovel_coup_softness", 0.012)),
                sdf_res=int(self.config.get("shovel_sdf_res", 128)),
            ),
            morph=gs.morphs.Mesh(
                file=self.blade_mesh_path,
                scale=tuple(self.blade_size),
                pos=tuple(start_pos),
                euler=(0.0, 0.0, 0.0),
            ),
            surface=gs.surfaces.Default(
                color=self.blade_color,
                vis_mode="visual",
            ),
        )

    def init_robots_pose(self):
        self.shovel.set_position(self._batched(self.current_pos))
        self.shovel.set_quaternion(self._batched(self.current_quat))

    def custom_simulation(self, sid):
        frame_t = sid / max(float(self.config.get("frame_steps", 1)), 1.0)
        pos, quat = self._trajectory_pose(frame_t)
        vel = (pos - self.current_pos) / max(float(self.config.get("dt", 0.01)), 1e-6)

        self.current_pos = pos
        self.current_quat = quat
        self.shovel.set_position(self._batched(pos))
        self.shovel.set_quaternion(self._batched(quat))
        self.shovel.set_velocity(vel=self._batched(vel))

    def after_simulation_step(self, svr):
        vertices, faces, colors = self._tool_visual_mesh()
        svr.franka_mesh = {
            "vertices": vertices,
            "faces": faces,
            "colors": colors,
        }

    def _pose_from_config(self, offset_key, pitch_key):
        offset = np.array(self.config[offset_key], dtype=np.float64)
        pos = self._world_from_offset(offset)
        quat = gs.utils.geom.xyz_to_quat(
            np.array([float(self.config.get(pitch_key, 0.0)), 0.0, 0.0], dtype=np.float64),
            rpy=True,
            degrees=True,
        )
        return pos, quat.astype(np.float64)

    def _world_from_offset(self, offset):
        return np.array(
            [
                self.sand_center[0] + offset[0],
                self.sand_center[1] + offset[1],
                self.sand_top_z + offset[2],
            ],
            dtype=np.float64,
        )

    def _trajectory_pose(self, frame_t):
        waypoints = [
            ("shovel_start_offset", "shovel_start_pitch_deg"),
            ("shovel_under_sand_offset", "shovel_dig_pitch_deg"),
            ("shovel_scoop_offset", "shovel_dig_pitch_deg"),
            ("shovel_lift_offset", "shovel_lift_pitch_deg"),
            ("shovel_dump_offset", "shovel_dump_pitch_deg"),
            ("shovel_dump_offset", "shovel_dump_pitch_deg"),
        ]
        phase_keys = ["approach", "dig_under", "scoop_forward", "lift", "dump", "settle"]
        phases = self.config.get("shovel_phase_frames", {})
        durations = [float(phases.get(key, 10)) for key in phase_keys]

        elapsed = 0.0
        for phase_idx, duration in enumerate(durations):
            next_elapsed = elapsed + max(duration, 1.0)
            if frame_t <= next_elapsed or phase_idx == len(durations) - 1:
                local_t = np.clip((frame_t - elapsed) / max(duration, 1.0), 0.0, 1.0)
                local_t = local_t * local_t * (3.0 - 2.0 * local_t)
                start_pos, start_quat = self._pose_from_config(*waypoints[phase_idx])
                end_pos, end_quat = self._pose_from_config(*waypoints[min(phase_idx + 1, len(waypoints) - 1)])
                pos = (1.0 - local_t) * start_pos + local_t * end_pos
                quat = self._normalize_quat((1.0 - local_t) * start_quat + local_t * end_quat)
                return pos, quat
            elapsed = next_elapsed

        return self._pose_from_config(*waypoints[-1])

    def _normalize_quat(self, quat):
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm

    def _batched(self, value):
        return np.asarray(value, dtype=np.float32)[None, :]

    def _tool_visual_mesh(self):
        blade_vertices = np.asarray(self.shovel.mesh.raw_vertices, dtype=np.float64)
        blade_faces = np.asarray(self.shovel.mesh.faces_np, dtype=np.int64).reshape(-1, 3)
        blade_vertices = self._transform_vertices(blade_vertices, self.current_pos, self.current_quat)

        handle_vertices, handle_faces = self._box_mesh(
            center=self.current_pos + np.array([0.0, -0.19, 0.105], dtype=np.float64),
            size=np.array([0.055, 0.28, 0.055], dtype=np.float64),
            quat=self.current_quat,
        )
        handle_faces = handle_faces + len(blade_vertices)

        vertices_gs = np.vstack([blade_vertices, handle_vertices])
        faces_np = np.vstack([blade_faces, handle_faces])
        colors_np = np.vstack(
            [
                np.tile(np.array(self.blade_color[:3], dtype=np.float32), (len(blade_vertices), 1)),
                np.tile(np.array([0.18, 0.18, 0.18], dtype=np.float32), (len(handle_vertices), 1)),
            ]
        )

        vertices = gs_to_pt3d(torch.from_numpy(vertices_gs).to(self.device, dtype=torch.float32))
        faces = torch.from_numpy(faces_np).to(self.device, dtype=torch.int64)
        colors = torch.from_numpy(colors_np).to(self.device, dtype=torch.float32)
        return vertices, faces, colors

    def _transform_vertices(self, vertices, pos, quat):
        rot = gs.utils.geom.quat_to_R(quat.astype(np.float64))
        return vertices @ rot.T + pos

    def _box_mesh(self, center, size, quat):
        sx, sy, sz = size / 2.0
        vertices = np.array(
            [
                [-sx, -sy, -sz],
                [sx, -sy, -sz],
                [sx, sy, -sz],
                [-sx, sy, -sz],
                [-sx, -sy, sz],
                [sx, -sy, sz],
                [sx, sy, sz],
                [-sx, sy, sz],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 7, 6],
                [4, 6, 5],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=np.int64,
        )
        return self._transform_vertices(vertices, center, quat), faces
