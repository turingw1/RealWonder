from simulation.case_simulation.case_handler import CaseHandler, register_case
import math
import numpy as np
import torch
import genesis as gs


@register_case("towel_folding")
class TowelFolding(CaseHandler):
    """PBD cloth towel fold on a high-friction camera-plane table.

    The reconstructed towel is the same SAM3D mesh/point-cloud object used by
    RealWonder. The table is modeled as a rigid support plane parallel to the
    image plane, so no cloth particles are fixed; only edge bands are moved as
    virtual gripper contact regions.
    """

    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self._init_bounds = []
        self._init_particles = []
        self._mask_reported = False
        self.table_plane_y = None

    def detect_ground_plane(self, ground_plane):
        lower = self.all_obj_occupied_lower_bound.detach().cpu().numpy()
        upper = self.all_obj_occupied_upper_bound.detach().cpu().numpy()
        center = (lower + upper) * 0.5
        axis = str(self.config.get("table_plane_axis", "y")).lower()

        if axis == "y":
            offset = float(self.config.get("table_plane_y_offset", 0.006))
            plane_y = float(lower[1] - abs(offset))
            self.table_plane_y = plane_y
            pos = (float(center[0]), plane_y, float(center[2]))
            normal = (0.0, 1.0, 0.0)
        else:
            offset = float(self.config.get("table_plane_z_offset", 0.004))
            plane_z = float(lower[2] - abs(offset))
            pos = (float(center[0]), float(center[1]), plane_z)
            normal = (0.0, 0.0, 1.0)

        self.normal = np.array(normal)
        self.ground_anchor = np.array(pos)
        self.scene.add_entity(
            material=gs.materials.Rigid(
                rho=float(self.config.get("plane_rho", 1000.0)),
                friction=min(5.0, float(self.config.get("plane_friction", 5.0))),
                coup_friction=min(5.0, float(self.config.get("plane_coup_friction", 5.0))),
                coup_softness=float(self.config.get("plane_coup_softness", 0.001)),
            ),
            morph=gs.morphs.Plane(pos=pos, normal=normal),
        )

    def after_scene_building(self):
        self._init_bounds = []
        self._init_particles = []
        for obj in self.all_objs:
            particles = torch.tensor(obj.init_particles, device=self.device, dtype=torch.float32)
            self._init_particles.append(particles)
            self._init_bounds.append(
                {
                    "min": particles.min(0).values,
                    "max": particles.max(0).values,
                    "center": particles.mean(0),
                }
            )
        self.fix_particles()

    def fix_particles(self):
        # Required by this demo: table friction, not pinned cloth anchors.
        return

    def custom_simulation(self, sid):
        total_steps = max(1, int(self.config["simulated_frames_num"]) * int(self.config["frame_steps"]))
        progress = min(1.0, sid / total_steps)
        damping = float(self.config.get("fold_velocity_damping", 0.90))
        blend = float(self.config.get("fold_handle_position_blend", 0.50))
        hold_blend = float(self.config.get("fold_handle_hold_blend", 0.10))
        velocity_gain = float(self.config.get("fold_velocity_gain", 7.0))

        half_end = float(self.config.get("fold_half_phase_end", 0.48))
        half_settle_end = float(self.config.get("fold_half_settle_end", 0.58))
        quarter_end = float(self.config.get("fold_quarter_phase_end", 0.88))
        half_end = max(1e-4, min(half_end, 0.80))
        half_settle_end = max(half_end + 1e-4, min(half_settle_end, 0.90))
        quarter_end = max(half_settle_end + 1e-4, min(quarter_end, 0.98))

        def smooth(phase):
            phase = max(0.0, min(1.0, phase))
            return 0.5 - 0.5 * torch.cos(torch.tensor(phase * math.pi, device=self.device))

        for obj_idx, obj in enumerate(self.all_objs):
            if self.config["material_type"][obj_idx] != "pbd_cloth":
                continue

            state = obj.solver.get_state(0)
            p_start = obj.particle_start
            n_p = obj.n_particles
            pos = state.pos[0, p_start:p_start + n_p, :]
            vel = state.vel[0, p_start:p_start + n_p, :]
            free = state.free[0, p_start:p_start + n_p].bool()

            b = self._init_bounds[obj_idx]
            init_pos = self._init_particles[obj_idx].to(pos.device)
            size = torch.clamp(b["max"] - b["min"], min=1e-6)
            x_norm = (init_pos[:, 0] - b["min"][0]) / size[0]
            z_norm = (init_pos[:, 2] - b["min"][2]) / size[2]

            edge_band = float(self.config.get("fold_grip_edge_band", 0.10))
            edge_margin = float(self.config.get("fold_grip_edge_margin", 0.04))
            left_handle = free & (x_norm < edge_band) & (z_norm > edge_margin) & (z_norm < 1.0 - edge_margin)
            bottom_handle = free & (z_norm < edge_band) & (x_norm > edge_margin) & (x_norm < 1.0 - edge_margin)

            if sid == 0 and not self._mask_reported:
                print(
                    "[towel_folding] table-friction fold masks: "
                    f"left_edge_handle={int(left_handle.sum().item())}, "
                    f"bottom_edge_handle={int(bottom_handle.sum().item())}, "
                    f"free={int(free.sum().item())}, fixed=0"
                )
                self._mask_reported = True

            vel.mul_(damping)

            crease_x = b["min"][0] + size[0] * float(self.config.get("fold_half_crease_x_norm", 0.50))
            crease_z = b["min"][2] + size[2] * float(self.config.get("fold_quarter_crease_z_norm", 0.50))
            lift_scale = float(self.config.get("fold_arc_lift_scale", 0.85))
            landing_y = float(self.config.get("fold_landing_y_offset", 0.010))

            def apply(mask, target, active_blend):
                if not bool(mask.any()):
                    return
                desired = pos[mask] * (1.0 - active_blend) + target[mask] * active_blend
                delta = desired - pos[mask]
                max_delta = float(self.config.get("fold_max_control_delta", 0.018))
                delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True).clamp(min=1e-8)
                delta = delta * torch.clamp(max_delta / delta_norm, max=1.0)
                pos[mask] = pos[mask] + delta
                vel[mask] = delta * velocity_gain

            def left_half_target(phase):
                ramp = smooth(phase)
                angle = ramp * math.pi
                dist = torch.clamp(crease_x - init_pos[:, 0], min=0.0)
                target = init_pos.clone()
                target[:, 0] = crease_x - dist * torch.cos(angle)
                target[:, 1] = init_pos[:, 1] + dist * torch.sin(angle) * lift_scale + landing_y * ramp
                return target

            def bottom_quarter_target(phase):
                ramp = smooth(phase)
                angle = ramp * math.pi
                dist = torch.clamp(crease_z - init_pos[:, 2], min=0.0)
                target = init_pos.clone()
                target[:, 2] = crease_z - dist * torch.cos(angle)
                target[:, 1] = init_pos[:, 1] + dist * torch.sin(angle) * lift_scale + landing_y * ramp
                return target

            if progress < half_end:
                target = left_half_target(progress / half_end)
                apply(left_handle, target, blend)
            else:
                target = left_half_target(1.0)
                apply(left_handle, target, hold_blend)

            if progress >= half_settle_end:
                quarter_phase = min(1.0, (progress - half_settle_end) / (quarter_end - half_settle_end))
                target = bottom_quarter_target(quarter_phase)
                active_blend = blend if progress < quarter_end else hold_blend
                apply(bottom_handle, target, active_blend)

            obj.solver.set_state(0, state)
