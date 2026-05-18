from simulation.case_simulation.case_handler import CaseHandler, register_case
import math
import torch


@register_case("tshirt_folding")
class TShirtFolding(CaseHandler):
    """PBD cloth case that folds a reconstructed T-shirt with velocity fields.

    This keeps the RealWonder santa_cloth modeling path: a SAM/SAM3D mesh is
    converted to Genesis PBD cloth particles, then case-specific Genesis logic
    drives the particle motion.
    """

    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self._init_bounds = []
        self._init_particles = []
        self._mask_reported = False

    def detect_ground_plane(self, ground_plane):
        # The single-view garment mesh is a camera-facing cloth surface, like
        # santa_cloth. A horizontal rigid plane tends to overconstrain it.
        pass

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
        if not self.config.get("fix_collar", True):
            return
        top_band = float(self.config.get("fixed_top_band", 0.08))
        center_width = float(self.config.get("fixed_center_width", 0.34))
        for i, obj in enumerate(self.all_objs):
            particles = torch.tensor(obj.init_particles, device=self.device, dtype=torch.float32)
            b = self._init_bounds[i]
            size = torch.clamp(b["max"] - b["min"], min=1e-6)
            x_norm = (particles[:, 0] - b["min"][0]) / size[0]
            z_norm = (particles[:, 2] - b["min"][2]) / size[2]
            center_mask = (x_norm > 0.5 - center_width / 2) & (x_norm < 0.5 + center_width / 2)
            top_mask = z_norm > 1.0 - top_band
            fixed_points = particles[center_mask & top_mask]
            for point in fixed_points:
                obj.fix_particle(obj.find_closest_particle(tuple(point.tolist())), 0)
            print(f"[tshirt_folding] fixed {len(fixed_points)} collar/top particles")

    def custom_simulation(self, sid):
        if (
            self.config.get("fold_strategy") == "flip_then_drag_fold"
            or self.config.get("fold_test_mode") == "flip_then_drag"
        ):
            return self._custom_flip_then_drag_fold(sid)

        total_steps = max(1, int(self.config["simulated_frames_num"]) * int(self.config["frame_steps"]))
        progress = min(1.0, sid / total_steps)
        damping = float(self.config.get("fold_velocity_damping", 0.92))

        side_end = float(self.config.get("fold_side_phase_end", 0.62))
        bottom_end = float(self.config.get("fold_bottom_phase_end", 0.86))
        side_end = max(1e-4, min(side_end, 0.95))
        bottom_end = max(side_end + 1e-4, min(bottom_end, 0.98))

        grasp_side = str(self.config.get("fold_grasp_side", "right")).lower()
        edge_band = float(self.config.get("fold_handle_edge_band", 0.075))
        handle_z_min = float(self.config.get("fold_handle_z_min", 0.18))
        handle_z_max = float(self.config.get("fold_handle_z_max", 0.86))
        corner_x_band = float(self.config.get("fold_bottom_corner_x_band", 0.16))
        corner_z_max = float(self.config.get("fold_bottom_corner_z_max", 0.20))
        side_landing_x_norm = float(self.config.get("fold_side_landing_x_norm", 0.38))
        side_landing_z_shift = float(self.config.get("fold_side_landing_z_shift", 0.0))
        side_lift = float(self.config.get("fold_handle_lift_amount", self.config.get("fold_lift_amount", 0.18)))
        landing_y_offset = float(self.config.get("fold_landing_y_offset", 0.015))
        side_blend = float(self.config.get("fold_handle_position_blend", self.config.get("fold_position_blend", 0.62)))
        hold_blend = float(self.config.get("fold_handle_hold_blend", 0.18))
        side_velocity_gain = float(self.config.get("fold_side_strength", 7.0))

        carry_start = float(self.config.get("fold_passive_carry_start_x_norm", 0.82))
        passive_blend = float(self.config.get("fold_passive_drag_blend", 0.0))
        passive_strength = float(self.config.get("fold_passive_drag_strength", 0.25))

        bottom_z_band = float(self.config.get("fold_bottom_handle_z_band", 0.09))
        bottom_x_min = float(self.config.get("fold_bottom_handle_x_min", 0.30))
        bottom_x_max = float(self.config.get("fold_bottom_handle_x_max", 0.72))
        bottom_landing_z_norm = float(
            self.config.get("fold_bottom_landing_z_norm", self.config.get("bottom_fold_target_z", 0.58))
        )
        bottom_lift = float(self.config.get("fold_bottom_lift_amount", side_lift * 0.70))
        bottom_blend = float(self.config.get("fold_bottom_position_blend", side_blend * 0.75))
        bottom_velocity_gain = float(self.config.get("fold_bottom_strength", 6.0))

        def smooth(phase):
            phase = max(0.0, min(1.0, phase))
            return 0.5 - 0.5 * torch.cos(torch.tensor(phase * math.pi, device=self.device))

        def arc(phase):
            phase = max(0.0, min(1.0, phase))
            return torch.sin(torch.tensor(phase * math.pi, device=self.device))

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

            vel.mul_(damping)
            target = pos.clone()

            if grasp_side == "left":
                side_edge_mask = x_norm < edge_band
                side_corner_mask = (x_norm < corner_x_band) & (z_norm < corner_z_max)
                source_x = b["min"][0]
                landing_norm = 1.0 - side_landing_x_norm
                carry_weight = torch.clamp((carry_start - x_norm) / max(carry_start, 1e-6), 0.0, 1.0)
                passive_mask = free & (x_norm < carry_start) & (z_norm > handle_z_min) & (z_norm < handle_z_max)
            else:
                side_edge_mask = x_norm > 1.0 - edge_band
                side_corner_mask = (x_norm > 1.0 - corner_x_band) & (z_norm < corner_z_max)
                source_x = b["max"][0]
                landing_norm = side_landing_x_norm
                carry_weight = torch.clamp((x_norm - carry_start) / max(1.0 - carry_start, 1e-6), 0.0, 1.0)
                passive_mask = free & (x_norm > carry_start) & (z_norm > handle_z_min) & (z_norm < handle_z_max)

            side_handle = free & ((side_edge_mask & (z_norm > handle_z_min) & (z_norm < handle_z_max)) | side_corner_mask)
            passive_mask = passive_mask & (~side_handle)
            bottom_handle = free & (z_norm < bottom_z_band) & (x_norm > bottom_x_min) & (x_norm < bottom_x_max)

            landing_x = b["min"][0] + size[0] * landing_norm
            side_delta_x = landing_x - source_x
            side_delta_z = size[2] * side_landing_z_shift
            bottom_landing_z = b["min"][2] + size[2] * bottom_landing_z_norm

            if sid == 0 and not self._mask_reported:
                print(
                    "[tshirt_folding] handle-pull masks: "
                    f"side_handle={int(side_handle.sum().item())}, "
                    f"passive_carry={int(passive_mask.sum().item())}, "
                    f"bottom_handle={int(bottom_handle.sum().item())}, "
                    f"free={int(free.sum().item())}"
                )
                self._mask_reported = True

            # Phase 1: a small side-edge gripper lifts the garment edge and places it across the midline.
            side_phase = min(1.0, progress / side_end)
            side_ramp = smooth(side_phase)
            target[side_handle, 0] = init_pos[side_handle, 0] + side_delta_x * side_ramp
            target[side_handle, 1] = init_pos[side_handle, 1] + landing_y_offset * side_ramp + side_lift * arc(side_phase)
            target[side_handle, 2] = init_pos[side_handle, 2] + side_delta_z * side_ramp

            if progress <= side_end:
                if passive_blend > 0.0:
                    weighted_shift = side_delta_x * side_ramp * passive_strength * carry_weight[passive_mask]
                    target[passive_mask, 0] = init_pos[passive_mask, 0] + weighted_shift
                    target[passive_mask, 1] = (
                        init_pos[passive_mask, 1]
                        + landing_y_offset * side_ramp * passive_strength * carry_weight[passive_mask]
                        + side_lift * arc(side_phase) * passive_strength * carry_weight[passive_mask]
                    )
                    pos[passive_mask] = pos[passive_mask] * (1.0 - passive_blend) + target[passive_mask] * passive_blend
                    vel[passive_mask] = (target[passive_mask] - pos[passive_mask]) * side_velocity_gain * 0.35
                pos[side_handle] = pos[side_handle] * (1.0 - side_blend) + target[side_handle] * side_blend
                vel[side_handle] = (target[side_handle] - pos[side_handle]) * side_velocity_gain
            else:
                pos[side_handle] = pos[side_handle] * (1.0 - hold_blend) + target[side_handle] * hold_blend
                vel[side_handle] = (target[side_handle] - pos[side_handle]) * side_velocity_gain * 0.45

            # Phase 2: after the side lands, a bottom-hem gripper folds the lower edge upward.
            if progress > side_end:
                bottom_phase = min(1.0, (progress - side_end) / (bottom_end - side_end))
                bottom_ramp = smooth(bottom_phase)
                rightness = torch.clamp((x_norm - 0.5) / 0.5, 0.0, 1.0)
                if grasp_side == "left":
                    rightness = torch.clamp((0.5 - x_norm) / 0.5, 0.0, 1.0)
                target[bottom_handle, 0] = init_pos[bottom_handle, 0] + side_delta_x * rightness[bottom_handle] * bottom_ramp
                target[bottom_handle, 1] = (
                    init_pos[bottom_handle, 1]
                    + landing_y_offset * bottom_ramp
                    + bottom_lift * arc(bottom_phase)
                )
                target[bottom_handle, 2] = init_pos[bottom_handle, 2] + (
                    bottom_landing_z - init_pos[bottom_handle, 2]
                ) * bottom_ramp

                active_blend = bottom_blend if progress <= bottom_end else hold_blend
                pos[bottom_handle] = pos[bottom_handle] * (1.0 - active_blend) + target[bottom_handle] * active_blend
                vel[bottom_handle] = (target[bottom_handle] - pos[bottom_handle]) * bottom_velocity_gain

            if progress > bottom_end:
                vel.mul_(0.88)

            obj.solver.set_state(0, state)

    def _custom_flip_then_drag_fold(self, sid):
        total_steps = max(1, int(self.config["simulated_frames_num"]) * int(self.config["frame_steps"]))
        progress = min(1.0, sid / total_steps)
        damping = float(self.config.get("fold_velocity_damping", 0.90))
        blend = float(self.config.get("fold_flip_position_blend", 0.26))
        max_delta = float(self.config.get("fold_flip_max_control_delta", 0.02))

        def smooth(phase):
            phase = max(0.0, min(1.0, phase))
            return 0.5 - 0.5 * torch.cos(torch.tensor(phase * math.pi, device=self.device))

        def apply_region(pos, vel, mask, target, region_blend):
            if not bool(mask.any()):
                return
            proposed = pos[mask] * (1.0 - region_blend) + target[mask] * region_blend
            if max_delta > 0:
                delta = proposed - pos[mask]
                delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True).clamp(min=1e-8)
                scale = torch.clamp(max_delta / delta_norm, max=1.0)
                proposed = pos[mask] + delta * scale
            pos[mask] = proposed
            vel[mask] = 0.0

        def side_flip_target(init_pos, crease_x, is_left, phase, drag_phase=0.0):
            ramp = smooth(phase)
            angle = ramp * math.pi
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            target = init_pos.clone()
            dist = torch.abs(init_pos[:, 0] - crease_x)
            direction = -1.0 if is_left else 1.0
            target[:, 0] = crease_x + direction * dist * cos_a
            target[:, 1] = init_pos[:, 1] + dist * sin_a * float(self.config.get("fold_flip_lift_scale", 0.55))
            if drag_phase > 0:
                drag = smooth(drag_phase)
                pmin_x = init_pos[:, 0].min()
                pmax_x = init_pos[:, 0].max()
                center_world = pmin_x + (pmax_x - pmin_x) * float(self.config.get("fold_center_target_x_norm", 0.50))
                strength = drag * float(self.config.get("fold_side_drag_to_center_strength", 0.18))
                target[:, 0] = target[:, 0] * (1.0 - strength) + center_world * strength
                target[:, 1] = target[:, 1] * (1.0 - drag * 0.55) + (init_pos[:, 1] + 0.018) * (drag * 0.55)
            return target

        def bottom_flip_target(init_pos, crease_z, phase, drag_phase=0.0):
            ramp = smooth(phase)
            angle = ramp * math.pi
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            target = init_pos.clone()
            dist = torch.clamp(crease_z - init_pos[:, 2], min=0.0)
            target[:, 2] = crease_z - dist * cos_a
            target[:, 1] = init_pos[:, 1] + dist * sin_a * float(self.config.get("fold_bottom_flip_lift_scale", 0.52))
            if drag_phase > 0:
                drag = smooth(drag_phase)
                pmin_z = init_pos[:, 2].min()
                pmax_z = init_pos[:, 2].max()
                center_z = pmin_z + (pmax_z - pmin_z) * float(self.config.get("fold_bottom_drag_target_z_norm", 0.56))
                target[:, 2] = target[:, 2] * (1.0 - drag * 0.42) + center_z * (drag * 0.42)
                target[:, 1] = target[:, 1] * (1.0 - drag * 0.55) + (init_pos[:, 1] + 0.015) * (drag * 0.55)
            return target

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

            left_mask = free & (x_norm < float(self.config.get("fold_left_region_max_x_norm", 0.30))) & (
                z_norm > float(self.config.get("fold_side_region_min_z_norm", 0.38))
            )
            right_mask = free & (x_norm > float(self.config.get("fold_right_region_min_x_norm", 0.70))) & (
                z_norm > float(self.config.get("fold_side_region_min_z_norm", 0.38))
            )
            bottom_mask = free & (z_norm < float(self.config.get("fold_bottom_region_max_z_norm", 0.34))) & (
                x_norm > float(self.config.get("fold_bottom_region_min_x_norm", 0.24))
            ) & (x_norm < float(self.config.get("fold_bottom_region_max_x_norm", 0.76)))

            if sid == 0 and not self._mask_reported:
                print(
                    "[tshirt_folding] flip-then-drag masks: "
                    f"left_sleeve={int(left_mask.sum().item())}, "
                    f"right_sleeve={int(right_mask.sum().item())}, "
                    f"bottom={int(bottom_mask.sum().item())}, "
                    f"free={int(free.sum().item())}"
                )
                self._mask_reported = True

            vel.mul_(damping)

            left_crease = b["min"][0] + size[0] * float(self.config.get("fold_left_crease_x_norm", 0.32))
            right_crease = b["min"][0] + size[0] * float(self.config.get("fold_right_crease_x_norm", 0.68))
            bottom_crease = b["min"][2] + size[2] * float(self.config.get("fold_bottom_crease_z_norm", 0.36))

            if progress < 0.18:
                target = side_flip_target(init_pos, left_crease, is_left=True, phase=progress / 0.18)
                apply_region(pos, vel, left_mask, target, blend)
            elif progress < 0.30:
                target = side_flip_target(
                    init_pos,
                    left_crease,
                    is_left=True,
                    phase=1.0,
                    drag_phase=(progress - 0.18) / 0.12,
                )
                apply_region(pos, vel, left_mask, target, blend * 0.85)
            elif progress < 0.48:
                target = side_flip_target(init_pos, right_crease, is_left=False, phase=(progress - 0.30) / 0.18)
                apply_region(pos, vel, right_mask, target, blend)
            elif progress < 0.60:
                target = side_flip_target(
                    init_pos,
                    right_crease,
                    is_left=False,
                    phase=1.0,
                    drag_phase=(progress - 0.48) / 0.12,
                )
                apply_region(pos, vel, right_mask, target, blend * 0.85)
            elif progress < 0.82:
                target = bottom_flip_target(init_pos, bottom_crease, phase=(progress - 0.60) / 0.22)
                apply_region(pos, vel, bottom_mask, target, blend)
            elif progress < 0.94:
                target = bottom_flip_target(
                    init_pos,
                    bottom_crease,
                    phase=1.0,
                    drag_phase=(progress - 0.82) / 0.12,
                )
                apply_region(pos, vel, bottom_mask, target, blend * 0.85)
            else:
                target_left = side_flip_target(init_pos, left_crease, is_left=True, phase=1.0, drag_phase=1.0)
                target_right = side_flip_target(init_pos, right_crease, is_left=False, phase=1.0, drag_phase=1.0)
                target_bottom = bottom_flip_target(init_pos, bottom_crease, phase=1.0, drag_phase=1.0)
                apply_region(pos, vel, left_mask, target_left, 0.08)
                apply_region(pos, vel, right_mask, target_right, 0.08)
                apply_region(pos, vel, bottom_mask, target_bottom, 0.08)

            obj.solver.set_state(0, state)
