from __future__ import annotations

import numpy as np
import torch
import genesis as gs

from simulation.case_simulation.case_handler import CaseHandler, register_case


class BallDropLiquidBase(CaseHandler):
    """Shared handler for the water/honey rigid ball drop demos.

    The reconstructed foreground points remain responsible for RealWonder's
    video conditioning, while the physics proxy is deliberately simple and
    robust: a shallow particle liquid box, a rigid sphere, and fixed tray walls.
    """

    def _liquid_material_type(self):
        return self.config.get("material_type", [None])[0]

    def _uses_holdable_particle_liquid(self):
        return self._liquid_material_type() in {"pbd_liquid", "sph_liquid"}

    def _uses_scripted_pbd_liquid(self):
        return self._liquid_material_type() == "pbd_liquid"

    def _as_numpy(self, value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy().astype(np.float64)
        return np.asarray(value, dtype=np.float64)

    def _cfg_vec(self, key, default):
        return np.asarray(self.config.get(key, default), dtype=np.float64)

    def set_simulation_bounds(self, all_obj_occupied_lower_bound, all_obj_occupied_upper_bound):
        super().set_simulation_bounds(all_obj_occupied_lower_bound, all_obj_occupied_upper_bound)

        liquid_center, liquid_size = self._liquid_geometry()
        ball_center, ball_radius = self._ball_geometry(liquid_center, liquid_size)
        wall = float(self.config.get("liquid_tray_wall_thickness", 0.018))
        xy_clearance = float(
            self.config.get("liquid_tray_xy_clearance", self.config.get("liquid_tray_clearance", 0.0))
        )
        bottom_clearance = float(
            self.config.get("liquid_tray_bottom_clearance", self.config.get("liquid_tray_clearance", 0.0))
        )
        wall_height = float(
            self.config.get("liquid_tray_wall_height", max(liquid_size[2] * 2.8 + bottom_clearance, 0.12))
        )

        lower = liquid_center - liquid_size * 0.5
        upper = liquid_center + liquid_size * 0.5
        lower[:2] -= wall + xy_clearance
        upper[:2] += wall + xy_clearance
        lower[2] -= wall + bottom_clearance
        upper[2] = max(upper[2], lower[2] + wall + bottom_clearance + wall_height)
        lower = np.minimum(lower, ball_center - ball_radius)
        upper = np.maximum(upper, ball_center + ball_radius)

        margin = float(self.config.get("ball_drop_bounds_margin", 0.12))
        lower = torch.from_numpy(lower - margin).to(self.device, dtype=self.simulation_lower_bound.dtype)
        upper = torch.from_numpy(upper + margin).to(self.device, dtype=self.simulation_upper_bound.dtype)
        if self.config.get("liquid_tight_simulation_bounds", True):
            self.simulation_lower_bound = lower
            self.simulation_upper_bound = upper
        else:
            self.simulation_lower_bound = torch.minimum(self.simulation_lower_bound, lower)
            self.simulation_upper_bound = torch.maximum(self.simulation_upper_bound, upper)

    def _liquid_geometry(self):
        info = self.all_obj_info[0]
        center = self._as_numpy(info["center"])
        recon_size = np.maximum(self._as_numpy(info["size"]), 1e-4)

        min_size = self._cfg_vec("liquid_min_size", [0.32, 0.12, 0.045])
        max_size = self._cfg_vec("liquid_max_size", [0.72, 0.34, 0.12])
        scale = self._cfg_vec("liquid_size_scale", [1.15, 1.35, 0.45])
        size = np.clip(recon_size * scale, min_size, max_size)

        if "liquid_box_size" in self.config:
            size = self._cfg_vec("liquid_box_size", size)

        # Keep the physical proxy shallow even if single-view depth guesses a
        # thick volume.
        size[2] = float(self.config.get("liquid_height", size[2]))
        size = np.maximum(size, min_size)

        lower_z = center[2] - recon_size[2] * 0.5
        upper_z = center[2] + recon_size[2] * 0.5
        anchor = self.config.get("liquid_anchor", "bottom")
        if anchor == "top":
            # For no-container reference images, SAM3D reconstructs the visible
            # liquid as a deep slab. Anchor the shallow PBD proxy to the visible
            # surface instead of the reconstructed bottom.
            center[2] = upper_z - size[2] * 0.5 + float(self.config.get("liquid_top_offset", 0.0))
        elif anchor == "center":
            center[2] = center[2] + float(self.config.get("liquid_center_z_offset", 0.0))
        else:
            # Put the proxy bottom just above the reconstructed bottom. A small
            # clearance keeps initial PBD particles from interpenetrating the plane.
            clearance = float(
                self.config.get(
                    "liquid_bottom_clearance",
                    1.5 * float(self.config.get("particle_size", 0.01)),
                )
            )
            center[2] = lower_z + size[2] * 0.5 + clearance
        return center, size

    def _ball_geometry(self, liquid_center, liquid_size):
        info = self.all_obj_info[1]
        center = self._as_numpy(info["center"])
        recon_size = np.maximum(self._as_numpy(info["size"]), 1e-4)

        radius = float(
            self.config.get(
                "ball_radius",
                np.clip(np.max(recon_size) * 0.5, 0.035, 0.09),
            )
        )

        liquid_top = liquid_center[2] + liquid_size[2] * 0.5
        min_start_z = liquid_top + radius + float(self.config.get("ball_surface_gap", 0.055))
        if self.config.get("ball_snap_above_liquid", True):
            center[2] = max(center[2], min_start_z)

        if "ball_initial_pos" in self.config:
            center = self._cfg_vec("ball_initial_pos", center)

        return center, radius

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        if len(self.all_obj_info) < 2:
            raise ValueError(
                "ball_drop_liquid expects two segmented objects: "
                "object 0 = liquid pool, object 1 = ball."
            )
        if len(obj_materials) < 2:
            raise ValueError("ball_drop_liquid requires material_type: [pbd_liquid|sph_liquid, rigid].")

        self.obj_materials = obj_materials
        self.obj_vis_modes = obj_vis_modes
        self.scene = scene
        self.objs = []

        liquid_center, liquid_size = self._liquid_geometry()
        ball_center, ball_radius = self._ball_geometry(liquid_center, liquid_size)
        self.liquid_center = liquid_center
        self.liquid_size = liquid_size
        self.ball_center = ball_center
        self.ball_radius = ball_radius

        liquid_color = tuple(self.config.get("liquid_color", [0.35, 0.72, 1.0, 0.55]))
        ball_color = tuple(self.config.get("ball_color", [0.08, 0.09, 0.11, 1.0]))

        liquid = self.scene.add_entity(
            material=obj_materials[0],
            morph=gs.morphs.Box(
                pos=tuple(liquid_center),
                size=tuple(liquid_size),
            ),
            surface=gs.surfaces.Default(
                color=liquid_color,
                vis_mode=obj_vis_modes[0],
            ),
        )
        ball = self.scene.add_entity(
            material=obj_materials[1],
            morph=gs.morphs.Sphere(
                pos=tuple(ball_center),
                radius=ball_radius,
                fixed=False,
                collision=True,
                visualization=True,
            ),
            surface=gs.surfaces.Default(
                color=ball_color,
                vis_mode=obj_vis_modes[1],
            ),
        )

        self.objs.extend([liquid, ball])
        print(
            "[ball_drop_liquid] liquid box "
            f"center={liquid_center.round(4).tolist()}, size={liquid_size.round(4).tolist()}; "
            f"ball center={ball_center.round(4).tolist()}, radius={ball_radius:.4f}"
        )
        return self.objs

    def custom_setup(self):
        if not self.config.get("liquid_enable_tray_walls", True):
            self.tray_walls = []
            return None

        center = self.liquid_center
        size = self.liquid_size
        wall = float(self.config.get("liquid_tray_wall_thickness", 0.018))
        xy_clearance = float(
            self.config.get("liquid_tray_xy_clearance", self.config.get("liquid_tray_clearance", 0.0))
        )
        bottom_clearance = float(
            self.config.get("liquid_tray_bottom_clearance", self.config.get("liquid_tray_clearance", 0.0))
        )
        wall_height = float(self.config.get("liquid_tray_wall_height", max(size[2] * 2.8 + bottom_clearance, 0.12)))
        wall_base_z = center[2] - size[2] * 0.5 - bottom_clearance
        wall_z = wall_base_z + wall_height * 0.5
        half_x = size[0] * 0.5
        half_y = size[1] * 0.5

        material = gs.materials.Rigid(
            needs_coup=True,
            rho=1000.0,
            friction=float(self.config.get("liquid_tray_friction", 0.8)),
            coup_friction=float(self.config.get("liquid_tray_coup_friction", 1.0)),
            coup_softness=float(self.config.get("liquid_tray_coup_softness", 0.0015)),
        )
        surface = gs.surfaces.Default(
            color=tuple(self.config.get("liquid_tray_color", [0.55, 0.58, 0.60, 0.18])),
            vis_mode="visual",
        )
        visible = bool(self.config.get("liquid_tray_visible", False))

        def add_wall(name, pos, box_size):
            entity = self.scene.add_entity(
                material=material,
                morph=gs.morphs.Box(
                    pos=tuple(pos),
                    size=tuple(box_size),
                    fixed=True,
                    collision=True,
                    visualization=visible,
                ),
                surface=surface,
            )
            return name, entity

        wall_span_x = size[0] + 2 * (wall + xy_clearance)
        wall_span_y = size[1] + 2 * (wall + xy_clearance)
        self.tray_walls = [
            add_wall("left", center + np.array([-half_x - xy_clearance - wall * 0.5, 0.0, wall_z - center[2]]), (wall, wall_span_y, wall_height)),
            add_wall("right", center + np.array([half_x + xy_clearance + wall * 0.5, 0.0, wall_z - center[2]]), (wall, wall_span_y, wall_height)),
            add_wall("front", center + np.array([0.0, -half_y - xy_clearance - wall * 0.5, wall_z - center[2]]), (wall_span_x, wall, wall_height)),
            add_wall("back", center + np.array([0.0, half_y + xy_clearance + wall * 0.5, wall_z - center[2]]), (wall_span_x, wall, wall_height)),
        ]
        if self.config.get("liquid_enable_tray_bottom", True):
            bottom_z = center[2] - size[2] * 0.5 - bottom_clearance - wall * 0.5
            self.tray_walls.append(
                add_wall(
                    "bottom",
                    np.array([center[0], center[1], bottom_z]),
                    (wall_span_x, wall_span_y, wall),
                )
            )

    def fix_particles(self):
        if len(self.all_objs) > 1 and self.config.get("ball_lock_xy", True):
            self.ball_initial_xy = self.all_objs[1].get_pos().detach().cpu().numpy()[:2].copy()

        if len(self.all_objs) > 1 and "ball_initial_velocity" in self.config:
            vel = np.asarray(self.config["ball_initial_velocity"], dtype=np.float64)
            qvel = np.zeros(int(getattr(self.all_objs[1], "n_dofs", 0)), dtype=np.float64)
            if qvel.size >= 3:
                qvel[:3] = vel[:3]
                self.all_objs[1].set_dofs_velocity(qvel)
                print(f"[ball_drop_liquid] initial ball qvel={qvel.round(4).tolist()}")

    def _lock_ball_xy(self):
        if not self.config.get("ball_lock_xy", True) or not hasattr(self, "ball_initial_xy"):
            return
        pos = self.all_objs[1].get_pos().detach().cpu().numpy()
        pos[:2] = self.ball_initial_xy
        self.all_objs[1].set_pos(pos, zero_velocity=False)

    def _limit_ball_sink(self, sid):
        if not self.config.get("ball_limit_sink", True):
            return
        if len(self.all_objs) < 2:
            return

        ball = self.all_objs[1]
        pos = ball.get_pos().detach().cpu().numpy()
        liquid_top = self.liquid_center[2] + self.liquid_size[2] * 0.5
        contact_started = hasattr(self, "splash_start_sid") or pos[2] - self.ball_radius <= liquid_top
        if not contact_started:
            return

        if not hasattr(self, "ball_contact_sid"):
            self.ball_contact_sid = getattr(self, "splash_start_sid", sid)

        progress = max(0, sid - self.ball_contact_sid)
        start_depth = float(self.config.get("ball_sink_start_depth", 0.018))
        speed = float(self.config.get("ball_sink_speed_per_step", 0.0022))
        max_depth = float(self.config.get("ball_max_sink_depth", self.liquid_size[2] + self.ball_radius * 1.2))
        allowed_depth = min(max_depth, start_depth + speed * progress)
        min_z = liquid_top + self.ball_radius - allowed_depth

        if pos[2] < min_z:
            pos[2] = min_z
            ball.set_pos(pos, zero_velocity=True)

    def _log_liquid_particle_bounds(self, sid):
        if not self.config.get("liquid_debug_particle_bounds", False):
            return
        interval = int(self.config.get("liquid_debug_particle_bounds_interval", 20))
        if interval <= 0 or sid % interval != 0:
            return
        if len(self.all_objs) < 1:
            return

        particles = np.asarray(self.all_objs[0].get_particles()[0])
        if particles.size == 0:
            return
        lower = particles.min(axis=0)
        upper = particles.max(axis=0)
        liquid_lower = self.liquid_center - self.liquid_size * 0.5
        liquid_upper = self.liquid_center + self.liquid_size * 0.5
        outside_xy = np.any((particles[:, :2] < liquid_lower[:2]) | (particles[:, :2] > liquid_upper[:2]), axis=1)
        below = particles[:, 2] < liquid_lower[2]
        ball_pos = self.all_objs[1].get_pos().detach().cpu().numpy() if len(self.all_objs) > 1 else None
        ball_text = ""
        if ball_pos is not None:
            liquid_top = self.liquid_center[2] + self.liquid_size[2] * 0.5
            ball_text = f" ball_z={ball_pos[2]:.4f} ball_bottom={ball_pos[2] - self.ball_radius:.4f} liquid_top={liquid_top:.4f}"
        print(
            "[ball_drop_liquid] "
            f"sid={sid:04d} particle_bounds="
            f"{lower.round(4).tolist()}..{upper.round(4).tolist()} "
            f"outside_xy={int(outside_xy.sum())}/{particles.shape[0]} "
            f"below_liquid_bottom={int(below.sum())}/{particles.shape[0]}"
            f"{ball_text}"
        )

    def _drive_ball_sink(self, sid):
        if not self.config.get("ball_drive_sink", False):
            return
        if len(self.all_objs) < 2:
            return

        ball = self.all_objs[1]
        pos = ball.get_pos().detach().cpu().numpy()
        liquid_top = self.liquid_center[2] + self.liquid_size[2] * 0.5
        trigger_margin = float(self.config.get("ball_drive_sink_trigger_margin", 0.0))
        if pos[2] - self.ball_radius > liquid_top + trigger_margin:
            return

        if not hasattr(self, "ball_drive_sink_sid"):
            self.ball_drive_sink_sid = sid

        progress = max(0, sid - self.ball_drive_sink_sid)
        start_depth = float(self.config.get("ball_drive_sink_start_depth", 0.0))
        speed = float(self.config.get("ball_drive_sink_speed_per_step", 0.001))
        max_depth = float(self.config.get("ball_drive_sink_max_depth", self.ball_radius))
        target_depth = min(max_depth, start_depth + speed * progress)
        target_z = liquid_top + self.ball_radius - target_depth

        if pos[2] > target_z:
            pos[2] = target_z
            ball.set_pos(pos, zero_velocity=False)

    def custom_simulation(self, sid):
        # Gravity drives the drop; x/y are locked so the rendered ball falls
        # through the same image-space column as the input ball.
        self._lock_ball_xy()
        return None

    def _hold_liquid_pre_contact(self, sid):
        if not self._uses_holdable_particle_liquid():
            return
        if not self.config.get("liquid_hold_until_splash", True):
            return
        if len(self.all_objs) < 2 or hasattr(self, "splash_start_sid"):
            return

        ball_pos = self.all_objs[1].get_pos().detach().cpu().numpy()
        liquid_top = self.liquid_center[2] + self.liquid_size[2] * 0.5
        trigger_margin = float(self.config.get("liquid_splash_trigger_margin", 0.0))
        if ball_pos[2] - self.ball_radius <= liquid_top + trigger_margin:
            return

        liquid = self.all_objs[0]
        pos_all = liquid.solver.particles.pos.to_numpy()
        vel_all = liquid.solver.particles.vel.to_numpy()
        start, end = liquid.particle_start, liquid.particle_end
        init = liquid.init_particles
        if pos_all.ndim == 4:
            pos_all[0, start:end, 0] = init
            vel_all[0, start:end, 0] = 0.0
        else:
            pos_all[start:end, 0] = init
            vel_all[start:end, 0] = 0.0
        liquid.solver.particles.pos.from_numpy(pos_all)
        liquid.solver.particles.vel.from_numpy(vel_all)

    def _inject_splash(self, sid):
        if not self._uses_scripted_pbd_liquid():
            return
        if not self.config.get("liquid_splash_enabled", False):
            return
        if len(self.all_objs) < 2:
            return

        ball_pos = self.all_objs[1].get_pos().detach().cpu().numpy()
        liquid_top = self.liquid_center[2] + self.liquid_size[2] * 0.5
        trigger_margin = float(self.config.get("liquid_splash_trigger_margin", 0.02))
        if ball_pos[2] - self.ball_radius > liquid_top + trigger_margin:
            return

        if not hasattr(self, "splash_start_sid"):
            self.splash_start_sid = sid
        splash_frames = int(self.config.get("liquid_splash_frames", 18))
        progress = sid - self.splash_start_sid
        if progress < 0 or progress >= splash_frames:
            return

        liquid = self.all_objs[0]
        pos_all = liquid.solver.particles.pos.to_numpy()
        vel_all = liquid.solver.particles.vel.to_numpy()
        start, end = liquid.particle_start, liquid.particle_end
        if pos_all.ndim == 4:
            pos = pos_all[0, start:end, 0].copy()
            vel = vel_all[0, start:end, 0].copy()
        else:
            pos = pos_all[start:end, 0].copy()
            vel = vel_all[start:end, 0].copy()

        rel_xy = pos[:, :2] - ball_pos[:2]
        dist = np.linalg.norm(rel_xy, axis=1)
        radius = float(self.config.get("liquid_splash_radius", max(self.ball_radius * 3.0, 0.18)))
        band = float(self.config.get("liquid_splash_surface_band", max(self.liquid_size[2] * 1.2, 0.08)))
        near_surface = pos[:, 2] > liquid_top - band
        mask = (dist < radius) & near_surface
        if not np.any(mask):
            return

        radial = np.zeros_like(rel_xy)
        radial[mask] = rel_xy[mask] / np.maximum(dist[mask, None], 1e-6)
        radial[:, 0] += 0.15 * np.sign(rel_xy[:, 0])
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        radial = radial / np.maximum(radial_norm, 1e-6)

        temporal = max(0.0, 1.0 - progress / max(1, splash_frames - 1))
        weight = np.zeros(pos.shape[0], dtype=pos.dtype)
        weight[mask] = (1.0 - dist[mask] / radius) * temporal
        up_strength = float(self.config.get("liquid_splash_up_strength", 0.045))
        out_strength = float(self.config.get("liquid_splash_out_strength", 0.018))
        up_velocity = float(self.config.get("liquid_splash_up_velocity", 1.0))
        out_velocity = float(self.config.get("liquid_splash_out_velocity", 0.25))

        pos[:, 2] += up_strength * weight
        pos[:, :2] += out_strength * weight[:, None] * radial
        vel[:, 2] += up_velocity * weight
        vel[:, :2] += out_velocity * weight[:, None] * radial

        if pos_all.ndim == 4:
            pos_all[0, start:end, 0] = pos
            vel_all[0, start:end, 0] = vel
        else:
            pos_all[start:end, 0] = pos
            vel_all[start:end, 0] = vel
        liquid.solver.particles.pos.from_numpy(pos_all)
        liquid.solver.particles.vel.from_numpy(vel_all)

    def after_physics_step(self, sid):
        self._hold_liquid_pre_contact(sid)
        self._inject_splash(sid)
        self._drive_ball_sink(sid)
        self._limit_ball_sink(sid)
        self._lock_ball_xy()
        self._log_liquid_particle_bounds(sid)


@register_case("ball_drop_water")
class BallDropWater(BallDropLiquidBase):
    pass


@register_case("ball_drop_honey")
class BallDropHoney(BallDropLiquidBase):
    pass
