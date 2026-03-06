"""Santa cloth demo case handler — PBD cloth with controllable wind."""

import numpy as np
import torch

from case_handlers.base import DemoCaseHandler, register_demo_case


@register_demo_case("santa_cloth")
class SantaClothDemoHandler(DemoCaseHandler):

    force_scale = 1.0

    def __init__(self, config):
        super().__init__(config)
        self._wind_direction = np.zeros(3, dtype=np.float32)
        self._wind_strength = 0.0
        self._wind_bounds = None  # (z_low, z_high)

    def get_ui_config(self):
        objects = [
            {
                "idx": 0,
                "label": "Santa's Clothes",
                "directions": ["left", "none", "right"],
                "default_direction": "none",
                "default_strength": 1.0,
                "max_strength": 2.0,
            },
        ]
        for obj in objects:
            if obj["idx"] < len(self._object_masks_b64):
                obj["mask_b64"] = self._object_masks_b64[obj["idx"]]
        return {"num_objects": len(objects), "objects": objects}

    def configure_simulation(self, simulator):
        """Pre-compute wind parameters from stored forces (any thread)."""
        for f in self._forces:
            direction = np.array(f["direction"], dtype=np.float32)
            strength = f["strength"]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                self._wind_direction = np.zeros(3, dtype=np.float32)
                self._wind_strength = 0.0
                continue
            self._wind_direction = direction / norm
            self._wind_strength = strength * self.force_scale

        if self._wind_bounds is None and len(simulator.all_obj_info) > 0:
            info = simulator.all_obj_info[0]
            z_min = float(info["min"][2])
            z_max = float(info["max"][2])
            z_range = z_max - z_min
            self._wind_bounds = (
                z_min + z_range * 0.05,
                z_min + z_range * 0.8,
            )

    def apply_forces(self, simulator, step_count):
        """Apply wind to PBD cloth by modifying particle velocities."""
        if self._wind_strength < 1e-6:
            return
        if self._wind_bounds is None:
            return

        wind_lowest, wind_highest = self._wind_bounds
        dt = simulator.dt

        for obj_idx, obj in enumerate(simulator.objs):
            mt = simulator.material_type[obj_idx] if obj_idx < len(simulator.material_type) else "rigid"
            if mt not in ("pbd_cloth", "pbd_elastic", "pbd_particle"):
                continue

            solver = obj.solver
            state = solver.get_state(0)
            if state is None:
                continue

            p_start = obj.particle_start
            n_p = obj.n_particles

            z = state.pos[0, p_start:p_start + n_p, 2]
            is_free = state.free[0, p_start:p_start + n_p].bool()
            in_zone = (z > wind_lowest) & (z < wind_highest)
            mask = is_free & in_zone
            if not mask.any():
                continue

            t = torch.zeros_like(z)
            t[mask] = (z[mask] - wind_lowest) / (wind_highest - wind_lowest)
            scaler = torch.zeros_like(z)
            scaler[mask] = torch.exp(t[mask] ** 2)

            wind_dir = torch.tensor(
                self._wind_direction, dtype=z.dtype, device=z.device,
            )

            wind_delta = wind_dir.unsqueeze(0) * (self._wind_strength * scaler.unsqueeze(1) * dt)
            state.vel[0, p_start:p_start + n_p, :] += wind_delta

            solver.set_state(0, state)
