"""Base demo case handler with registry pattern.

Provides a registry + decorator for per-case UI configuration and
force application logic in the demo_web frontend.
"""

import numpy as np

DEMO_CASE_REGISTRY = {}


def register_demo_case(case_name: str):
    """Decorator to register a DemoCaseHandler subclass."""
    def decorator(cls):
        if case_name in DEMO_CASE_REGISTRY:
            raise ValueError(f"Demo case '{case_name}' already registered!")
        DEMO_CASE_REGISTRY[case_name] = cls
        return cls
    return decorator


class DemoCaseHandler:
    """Base class for per-case UI config and force application in demo_web.

    Subclasses override ``get_ui_config`` and optionally ``apply_forces``
    to customise behaviour for specific cases.
    """

    # Per-object physics force multiplier applied on top of the UI strength
    # slider.  Subclasses override this so the UI always shows a normalised
    # 0-5 range while the actual force magnitude is case-appropriate.
    # Either a single float (applied to all objects) or a list of floats
    # (one per object).
    force_scale = 1.0

    def __init__(self, config):
        self.config = config
        self._forces = []  # list of {"obj_idx", "direction", "strength"}
        self._object_masks_b64 = []  # per-object mask images as base64 PNGs

    @property
    def num_objects(self):
        return len(self.config.get("material_type", []))

    def set_object_masks(self, masks_b64_list):
        """Store base64-encoded mask PNGs for each object."""
        self._object_masks_b64 = list(masks_b64_list) if masks_b64_list else []

    # -- UI configuration --------------------------------------------------

    def get_ui_config(self):
        """Return JSON-serialisable dict describing per-object controls.

        Default: one control per object with left/right/none, strength 1.0.
        Includes mask_b64 for each object if masks were set.
        """
        objects = []
        for idx in range(self.num_objects):
            obj = {
                "idx": idx,
                "label": f"Object {idx}",
                "directions": ["left", "none", "right"],
                "default_direction": "none",
                "default_strength": 1.0,
                "max_strength": 2.0,
            }
            if idx < len(self._object_masks_b64):
                obj["mask_b64"] = self._object_masks_b64[idx]
            objects.append(obj)
        return {"num_objects": self.num_objects, "objects": objects}

    # -- Force management --------------------------------------------------

    def get_force_config_from_ui(self, ui_forces):
        """Map UI force dicts to 3D vectors.

        Args:
            ui_forces: list of ``{"obj_idx", "direction", "strength"}``
                       where direction is either a legacy string
                       ("left"/"right"/"none") or a 3-element list [dx, dy, dz].

        Returns:
            list of ``{"obj_idx", "direction": [dx,dy,dz], "strength"}``.
        """
        legacy_map = {
            "left": [-1.0, 0.0, 0.0],
            "right": [1.0, 0.0, 0.0],
            "none": [0.0, 0.0, 0.0],
        }
        result = []
        for f in ui_forces:
            d = f.get("direction", [0.0, 0.0, 0.0])
            if isinstance(d, str):
                vec = legacy_map.get(d, [0.0, 0.0, 0.0])
            else:
                vec = [float(v) for v in d]
            result.append({
                "obj_idx": int(f.get("obj_idx", 0)),
                "direction": vec,
                "strength": float(f.get("strength", 0.0)),
            })
        return result

    def set_forces(self, forces):
        """Store resolved force configs (output of ``get_force_config_from_ui``)."""
        self._forces = list(forces)

    def configure_simulation(self, simulator):
        """Called from the main thread before the generation loop starts.

        Override in subclasses that need to set simulation state requiring
        the main thread's CUDA context (e.g. taichi field writes).
        """
        pass

    def reset_forces(self):
        self._forces = []

    def apply_forces(self, simulator, step_count):
        """Apply stored forces to the simulator's objects.

        Default behaviour: apply a constant force every step to each rigid
        object that has a non-zero direction.
        """
        for f in self._forces:
            obj_idx = f["obj_idx"]
            direction = np.array(f["direction"], dtype=np.float32)
            strength = f["strength"]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            direction = direction / norm
            if isinstance(self.force_scale, (list, tuple)):
                scale = self.force_scale[obj_idx] if obj_idx < len(self.force_scale) else 1.0
            else:
                scale = self.force_scale
            force_magnitude = strength * scale
            mt = simulator.material_type[obj_idx] if obj_idx < len(simulator.material_type) else "rigid"
            if mt == "rigid":
                simulator.objs[obj_idx].solver.apply_links_external_force(
                    force=(direction * force_magnitude).reshape(1, 3),
                    links_idx=[simulator.objs[obj_idx].idx],
                )


def get_demo_case_handler(case_name, config):
    """Factory: return a handler for *case_name*, falling back to default."""
    cls = DEMO_CASE_REGISTRY.get(case_name, DemoCaseHandler)
    return cls(config)
