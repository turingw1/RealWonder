from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np


@register_case("blocks")
class Blocks(CaseHandler):
    """Four rigid toy blocks: push the red block up-right with a force.

    This intentionally mirrors the Persimmon case style: no prescribed
    kinematic teleporting, just a short external force on the target body.
    Stack contact snapping is handled generically by the simulator through
    config keys so both collision geometry and rendered point clouds move
    together before Genesis builds the scene.
    """

    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self._reported = False

    def custom_simulation(self, sid):
        target_idx = int(self.config.get("push_object_index", 2))
        if target_idx < 0 or target_idx >= len(self.all_objs):
            return None

        start_step = int(self.config.get("push_start_step", 0))
        force_steps = int(self.config.get("push_force_steps", 8))
        if sid < start_step or sid >= start_step + force_steps:
            return None

        direction = np.array(
            self.config.get("push_force_direction_gs", [1.0, 0.0, 1.0]),
            dtype=np.float32,
        )
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            direction = np.array([1.0, 0.0, 1.0], dtype=np.float32)
            norm = np.linalg.norm(direction)
        direction = direction / norm

        strength = float(self.config.get("push_force_strength", 160.0))
        force = (strength * direction).reshape(1, 3)
        self.all_objs[target_idx].solver.apply_links_external_force(
            force=force,
            links_idx=[self.all_objs[target_idx].idx],
        )

        if sid == start_step and not self._reported:
            print(
                "[blocks] applying force to object "
                f"{target_idx}, direction={direction.tolist()}, "
                f"strength={strength:.3f}, steps={force_steps}"
            )
            self._reported = True

        return None
