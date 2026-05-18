from simulation.case_simulation.case_handler import CaseHandler, register_case
import math
import numpy as np
import torch


@register_case("block_stack_pull")
class BlockStackPull(CaseHandler):
    """Rigid block stack case: pull the second block from the bottom.

    The visual object masks and reconstructed meshes still come from the
    RealWonder SAM2/SAM3D path. Genesis receives one rigid body per block; the
    custom case logic only applies a prescribed motion to the target block so
    the remaining blocks respond through rigid contact, gravity, and friction.
    """

    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self._initial_pos = []
        self._initial_quat = []
        self._reported = False

    def init_robots_pose(self):
        self._initial_pos = []
        self._initial_quat = []
        for obj in self.all_objs:
            self._initial_pos.append(obj.get_pos().detach().clone())
            self._initial_quat.append(obj.get_quat().detach().clone())

    def custom_simulation(self, sid):
        if not self._initial_pos:
            return

        target_idx = int(self.config.get("pull_object_index", 2))
        if target_idx < 0 or target_idx >= len(self.all_objs):
            return

        start_step = int(self.config.get("pull_start_step", 16))
        pull_steps = max(1, int(self.config.get("pull_steps", 70)))
        release_step = start_step + pull_steps
        settle_hold_steps = int(self.config.get("pull_hold_steps", 5))

        if sid < start_step or sid > release_step + settle_hold_steps:
            return

        target_obj = self.all_objs[target_idx]
        target_info = self.all_obj_info[target_idx]
        target_size = target_info["size"].detach().cpu().numpy()

        direction = np.array(self.config.get("pull_direction_gs", [1.0, 0.0, 0.0]), dtype=np.float32)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-8:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            direction_norm = 1.0
        direction = direction / direction_norm

        distance_scale = float(self.config.get("pull_distance_scale", 1.65))
        min_distance = float(self.config.get("pull_min_distance", 0.16))
        pull_axis = int(np.argmax(np.abs(direction)))
        distance = max(min_distance, float(target_size[pull_axis]) * distance_scale)

        phase = min(1.0, max(0.0, (sid - start_step) / pull_steps))
        smooth = 0.5 - 0.5 * math.cos(math.pi * phase)

        displacement = torch.tensor(
            direction * distance * smooth,
            device=self.device,
            dtype=self._initial_pos[target_idx].dtype,
        )
        target_pos = self._initial_pos[target_idx] + displacement

        if sid == start_step and not self._reported:
            print(
                "[block_stack_pull] target object "
                f"{target_idx}, size={target_size.tolist()}, "
                f"direction={direction.tolist()}, distance={distance:.4f}, "
                f"steps={pull_steps}"
            )
            self._reported = True

        # Keep the pulled block upright while a virtual gripper extracts it.
        # After the hold window ends, the block is released to regular physics.
        target_obj.set_pos(target_pos, zero_velocity=True)
        if self.config.get("hold_target_quat", True):
            target_obj.set_quat(self._initial_quat[target_idx], zero_velocity=True)
