from __future__ import annotations

import numpy as np
import genesis as gs

from simulation.case_simulation.case_handler import CaseHandler, register_case


@register_case("hourglass_sand_flow")
class HourglassSandFlow(CaseHandler):
    """MPM sand inside a fixed procedural hourglass collision proxy.

    The visible sand is reconstructed by the normal RealWonder SAM2/SAM3D path.
    The transparent glass shell is kept as a fixed Genesis collision proxy because
    single-view reconstruction does not provide a physically usable hollow glass
    volume.
    """

    def detect_ground_plane(self, ground_plane):
        if self.config.get("hourglass_enable_proxy", True):
            # A default plane at the upper-sand lower bound would block the falling
            # stream. The hourglass proxy supplies the physical bottom instead.
            return None

        self.ground_anchor = self.all_obj_occupied_lower_bound.cpu().numpy()
        self.ground_anchor[2] -= float(self.config.get("hourglass_receiver_plane_drop", 0.34))
        self.normal = np.array([0, 0, 1])
        self.scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True,
                rho=1000.0,
                friction=float(self.config.get("plane_friction", 4.0)),
                coup_friction=float(self.config.get("plane_coup_friction", 4.0)),
                coup_softness=float(self.config.get("plane_coup_softness", 0.002)),
            ),
            morph=gs.morphs.Plane(pos=tuple(self.ground_anchor), normal=self.normal),
        )
        print(
            "[hourglass_sand_flow] collision proxy disabled; "
            f"using receiver plane at z={self.ground_anchor[2]:.4f}"
        )

    def custom_setup(self):
        if not self.config.get("hourglass_enable_proxy", True):
            self.proxy_boxes = []
            return None

        sand = self.all_obj_info[0]
        center = sand["center"].detach().cpu().numpy().astype(np.float64)
        size = sand["size"].detach().cpu().numpy().astype(np.float64)
        size = np.maximum(size, np.array([0.10, 0.04, 0.10], dtype=np.float64))

        width = max(size[0] * float(self.config.get("hourglass_shell_width_scale", 1.45)), 0.24)
        depth = max(size[1] * float(self.config.get("hourglass_shell_depth_scale", 2.0)), 0.12)
        height = max(size[2] * float(self.config.get("hourglass_shell_height_scale", 4.2)), 0.62)
        wall = max(min(width, depth, height) * float(self.config.get("hourglass_wall_thickness_scale", 0.08)), 0.012)
        neck_gap = max(width * float(self.config.get("hourglass_neck_gap_scale", 0.24)), 0.055)
        bottom_extra = float(self.config.get("hourglass_bottom_extra", 0.10))

        shell_center = center.copy()
        shell_center[2] += size[2] * float(self.config.get("hourglass_shell_center_z_shift", -1.25))

        self.shell_center = shell_center
        self.shell_dims = {
            "width": float(width),
            "depth": float(depth),
            "height": float(height),
            "wall": float(wall),
            "neck_gap": float(neck_gap),
        }

        material = gs.materials.Rigid(
            needs_coup=True,
            rho=1000.0,
            friction=float(self.config.get("hourglass_wall_friction", 4.0)),
            coup_friction=float(self.config.get("hourglass_wall_coup_friction", 4.0)),
            coup_softness=float(self.config.get("hourglass_wall_coup_softness", 0.0015)),
        )
        surface = gs.surfaces.Default(color=(0.70, 0.90, 1.00, 0.25), vis_mode="visual")

        def add_box(name: str, offset, box_size):
            pos = shell_center + np.asarray(offset, dtype=np.float64)
            entity = self.scene.add_entity(
                material=material,
                morph=gs.morphs.Box(
                    pos=tuple(pos),
                    size=tuple(np.asarray(box_size, dtype=np.float64)),
                    fixed=True,
                    collision=True,
                    visualization=False,
                ),
                surface=surface,
            )
            return name, entity, pos.tolist(), list(map(float, box_size))

        z_top = height / 2.0
        z_bottom = -height / 2.0 - bottom_extra
        half_width = width / 2.0
        half_depth = depth / 2.0
        neck_side = max((width - neck_gap) / 2.0, wall)

        self.proxy_boxes = [
            add_box("front_plate", (0, -half_depth, -bottom_extra / 2), (width, wall, height + bottom_extra)),
            add_box("back_plate", (0, half_depth, -bottom_extra / 2), (width, wall, height + bottom_extra)),
            add_box("left_wall", (-half_width, 0, -bottom_extra / 2), (wall, depth, height + bottom_extra)),
            add_box("right_wall", (half_width, 0, -bottom_extra / 2), (wall, depth, height + bottom_extra)),
            add_box("top_cap", (0, 0, z_top), (width, depth, wall)),
            add_box("bottom_cap", (0, 0, z_bottom), (width, depth, wall)),
            add_box("neck_left_baffle", (-(neck_gap + neck_side) / 4.0, 0, 0), (neck_side, depth, wall)),
            add_box("neck_right_baffle", ((neck_gap + neck_side) / 4.0, 0, 0), (neck_side, depth, wall)),
        ]
        print(
            "[hourglass_sand_flow] proxy dims "
            f"width={width:.4f}, depth={depth:.4f}, height={height:.4f}, "
            f"wall={wall:.4f}, neck_gap={neck_gap:.4f}"
        )

    def fix_particles(self):
        self.init_pos = self.all_objs[0].get_particles()[0]

    def custom_simulation(self, sid):
        # Pure gravity-driven granular flow; the case-specific geometry is set up
        # once in custom_setup and Genesis advances the MPM sand state.
        return None
