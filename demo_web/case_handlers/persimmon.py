"""Persimmon demo case handler — 3 rigid objects, force for first 5 steps only."""

import numpy as np

from case_handlers.base import DemoCaseHandler, register_demo_case


@register_demo_case("persimmon")
class PersimmonDemoHandler(DemoCaseHandler):

    # Per-object force multiplier: top persimmon is lighter so needs less
    # force to move the same distance.  [top, middle, bottom]
    force_scale = [50.0, 200.0, 100.0]

    def get_ui_config(self):
        objects = [
            {
                "idx": 0,
                "label": "Top Persimmon",
                "directions": ["left", "none", "right"],
                "default_direction": "none",
                "default_strength": 1.0,
                "max_strength": 2.0,
            },
            {
                "idx": 1,
                "label": "Middle Persimmon",
                "directions": ["left", "none", "right"],
                "default_direction": "none",
                "default_strength": 1.0,
                "max_strength": 2.0,
            },
            {
                "idx": 2,
                "label": "Bottom Persimmon",
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

    def apply_forces(self, simulator, step_count):
        """Only apply forces for the first 5 simulation steps (matching offline persimmon.py)."""
        if step_count > 5:
            return
        super().apply_forces(simulator, step_count)
