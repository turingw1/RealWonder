"""Lamp demo case handler — single rigid object, constant force."""

from case_handlers.base import DemoCaseHandler, register_demo_case


@register_demo_case("lamp")
class LampDemoHandler(DemoCaseHandler):

    force_scale = 2.5

    def get_ui_config(self):
        objects = [
            {
                "idx": 0,
                "label": "Lamp",
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
