"""Interactive UI handler for the T-shirt folding case."""

from case_handlers.base import DemoCaseHandler, register_demo_case


@register_demo_case("tshirt_folding")
class TShirtFoldingDemoHandler(DemoCaseHandler):
    def get_ui_config(self):
        objects = [
            {
                "idx": 0,
                "label": "T-shirt",
                "directions": ["none"],
                "default_direction": "none",
                "default_strength": 0.0,
                "max_strength": 0.0,
            }
        ]
        if self._object_masks_b64:
            objects[0]["mask_b64"] = self._object_masks_b64[0]
        return {"num_objects": 1, "objects": objects}

    def configure_simulation(self, simulator):
        return None

    def apply_forces(self, simulator, step_count):
        return None
