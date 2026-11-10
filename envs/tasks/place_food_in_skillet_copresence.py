"""Co-presence variant of ``place_food_in_skillet`` (``place_food_in_skillet_copresence``)."""

import dataclasses

from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)
from ..task_bases.place_food_in_skillet import SkilletTaskCore


class PlaceFoodInSkilletCopresence(NeutralAvatarTableWorkMixin, SkilletTaskCore):
    """Robot places food in the skillet while the human performs unrelated table work."""

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        # Co-presence data collection is a single-food task, but the food identity
        # is sampled per seeded episode from SkilletTaskCore.RANDOM_FOOD_POOL.
        # Callers can still pin it with foods: [name] or disable sampling with
        # random_object: false for deterministic diagnostics.
        cfg.setdefault("random_object", True)
        super().__init__(cfg)

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # Kitchen table is wide; keep neutral work away from the cooktop
        # centered at (-0.12,0.08), the skillet start, and food starts.
        work_xy=(0.58, 0.26),
        mouse_work_xy=(0.58, 0.32),
        table_xy_bounds=(-0.80, 0.80, -0.45, 0.55),
        max_table_edge_gap=0.67,
    )

    INSTRUCTION = (
        "place the food in the skillet while the human works nearby on the "
        "same table"
    )

    # The round apple is picked with the family's top-down primitive (the
    # route the burger and bread use) instead of the one-shot side grasp.
    # Native MPLib probes show every IK solution of the side-grasp pre and
    # grasp poses intersects the counter top and the cooktop slab (the hand
    # body spans +-3.2 cm about a TCP only 2.5 cm above the counter), so the
    # bare pre-grasp RRT sweeps low through the cooktop and execution is
    # contact-blocked 7-12 cm short on every apple draw; top-down goals at
    # the apple centre are collision-free against counter, cooktop and pan.
    # Only the grasp route changes: spawn, friction, close value, the food
    # sampling and its RNG use, and the lift/transit/place code are not
    # affected.  Sibling variants keep their own specs.
    NEUTRAL_APPLE_TOP_DOWN = True
    # Apple close value.  The spec's 0.30 close lifts the apple but loses it
    # as the transit ends over the pan rim: with the family's 80 N finger cap
    # a 0.30 command on a 5.4 cm apple saturates the squeeze, and the base
    # path then re-grips to 0.20 after the lift; a round object squeezed at
    # the cap pops out (the crush-eject mode also measured on this family's
    # burger, which lifts reliably at 0.58-0.62).  0.62 (4.96 cm cage,
    # 1.7 mm per side on the 5.31 cm hull) does not lift the apple under the
    # soft mu-4 contact.  0.50 commands a 4.0 cm cage, about 6.5 mm per
    # side, roughly 59 N nominal: firm but below the cap.  No post-lift
    # re-grip.  Burger/bread specs are not affected.
    NEUTRAL_APPLE_CLOSE_VALUE = 0.50
    APPLE_POST_LIFT_CLOSE_VALUE = None

    def _enabled_food_specs(self):
        specs = super()._enabled_food_specs()
        if not self.NEUTRAL_APPLE_TOP_DOWN:
            return specs
        out = []
        for spec in specs:
            if spec.name == "apple":
                changes = {}
                if spec.side_grasp:
                    changes["side_grasp"] = False
                if self.NEUTRAL_APPLE_CLOSE_VALUE is not None:
                    changes["close_value"] = float(self.NEUTRAL_APPLE_CLOSE_VALUE)
                spec = dataclasses.replace(spec, **changes) if changes else spec
            out.append(spec)
        return out
