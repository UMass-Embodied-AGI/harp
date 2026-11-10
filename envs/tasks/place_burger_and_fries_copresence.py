"""Co-presence variant of ``place_burger_fries`` (``place_burger_and_fries_copresence``)."""

from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)
from ..task_bases.place_burger_fries import PlaceBurgerFries


class PlaceBurgerAndFriesCopresence(NeutralAvatarTableWorkMixin, PlaceBurgerFries):
    """Robot places food on the tray while the human performs unrelated table work."""

    # Plan around the human's objects as exact boxes.
    NEUTRAL_OBJECT_PLANNER_BOXES = True
    # Turn the hand 90 deg for the round patty when the default
    # hand footprint would graze the human's bottle/mouse box.
    BURGER_PROP_AWARE_YAW = True

    NEUTRAL_AVATAR_MOTION_CHOICES = (
        "wipe_table_1",
        "wipe_table_2",
        "use_mouse",
        "open_bottle",
    )

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # Tray is left/back and foods spawn near center-right; use the far
        # right/back corner of the default table.
        work_xy=(0.42, 0.12),
        mouse_work_xy=(0.42, 0.16),
        table_xy_bounds=(-0.50, 0.50, -0.35, 0.35),
        max_table_edge_gap=0.67,
        # The full-state expert must see the human: planning with the avatar
        # hidden made the privileged baseline as blind as the no-intent one.
        avatar_as_planner_obstacle=True,
    )

    INSTRUCTION = (
        "place the meal items on the tray while the human works nearby on the "
        "same table"
    )
