"""Interrupt thresholds fitted to the demonstrated intent hover.

Each scripted training demo (hr_v2_nyx37) moves the TCP to an intent hover
above the object and only then does the human start.  The eval-mode progress
trigger must fire on that hover -- not only on the later grasp descent, where
the human would pull the object from a closing gripper.  The geometry below is
the worst case (largest hover distance, lowest progress) measured by replaying
the demos (interrupt_check demo probe, 3-6 demos per task).
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.interaction.interrupt_progress import ProgressTrigger, ProgressTriggerParams
from envs.tasks.categorize_objects_interruption import CategorizeObjectsInterruption
from envs.tasks.dump_bin_interruption import DumpBinInterruption
from envs.tasks.place_bread_in_basket_interruption import PlaceBreadInBasketInterruption
from envs.tasks.place_burger_and_fries_interruption import PlaceBurgerAndFriesInterruption
from envs.tasks.put_object_in_cabinet_interruption import PutObjectInCabinetInterruption
from envs.tasks.stack_three_bowls_interruption import StackThreeBowlsInterruption

OBJ = np.array([0.40, -0.10, 0.78])

# task class, fitted (radius, progress, min_steps, min_moved),
# demo reset distance d0, demo hover distance, TCP travel reset -> hover
CASES = [
    (CategorizeObjectsInterruption, (0.26, 0.25, 20, 0.05), 0.325, 0.218, 0.38),
    (PlaceBurgerAndFriesInterruption, (0.32, 0.0, 10, 0.10), 0.30, 0.296, 0.26),
    (PutObjectInCabinetInterruption, (0.22, 0.30, 20, 0.05), 0.30, 0.179, 0.13),
    (StackThreeBowlsInterruption, (0.24, 0.35, 20, 0.05), 0.30, 0.166, 0.15),
    (DumpBinInterruption, (0.22, 0.40, 20, 0.05), 0.30, 0.152, 0.15),
    (PlaceBreadInBasketInterruption, (0.22, 0.40, 20, 0.05), 0.30, 0.156, 0.15),
]
IDS = [c[0].__name__ for c in CASES]


def _poses(d0, hover_d, travel):
    """Hover straight above the object; reset pose at distance d0 from the
    object and ``travel`` from the hover."""
    cos_t = (d0 ** 2 + hover_d ** 2 - travel ** 2) / (2.0 * d0 * hover_d)
    theta = float(np.arccos(np.clip(cos_t, -1.0, 1.0)))
    reset = OBJ + d0 * np.array([-np.sin(theta), 0.0, np.cos(theta)])
    hover = OBJ + np.array([0.0, 0.0, hover_d])
    return reset, hover


def _params(cls):
    return ProgressTriggerParams.for_task(cls)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_fitted_thresholds_are_the_class_attributes(case):
    cls, (radius, progress, min_steps, min_moved), *_ = case
    p = _params(cls)
    assert p.commit_radius_m == pytest.approx(radius)
    assert p.trigger_progress == pytest.approx(progress)
    assert p.min_policy_steps == min_steps
    assert p.min_moved_m == pytest.approx(min_moved)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_interrupt_trigger_fires_on_the_demonstrated_intent_hover(case):
    cls, (_r, _p, min_steps, _m), d0, hover_d, travel = case
    reset, hover = _poses(d0, hover_d, travel)
    assert np.linalg.norm(reset - OBJ) == pytest.approx(d0)
    assert np.linalg.norm(reset - hover) == pytest.approx(travel)

    trigger = ProgressTrigger(_params(cls))
    trigger.arm(reset, {0: OBJ})
    # A retreat/park pose away from the object never fires.
    retreat = OBJ + np.array([-0.45, 0.0, 0.30])
    for step in range(1, min_steps + 5):
        assert trigger.update(step, retreat, {0: OBJ}) is None
    assert trigger.block_reason == "no_progress_within_commit_radius"
    # The demonstrated hover does.
    assert trigger.update(min_steps + 5, hover, {0: OBJ}) == 0
    assert trigger.trigger_dist_m == pytest.approx(hover_d)

    # The generic contract thresholds (0.22 m / 0.5) miss this hover: the
    # trigger would wait for the grasp descent.
    generic = ProgressTrigger(ProgressTriggerParams())
    generic.arm(reset, {0: OBJ})
    for step in range(1, 60):
        generic.update(step, hover, {0: OBJ})
    assert generic.fired is False


def test_burger_trigger_needs_the_tcp_to_leave_its_reset_pose():
    # Progress 0 means any pose within 0.32 m qualifies, including the reset
    # pose itself (~0.30 m from the food); INTERRUPT_MIN_MOVED_M keeps the
    # human from starting before the robot has moved.
    reset, _hover = _poses(0.30, 0.296, 0.26)
    trigger = ProgressTrigger(_params(PlaceBurgerAndFriesInterruption))
    trigger.arm(reset, {0: OBJ})
    nudged = reset + np.array([0.0, 0.08, 0.0])
    for step in range(1, 40):
        assert trigger.update(step, nudged, {0: OBJ}) is None
    assert trigger.block_reason == "robot_not_moved"
