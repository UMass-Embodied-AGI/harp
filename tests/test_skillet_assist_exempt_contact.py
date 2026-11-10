"""place_food_in_skillet_assist: human-initiated contact during the human's
pan drop, with the robot holding still, is exempt from the scored collision
but still recorded."""
from __future__ import annotations

import types

import numpy as np

from envs.tasks.place_food_in_skillet_contribution import PlaceFoodInSkilletContribution as T


class _Arm:
    def __init__(self):
        self.q = np.zeros(7)

    def get_arm_qpos(self):
        return self.q.copy()


def _stub(at_frame=600, drop=620):
    arm = _Arm()
    motion = types.SimpleNamespace(at_frame=at_frame, detach_frame=None)
    s = types.SimpleNamespace(
        EXEMPT_DROP_PRE_FRAMES=T.EXEMPT_DROP_PRE_FRAMES,
        EXEMPT_DROP_POST_FRAMES=T.EXEMPT_DROP_POST_FRAMES,
        EXEMPT_ROBOT_STILL_RAD_S=T.EXEMPT_ROBOT_STILL_RAD_S,
        SIM_STEP_IDX=100,
        avatar=types.SimpleNamespace(motion_modules={"pick_and_place": motion}),
        robot=types.SimpleNamespace(get_arm=lambda tag: arm),
        avatar_collided=False,
        _exempt_contacts=[], _exempt_drop_frame=drop, _exempt_prev_arm=None,
        _sim_dt=lambda: 0.002,
    )
    for name in ("_human_pick_place_motion", "_human_in_drop_window",
                 "_robot_still_since_last_check", "_apply_exempt_human_contact",
                 "exempt_human_contact_metrics"):
        setattr(s, name, types.MethodType(getattr(T, name), s))
    return s, arm, motion


def _rec(collided=True):
    return {"collided": collided, "min_distance": -0.004, "sim_step": 0,
            "pairs": [("link7", "r_palm", -0.004)]}


def _check(s, rec, steps=5):
    s.SIM_STEP_IDX += steps
    rec["sim_step"] = s.SIM_STEP_IDX
    if rec["collided"]:
        s.avatar_collided = True
    return s._apply_exempt_human_contact(rec, False)


def test_still_robot_contact_in_drop_window_is_exempt_and_recorded():
    s, _arm, _m = _stub()
    assert not _check(s, _rec(collided=False))     # first check: baseline qpos
    rec = _rec()
    assert _check(s, rec)
    assert rec["collided"] is False and rec["exempt_human_contact"] is True
    assert rec["pairs"] == [] and rec["exempt_pairs"][0][1] == "r_palm"
    assert s.avatar_collided is False
    m = s.exempt_human_contact_metrics()
    assert m["any"] and m["n_checks"] == 1 and m["pairs"] == [("link7", "r_palm")]


def test_moving_robot_is_never_exempt():
    s, arm, _m = _stub()
    _check(s, _rec(collided=False))
    arm.q[3] += 0.01                       # 0.01 rad in 5 sim steps = 1 rad/s
    rec = _rec()
    assert not _check(s, rec)
    assert rec["collided"] is True and s.avatar_collided is True
    assert not s.exempt_human_contact_metrics()["any"]


def test_contact_outside_drop_window_is_not_exempt():
    for frame in (620 - T.EXEMPT_DROP_PRE_FRAMES - 1, 620 + T.EXEMPT_DROP_POST_FRAMES + 1):
        s, _arm, _m = _stub(at_frame=frame)
        _check(s, _rec(collided=False))
        rec = _rec()
        assert not _check(s, rec) and rec["collided"] is True


def test_no_human_drop_scheduled_is_not_exempt():
    s, _arm, _m = _stub(drop=None)
    _check(s, _rec(collided=False))
    assert not _check(s, _rec())


def test_exemption_is_task_scoped():
    from envs.base_task import BaseTask
    assert not hasattr(BaseTask, "_apply_exempt_human_contact")
    assert T._tick_avatar_checks is not BaseTask._tick_avatar_checks
