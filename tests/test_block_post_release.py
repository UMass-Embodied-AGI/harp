"""Blocks expert: straight rise after a release, assist yield retreat."""

from types import SimpleNamespace

import numpy as np

from envs.task_bases.blocks_ranking_rgb import BlocksRankingRGB


def _task(ee_z, collided=False):
    calls = []
    task = SimpleNamespace(
        BLOCK_RETREAT_AFTER_RELEASE=True, BLOCK_RETREAT_SETTLE_STEPS=3,
        TABLE_TOP_Z=.765, TRANSPORT_Z_ABOVE_TABLE=.2, avatar_collided=collided,
        _active_block_grasp_plan=object(),
        robot=SimpleNamespace(get_arm=lambda tag: SimpleNamespace(
            get_ee_pose=lambda: np.array([.1, -.08, ee_z, 0., 1., 0., 0.]))))
    task.step_sim = lambda: calls.append("sim")
    task._block_straight_move = lambda pos, tag: calls.append(("rise", tuple(np.round(pos, 3)))) or True
    return task, calls


def test_release_rises_straight_to_transport_height():
    task, calls = _task(.845)
    BlocksRankingRGB._after_pick_place_release(task, None, None, "right", None)
    assert calls == ["sim"] * 3 + [("rise", (.1, -.08, .965))]


def test_release_rise_skipped_when_already_high_or_after_contact():
    for z, collided in ((.96, False), (.845, True)):
        task, calls = _task(z, collided)
        BlocksRankingRGB._after_pick_place_release(task, None, None, "right", None)
        assert calls == ["sim"] * 3


def test_release_rise_needs_an_active_block_pickup():
    task, calls = _task(.845)
    task._active_block_grasp_plan = None
    BlocksRankingRGB._after_pick_place_release(task, None, None, "right", None)
    assert calls == []
