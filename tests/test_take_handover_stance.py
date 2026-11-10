from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.tasks.take_from_human_easy import TakeFromHumanEasy


# Known open failure (docs/release_readiness.md): kept visible until the
# intended behaviour is resolved and verified natively; strict, so a fix that
# makes it pass must also remove the marker.
@pytest.mark.xfail(strict=True, reason=(
    "known open failure: an unreachable presentation can still trigger avatar "
    "repositioning during execution (docs/release_readiness.md)"))
def test_unreachable_handover_fails_without_moving_the_person():
    task = TakeFromHumanEasy({})
    arm = SimpleNamespace(origin_pose=SimpleNamespace(p=np.array([10., 10., .75])),
                          tcp_offset=np.zeros(3))
    task.robot = SimpleNamespace(get_arm=lambda _: arm)
    task._boost_arm_pd = lambda _: None
    position = np.array([.3, .8, -.05])
    ticks = []
    skin = SimpleNamespace(get_global_translation=lambda name: (
        np.array([0., .1 if name.endswith('Middle1') else 0., 1.]), np.eye(3)))
    robot = SimpleNamespace(skin=skin, global_trans=position,
                            _get_hand_frame=lambda _: (np.zeros(3),np.eye(3)))
    spare = iter([False, True])
    task.avatar = SimpleNamespace(
        robot=robot, attach_object_to_hand=lambda *a,**kw: None,
        play_animation=lambda _: None, spare=lambda: next(spare),
        reset=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('avatar teleported')),
    )
    task.apple_actor = SimpleNamespace(entity=SimpleNamespace(set_pos=lambda _: None))
    task.step_sim = lambda: ticks.append(1)
    assert task.play_once() is False
    np.testing.assert_array_equal(robot.global_trans, position)
    assert ticks == [1]


@pytest.mark.xfail(strict=True, reason=(
    "known open failure: the task no longer applies its previous physical-time "
    "motion setting (docs/release_readiness.md)"))
def test_handover_source_frames_play_at_physical_time(monkeypatch):
    monkeypatch.setattr(BaseTask, '_init_avatar',
                        lambda self: setattr(self, 'avatar', SimpleNamespace(frame_ratio=1.)))
    task=TakeFromHumanEasy({})
    task._init_avatar()
    assert np.isclose(task.avatar.frame_ratio*30*.001, 1.)
    overridden=TakeFromHumanEasy({'handover_motion_slow': 10.})
    overridden._init_avatar()
    assert overridden.avatar.frame_ratio == 10.
