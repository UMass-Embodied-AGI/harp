"""Controller-level contracts for the composed bread/basket family.

The scene-free characterization tests freeze the public task surface; these
tests pin the controller side of the phase-3 migration: ownership, strict
state round-trips, adoption of release-v1 task projections, and the absence
of task references inside the controllers.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from envs.interaction import (
    InteractionMode,
    PlaceBreadInBasketAssistController,
    PlaceBreadInBasketInterruptController,
)
from envs.interaction.inspect_motion import InspectMotionInterruptController
from envs.tasks.place_bread_in_basket_contribution import PlaceBreadInBasketContribution
from envs.tasks.place_bread_in_basket_interruption import PlaceBreadInBasketInterruption


class _FakeEntity:
    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_pos(self):
        return self._pos.copy()

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])


class _FakeActor:
    def __init__(self, pos):
        self.entity = _FakeEntity(pos)


def test_assist_task_composes_family_controller():
    task = PlaceBreadInBasketContribution({})
    controller = task.interaction_controller
    assert type(controller) is PlaceBreadInBasketAssistController
    assert controller.mode is InteractionMode.ASSIST
    assert controller.handles_policy_ticks is True
    assert controller.handles_scripted_rollout_ticks is False
    assert controller.start_delay_steps == task.AVATAR_START_DELAY_STEPS
    assert controller.rng_namespace == "place_bread_in_basket.assist.v1"


def test_interrupt_task_composes_extended_inspect_controller():
    task = PlaceBreadInBasketInterruption({})
    controller = task.interaction_controller
    assert type(controller) is PlaceBreadInBasketInterruptController
    assert isinstance(controller, InspectMotionInterruptController)
    assert controller.mode is InteractionMode.INTERRUPT
    assert controller.handles_policy_ticks is False
    assert controller.handles_scripted_rollout_ticks is False
    assert controller.region_x is not None and controller.region_y is not None
    assert "reset" not in PlaceBreadInBasketInterruption.__dict__


def test_assist_plan_publishes_release_v1_allocation_attributes():
    task = PlaceBreadInBasketContribution({"hard": True})
    controller = task.interaction_controller
    positions = controller.plan_positions(task, np.random.RandomState(3))
    assert len(positions) == 1 + task._num_breads
    assert task._avatar_bread_idx == controller.avatar_bread_idx
    assert task._robot_bread_indices == list(controller.robot_bread_indices)
    assert task._robot_bread_idx == controller.robot_bread_indices[0]
    assert task._avatar_bread_idx not in task._robot_bread_indices
    assert set(task._robot_bread_indices) | {task._avatar_bread_idx} == set(
        range(task._num_breads)
    )


def test_assist_scripted_routing_adopts_task_written_allocation():
    task = PlaceBreadInBasketContribution({"hard": True})
    task.basket = _FakeActor([0.0, 0.0, 0.75])
    task.breads = [
        _FakeActor([0.30, 0.0, 0.79]),
        _FakeActor([-0.30, 0.0, 0.79]),
        _FakeActor([0.10, 0.0, 0.79]),
    ]
    # Scene-init replay and legacy tooling write the task attributes directly.
    task._avatar_bread_idx = 1
    task._robot_bread_indices = [0, 2]
    controller = task.interaction_controller
    assert controller.scripted_robot_indices(task) == [2, 0]
    assert controller.avatar_bread_idx == 1
    assert controller.robot_bread_indices == (0, 2)

    # on_base_reset_ready performs the same adoption for scene-init replay.
    fresh = PlaceBreadInBasketContribution({"hard": True})
    fresh._avatar_bread_idx = 2
    fresh._robot_bread_idx = 0
    fresh.interaction_controller.on_base_reset_ready(fresh)
    assert fresh.interaction_controller.avatar_bread_idx == 2
    assert fresh.interaction_controller.robot_bread_indices == (0,)
    assert fresh._robot_bread_indices == [0]


def test_assist_eval_step_fires_once_and_publishes_state():
    task = PlaceBreadInBasketContribution({"eval_mode": True})
    controller = task.interaction_controller
    starts = []
    task._start_avatar_assist = lambda add_start_delay=True: starts.append(
        add_start_delay
    )
    # The task-facing trigger attribute is authoritative (legacy read it with
    # an EVAL_TRIGGER_STEP_MIN default), so arm it the way eval reset would.
    task._eval_assist_trigger_step = 3
    controller.eval_step(task, 2)
    assert starts == []
    assert task._eval_assist_fired is False
    controller.eval_step(task, 3)
    assert starts == [False]
    assert task._eval_assist_fired is True
    assert task._eval_assist_failed is False
    controller.eval_step(task, 4)
    assert starts == [False]


def test_assist_eval_reset_records_failure_without_raising():
    task = PlaceBreadInBasketContribution({"eval_mode": True})

    def failing(add_start_delay=True):
        raise RuntimeError("no avatar")

    task._start_avatar_assist = failing
    task.interaction_controller.eval_reset(task)
    assert task._eval_assist_fired is True
    assert task._eval_assist_failed is True
    assert task._eval_assist_trigger_step == 0


def test_assist_state_round_trip_is_complete_and_strict():
    task = PlaceBreadInBasketContribution({"hard": True})
    controller = task.interaction_controller
    controller.plan_positions(task, np.random.RandomState(11))
    controller.eval_assist_trigger_step = 4
    controller.eval_assist_fired = True
    controller.policy_step_count = 9
    state = controller.state_dict()

    restored_task = PlaceBreadInBasketContribution({"hard": True})
    restored = restored_task.interaction_controller
    restored.restore_task_state(restored_task, state)
    assert restored.state_dict() == state
    assert restored_task._avatar_bread_idx == controller.avatar_bread_idx
    assert restored_task._robot_bread_indices == list(controller.robot_bread_indices)
    assert restored_task._eval_assist_fired is True
    assert restored_task._eval_policy_step_count == 9
    assert restored_task._interaction_policy_step == 9

    with pytest.raises(ValueError):
        restored.load_state_dict({**state, "unexpected": 1})
    with pytest.raises(ValueError):
        restored.load_state_dict({k: v for k, v in state.items() if k != "robot_bread_indices"})
    with pytest.raises(ValueError):
        restored.load_state_dict({**state, "schema_version": "other"})
    with pytest.raises(ValueError):
        restored.load_state_dict({
            **state,
            "robot_bread_indices": [state["avatar_bread_idx"]],
        })
    with pytest.raises(TypeError):
        restored.load_state_dict({**state, "eval_assist_fired": 1})


def test_interrupt_plan_publishes_ordinal_and_interrupted_index():
    task = PlaceBreadInBasketInterruption({"hard": True, "interrupt_pick_ordinal": 1})
    controller = task.interaction_controller
    positions = controller.plan_positions(task, np.random.RandomState(5))
    assert len(positions) == 1 + task._num_breads
    assert task._interrupted_pick_ordinal == 1
    order = controller.scripted_pick_order_from_positions(positions[0], positions[1:])
    assert task._interrupted_idx == order[1]
    assert controller.interrupted_idx == task._interrupted_idx


def test_interrupt_reset_ready_adopts_task_projections_without_clobbering():
    task = PlaceBreadInBasketInterruption({})
    controller = task.interaction_controller
    controller.on_episode_seeded(task, 0)
    task._interrupted_idx = 1
    task._interrupted_pick_ordinal = 0
    controller.on_base_reset_ready(task)
    assert controller.interrupted_idx == 1
    assert controller.interrupted_pick_ordinal == 0
    # A fresh seed clears controller state but must not publish None back.
    controller.on_episode_seeded(task, 1)
    controller._publish_episode_state(task)
    assert task._interrupted_idx == 1


def _eval_interrupt_task(ee_xyz):
    """Interrupt task with scene-free fakes; ``ee_xyz`` is a mutable list."""
    task = PlaceBreadInBasketInterruption({"eval_mode": True})
    task.TABLE_TOP_Z = 0.75
    task._basket_rim_z = 0.82
    task.basket = _FakeActor([0.0, 0.0, 0.75])
    task.breads = [_FakeActor([0.10, 0.0, 0.79]), _FakeActor([0.30, 0.0, 0.79])]
    task._bread_center_mesh = np.zeros(3)
    task._bread_scale = 1.0
    resets = []
    plays = []

    class _AvatarRobot:
        def __init__(self):
            self.global_trans = np.asarray(task.avatar_init_pos, dtype=float).copy()

        def get_palm_center(self, hand_id):
            return self.global_trans + np.array([0.0, 0.0, 0.5])

    class _Avatar:
        def __init__(self):
            self.robot = _AvatarRobot()
            self.motion_modules = {}

        def reset(self, pos, rot):
            self.robot.global_trans = np.asarray(pos, dtype=float).copy()
            resets.append(self.robot.global_trans.copy())

        def spare(self):
            return False

    class _Arm:
        gripper_val = 1.0

        def get_ee_pose(self):
            return list(ee_xyz) + [1.0, 0.0, 0.0, 0.0]

    class _Robot:
        def get_arm(self, tag):
            return _Arm()

    task.avatar = _Avatar()
    task.robot = _Robot()
    task._dryrun_capture_palm = lambda: np.array([0.0, 0.5, 0.9])
    task._play_inspect_motion = lambda **kwargs: plays.append(kwargs)
    return task, resets, plays


def test_interrupt_eval_progress_trigger_glide_then_inspect_with_target_attached():
    # The trigger is the shared progress trigger (no random step,
    # no force distance); the glide starts from the avatar's current root,
    # lasts INTERRUPT_GLIDE_POLICY_STEPS (default 30) and the clip returns
    # the human to idle instead of freezing.
    ee = [-0.30, -0.40, 1.10]
    task, resets, plays = _eval_interrupt_task(ee)
    controller = task.interaction_controller

    controller.init_eval_state(task)
    assert task._eval_interrupt_fired is False
    for _ in range(25):  # robot hovers far away: no trigger
        controller.advance_eval_step(task)
    assert task._eval_interrupt_fired is False
    assert task._policy_step_count == 25

    ee[:] = [0.10, -0.05, 0.95]  # commits to loaf 0
    controller.advance_eval_step(task)
    assert task._eval_interrupt_fired is True
    assert task._eval_target_bread is task.breads[0]
    assert task._interrupt.trigger.trigger_target == 0
    assert resets == []

    glide = task._interrupt.human.glide_steps
    for _ in range(glide - 1):
        controller.advance_eval_step(task)
    assert plays == []
    controller.advance_eval_step(task)
    assert len(resets) == glide and len(plays) == 1
    assert plays[0]["attach_obj"] is task.breads[0].entity
    assert plays[0]["hand_id"] == task.AVATAR_HAND_ID
    assert plays[0]["return_to_idle_after"] == task._interrupt.human.idle_return_frames
    steps = np.linalg.norm(np.diff(np.asarray(resets), axis=0), axis=1)
    total = np.linalg.norm(resets[-1] - np.asarray(task.avatar_init_pos))
    assert steps.max() <= total / glide + 1e-9  # glide, no teleport
    np.testing.assert_allclose(
        resets[-1],
        np.asarray(task.avatar_init_pos) + (np.array([0.10, 0.0, 0.79]) - np.array([0.0, 0.5, 0.9])) + [0, 0, task.AVATAR_EXTRA_Z],
    )


def test_interrupt_state_round_trip_is_complete_and_strict():
    # Schema v2 serializes the progress trigger and the human
    # sequence instead of the retired glide/pace fields.
    ee = [-0.30, -0.40, 1.10]
    task, _resets, _plays = _eval_interrupt_task(ee)
    controller = task.interaction_controller
    controller.interrupted_pick_ordinal = 1
    controller.init_eval_state(task)
    ee[:] = [0.10, -0.05, 0.95]
    for _ in range(21):
        controller.advance_eval_step(task)
    assert task._eval_interrupt_fired is True
    controller.advance_eval_step(task)  # mid-glide
    state = controller.state_dict()
    assert state["schema_version"].endswith(".v2")

    restored_task, _r, restored_plays = _eval_interrupt_task(ee)
    restored = restored_task.interaction_controller
    restored.restore_task_state(restored_task, state)
    assert restored.state_dict() == state
    assert restored_task._interrupted_idx == 0
    assert restored_task._eval_target_bread is restored_task.breads[0]
    assert restored_task._policy_step_count == 22
    assert restored_task._interrupt.fired is True
    assert restored_task.evaluate.__func__ is PlaceBreadInBasketInterruption.evaluate
    # The rebound clip start fires after the remaining glide steps.
    for _ in range(restored_task._interrupt.human.glide_steps):
        restored.advance_eval_step(restored_task)
    assert len(restored_plays) == 1

    with pytest.raises(ValueError):
        restored.load_state_dict({**state, "extra": 0})
    with pytest.raises(ValueError):
        restored.load_state_dict({**state, "human": None})
    with pytest.raises(TypeError):
        restored.load_state_dict({**state, "eval_interrupt_fired": "yes"})
    with pytest.raises(ValueError):
        restored.load_state_dict({**state, "schema_version": "v0"})


@pytest.mark.parametrize(
    "task_cls",
    (PlaceBreadInBasketContribution, PlaceBreadInBasketInterruption),
)
def test_controllers_do_not_retain_owning_task(task_cls):
    task = task_cls({"hard": True})
    controller = task.interaction_controller
    controller.on_episode_seeded(task, 0)
    controller.plan_positions(task, np.random.RandomState(0))
    controller.on_base_reset_ready(task)
    ref = weakref.ref(task)
    del task
    gc.collect()
    assert ref() is None
    assert controller.state_dict()["schema_version"] == controller.STATE_SCHEMA
