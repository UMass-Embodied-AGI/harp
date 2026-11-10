"""categorize_cooperative shared-pool (assist 50-50) contracts, simulator-free."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.interaction.categorize import CategorizeCooperativeController
from envs.task_bases import categorize as core_module
from envs.tasks.categorize_objects_contribution import CategorizeObjectsContribution
from envs.utils import Pose

from tests.test_categorize_family_characterization import (
    LAYOUT_CASES, _capture_layout, _digest,
)


TABLE = 0.75


class _Entity:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)

    def get_pos(self):
        return self.pos.copy()

    def set_pos(self, pos):
        self.pos = np.asarray(pos, dtype=float)

    def get_AABB(self):
        return np.stack([self.pos - 0.02, self.pos + 0.02])


class _Actor:
    def __init__(self, pos):
        self.entity = _Entity(pos)


class _Motion:
    def __init__(self, attach_obj, hand_id, pick_pos):
        self._attach_obj = attach_obj
        self._hand_id = hand_id
        self.pick_pos = np.asarray(pick_pos, dtype=float)


class _Avatar:
    def __init__(self):
        self.robot = NS(attached_object=[None, None], two_hand_attached_object=None,
                        get_palm_center=lambda h: np.array([0.0, 0.6, 1.1]))
        self.motion_modules = {}
        self.calls = []
        self.busy = False

    def pick_and_place(self, pick_pos, place_pos, attach_obj=None, hand_id=1, **_kw):
        motion = _Motion(attach_obj, hand_id, pick_pos)
        self.motion_modules["pick_and_place"] = motion
        self.calls.append((attach_obj, hand_id))
        self.busy = True
        return motion

    def spare(self):
        return not self.busy


def _shared_task(objects=((-0.2, 0.06), (0.2, 0.06)), tcp=(0.0, -0.05, 1.2)):
    task = CategorizeObjectsContribution({})
    task.TABLE_TOP_Z = TABLE
    task.basket_poses = [Pose((x, 0.24, TABLE + 0.056)) for x in (-0.25, 0.0, 0.25)]
    task._sort_targets = [
        (_Actor((x, y, TABLE + 0.03)), name, 0, b)
        for (x, y), name, b in zip(objects, ("cube", "073_rubikscube"), (0, 2))
    ]
    task._avatar_sort_targets = []
    task._get_object_world_center = lambda actor, _n, _m=0: actor.entity.get_pos()
    task.avatar = _Avatar()
    tcp_state = {"p": np.asarray(tcp, dtype=float)}
    arm = NS(get_ee_pose=lambda: list(tcp_state["p"]) + [1.0, 0.0, 0.0, 0.0])
    task.robot = NS(get_arm=lambda _tag: arm)
    task._tcp = tcp_state
    task._episode_seed = 7
    task._episode_rngs = {}
    return task


def _capture_shared_layout(monkeypatch, seed, config=None):
    task = CategorizeObjectsContribution(dict(config or {}))
    task.scene = object()
    task.TABLE_TOP_Z = TABLE
    monkeypatch.setattr(core_module, "create_primitive", lambda *_a, **_k: object())
    task._spawn_categorize_object = (
        lambda name, model_id, x, y, z, quat: _Actor((x, y, z))
    )
    np.random.seed(seed)
    task.load_actors()
    return task


@pytest.mark.parametrize("seed", range(40))
def test_shared_layout_two_distinct_reliable_objects_in_one_strip(monkeypatch, seed):
    task = _capture_shared_layout(monkeypatch, seed)
    assert len(task.basket_poses) == 2
    assert len(task._sort_targets) == 2
    assert task._avatar_sort_targets == []
    names = [name for _a, name, _m, _b in task._sort_targets]
    baskets = [b for _a, _n, _m, b in task._sort_targets]
    assert len(set(names)) == 2 and len(set(baskets)) == 2
    assert set(names) <= {"cube", "073_rubikscube", "086_woodenblock"}
    xmin, xmax, ymin, ymax = task.SHARED_REGION
    pts = [actor.entity.get_pos()[:2] for actor, *_ in task._sort_targets]
    for x, y in pts:
        assert xmin - 1e-9 <= x <= xmax + 1e-9 and ymin - 1e-9 <= y <= ymax + 1e-9
    assert np.linalg.norm(pts[0] - pts[1]) >= task.SHARED_MIN_SEP - 1e-9
    groups = task._success_target_groups()
    assert [side for side, _ in groups] == ["shared"]
    assert len(groups[0][1]) == 2


def test_shared_layout_varies_which_objects_and_where(monkeypatch):
    combos, xs = set(), []
    for seed in range(30):
        task = _capture_shared_layout(monkeypatch, seed)
        combos.add(tuple(sorted(n for _a, n, _m, _b in task._sort_targets)))
        xs.append(task._sort_targets[0][0].entity.get_pos()[0])
    assert len(combos) == 3
    assert np.ptp(xs) > 0.15


@pytest.mark.parametrize("case", [c for c in LAYOUT_CASES
                                  if c[0] is CategorizeObjectsContribution and c[1].get("no_human")])
def test_no_human_mapping_keeps_the_legacy_layout_and_rng(monkeypatch, case):
    task_cls, config, seed, digest, tail = case
    task, snapshot, actual_tail = _capture_layout(monkeypatch, task_cls, config, seed)
    assert _digest(snapshot) == digest and actual_tail == tail
    assert [side for side, _ in task._success_target_groups()] == ["robot"]


@pytest.mark.parametrize("case", [c for c in LAYOUT_CASES
                                  if c[0] is not CategorizeObjectsContribution])
def test_sibling_layouts_and_rng_unchanged(monkeypatch, case):
    task_cls, config, seed, digest, tail = case
    _task, snapshot, actual_tail = _capture_layout(monkeypatch, task_cls, config, seed)
    assert _digest(snapshot) == digest and actual_tail == tail


def test_human_choice_is_uniform_and_leaves_global_rng_alone():
    np.random.seed(5)
    expected_tail = np.random.random_sample(4)
    picks = []
    for seed in range(200):
        task = _shared_task()
        task._episode_seed = seed
        ctrl = task.interaction_controller
        np.random.seed(5)
        assert ctrl.start_human(task) is True
        assert np.array_equal(np.random.random_sample(4), expected_tail)
        picks.append(ctrl.human_target)
        attach, hand = task.avatar.calls[-1]
        assert attach is task._sort_targets[ctrl.human_target][0]
        assert hand == (0 if ctrl.human_target == 1 else 1)  # closer hand
    assert 0.35 < np.mean(picks) < 0.65


def test_human_goes_for_its_object_no_matter_what():
    from envs.interaction.base import InteractionController
    picks = set()
    for seed in range(20):
        task = _shared_task()
        task._episode_seed = seed
        # Robot gripper down on object 0 and object 1 already in a box:
        # the human does not check availability.
        task._tcp["p"] = np.array([-0.2, 0.06, TABLE + 0.10])
        task._sort_targets[1][0].entity.set_pos((0.25, 0.24, TABLE + 0.03))
        ctrl = task.interaction_controller
        assert ctrl.start_human(task) is True
        picks.add(ctrl.human_target)
        assert len(task.avatar.calls) == 1
    assert picks == {0, 1}
    # No per-step re-check hook.
    assert (CategorizeCooperativeController.on_base_sim_step
            is InteractionController.on_base_sim_step)


def test_human_starts_at_reset_unless_delayed():
    task = _shared_task()
    task.config.update(eval_mode=True)
    task._open_basket_lids = lambda: None
    ctrl = task.interaction_controller
    ctrl.eval_reset(task)
    assert ctrl.human_started and len(task.avatar.calls) == 1
    ctrl.eval_step(task, 1)
    assert len(task.avatar.calls) == 1


def test_blind_rule_is_nearest_to_gripper():
    task = _shared_task(objects=((-0.25, 0.06), (0.10, 0.08)))
    assert task._robot_rule_target() == 1
    task._tcp["p"] = np.array([-0.3, 0.0, 1.2])
    assert task._robot_rule_target() == 0


def _stub_robot(task, order):
    task._boost_arm_pd = lambda: None
    task._open_basket_lids = lambda: None
    task._move_to_safe = lambda _arm: None
    task.step_sim = lambda: None

    def sort(idx, _arm="right", targets=None):
        order.append(idx)
        return True

    task._robot_sort = sort


@pytest.mark.parametrize("human_target", (0, 1))
def test_blind_commits_to_one_object_regardless_of_the_human(human_target):
    task = _shared_task(objects=((-0.25, 0.06), (0.10, 0.08)))
    task.config["eval_mode"] = True
    ctrl = task.interaction_controller
    ctrl.human_started = True
    ctrl._start_reach(task, human_target)
    task.avatar.busy = False
    order = []
    _stub_robot(task, order)
    assert task.play_blind_once() is True
    assert order == [1]
    m = ctrl.shared_pool_metrics(task)
    assert m["robot_target"] == 1 and m["same_target"] is (human_target == 1)


def test_full_state_sorts_only_the_other_object():
    for human in (0, 1):
        task = _shared_task()
        ctrl = task.interaction_controller
        ctrl.human_started = True
        ctrl._start_reach(task, human)
        task.avatar.busy = False
        order = []
        _stub_robot(task, order)
        assert task.play_once() is True
        assert order == [1 - human]
        assert ctrl.shared_pool_metrics(task)["same_target"] is False


def test_shared_metrics_surface():
    task = _shared_task()
    task.config["eval_mode"] = True
    ctrl = task.interaction_controller
    ctrl._start_reach(task, 0)
    ctrl.note_robot_target(0)
    m = ctrl.interaction_metrics(task)
    assert set(m["shared_pool"]) >= {"human_target", "robot_target", "same_target"}
    assert m["shared_pool"]["same_target"] is True
    assert m["eval_avatar_order"] == ["cube"]


def test_success_requires_no_avatar_collision():
    task = _shared_task()
    task.plan_success = True
    task.avatar_collision_checker = object()
    task.avatar_collided = True
    assert task.check_success() is False
