"""Scene-free contracts for the place_burger_fries_assist shared pool (simple
50-50): two shared burgers; the human takes a random one at reset and goes
for it no matter what; the full-state expert takes the other; the blind
expert commits to the burger nearest its gripper at t=0."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from envs.interaction.place_burger_fries import PlaceBurgerFriesAssistController
from envs.task_bases.place_burger_fries import _HAMBURG_ID
from envs.tasks.place_burger_and_fries_contribution import PlaceBurgerAndFriesContribution
from scripts.eval_expert_set import policy_play_mode


class _Entity:
    def __init__(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_pos(self):
        return self._pos.copy()

    def set_pos(self, pos):
        self._pos = np.asarray(pos, dtype=np.float64)

    def get_quat(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def get_AABB(self):
        return np.array([self._pos - 0.01, self._pos + 0.01])


class _Actor:
    def __init__(self, pos):
        self.entity = _Entity(pos)

    def get_pose(self):
        return SimpleNamespace(p=self.entity.get_pos(), q=self.entity.get_quat())


class _Avatar:
    def __init__(self):
        self.calls = []

    def reset(self, **kwargs):
        self.calls.append(("reset", kwargs))

    def pick_and_place(self, **kwargs):
        self.calls.append(("pick_and_place", kwargs))

    def spare(self):
        return True


TABLE = 0.765
REST = TABLE + 0.03


def _task(seed=0, ee=(0.3, -0.3, 1.1), burgers=((0.05, 0.0), (0.18, -0.05))):
    task = PlaceBurgerAndFriesContribution({})
    task.TABLE_TOP_Z = TABLE
    task._tray_xy = (-0.20, 0.05)
    task.tray_actor = _Actor([-0.20, 0.05, TABLE])
    spec = task._burger_spec()
    task.food_actors = [(_Actor([x, y, REST]), spec) for x, y in burgers]
    task._episode_seed = seed
    task._episode_rngs = {}
    task.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(
        get_ee_pose=lambda: np.array(list(ee) + [1.0, 0.0, 0.0, 0.0]),
        planner=SimpleNamespace(plan_path=lambda *a, **k: None),
    ))
    task.avatar = _Avatar()
    task.avatar_collider = None
    task.step_sim = lambda: None
    task.FINAL_SETTLE_STEPS = 0
    return task


def test_easy_layout_spawns_two_shared_burgers_in_one_region():
    task = PlaceBurgerAndFriesContribution({})
    specs = task._food_specs_for_episode()
    assert [s.asset_id for s in specs] == [_HAMBURG_ID, _HAMBURG_ID]
    region = (task.SHARED_REGION_X, task.SHARED_REGION_Y)
    assert all(s.spawn_xy_range == region for s in specs)
    assert specs[0].model_id == 4 and specs[0].close_value == 0.55
    assert len(PlaceBurgerAndFriesContribution({"hard": True})._food_specs_for_episode()) == 4
    for gone in ("AVATAR_PICKUP_XY", "AVATAR_SPAWN_OFFSCREEN_POS",
                 "POST_FALL_FRAMES", "_protective_stop", "_yield_to_human"):
        assert not hasattr(PlaceBurgerAndFriesContribution, gone)
    assert not hasattr(PlaceBurgerFriesAssistController, "on_base_sim_step") or (
        PlaceBurgerFriesAssistController.on_base_sim_step
        is PlaceBurgerFriesAssistController.__mro__[1].on_base_sim_step
    )


def test_human_target_is_uniform_and_seeded():
    controller = PlaceBurgerFriesAssistController()
    picks = [controller.human_target(_task(seed=s)) for s in range(400)]
    assert set(picks) == {0, 1}
    assert 0.4 < np.mean(np.array(picks) == 0) < 0.6
    assert controller.human_target(_task(seed=7)) == controller.human_target(_task(seed=7))


def test_human_goes_for_its_burger_no_matter_what():
    controller = PlaceBurgerFriesAssistController()
    task = _task(seed=3)
    target = controller.human_target(task)
    # Even if the robot's gripper is right on it.
    x, y = task._food_pos(target)[:2]
    task.robot = SimpleNamespace(get_arm=lambda tag: SimpleNamespace(
        get_ee_pose=lambda: np.array([x, y, REST + 0.05, 1.0, 0, 0, 0]),
    ))
    controller.start_avatar_burger_assist(task)
    kinds = [c[0] for c in task.avatar.calls]
    assert kinds == ["reset", "pick_and_place"]
    kwargs = task.avatar.calls[1][1]
    assert kwargs["attach_obj"] is task.food_actors[target][0]
    assert kwargs["natural"] is True
    assert task._shared_pool["human_target"] == target


def test_human_fires_at_reset_without_delay_opt_in():
    calls = []
    task = _task()
    task._start_avatar_burger_assist = lambda: calls.append(1)
    PlaceBurgerFriesAssistController().eval_reset(task)
    assert calls == [1] and task._eval_assist_trigger_step == 0


def test_blind_commits_to_nearest_burger_regardless_of_human():
    task = _task(ee=(0.20, -0.06, 1.1))
    placed = []
    task._place_foods = lambda idx, arm_tag="right": placed.append(list(idx)) or True
    task._boost_finger_pd = lambda arm: None
    task._human_target_idx = 1
    task._shared_pool = {"human_target": 1, "robot_target": None, "same_target": None}
    assert task.play_blind_once() is True
    assert placed == [[1]]
    task2 = _task(ee=(0.0, 0.0, 1.1))
    task2._place_foods = lambda idx, arm_tag="right": placed.append(list(idx)) or True
    task2._boost_finger_pd = lambda arm: None
    assert task2.play_blind_once() is True
    assert placed[-1] == [0]
    pool = task.interaction_controller.interaction_metrics(task)["shared_pool"]
    assert pool["robot_target"] == 1 and pool["same_target"] is True


def test_blind_grasp_failure_is_an_honest_failure():
    task = _task()
    task._place_foods = lambda idx, arm_tag="right": False
    task._boost_finger_pd = lambda arm: None
    assert task.play_blind_once() is False


def test_full_state_takes_the_other_burger():
    for seed in range(6):
        task = _task(seed=seed)
        placed = []
        task._place_foods = lambda idx, arm_tag="right": placed.append(list(idx)) or True
        task._boost_finger_pd = lambda arm: None
        task.config["eval_mode"] = False
        assert task.play_once() is True
        human = task._shared_pool["human_target"]
        assert placed == [[1 - human]]
        assert task._shared_pool["robot_target"] == 1 - human


def test_success_requires_no_avatar_collision():
    task = _task(burgers=((-0.20, 0.05), (-0.25, 0.08)))
    task.plan_success = True
    task.avatar_collision_checker = object()
    task.avatar_collided = False
    assert task.check_success() is True
    task.avatar_collided = True
    assert task.check_success() is False


def test_blind_expert_routes_to_play_blind_once():
    assert policy_play_mode(
        "place_burger_and_fries_contribution", PlaceBurgerAndFriesContribution, "expert",
    ) == ("play_blind_once", None)
