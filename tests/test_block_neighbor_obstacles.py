from types import SimpleNamespace

import numpy as np
import pytest

from envs.task_bases.block_grasp import block_neighbor_obstacles, refresh_block_obstacles


def fixture_task():
    objects = {"unrelated_prop": object()}
    clouds = []
    world = SimpleNamespace(add_object=lambda name, obj: objects.__setitem__(name, obj),
                            remove_object=lambda name: objects.pop(name))
    planner = SimpleNamespace(_mplib=SimpleNamespace(planning_world=world),
                              update_obstacles=lambda points, **kw: clouds.append(points.copy()))
    centers = {"target": np.array([0., 0., .795]), "neighbor": np.array([.1, 0., .795])}
    task = SimpleNamespace(
        robot=SimpleNamespace(get_arm=lambda tag: SimpleNamespace(planner=planner)),
        blocks={name: SimpleNamespace(get_pos=lambda name=name: centers[name],
                                       get_quat=lambda: np.array([1., 0., 0., 0.]))
                for name in centers},
        block_half_size=lambda key: .03, avatar_collider=None,
    )
    return task, objects, clouds, centers


@pytest.mark.parametrize("neutral", [False, True])
def test_neighbor_and_human_obstacles_coexist_and_only_owned_names_are_removed(neutral):
    task, objects, clouds, centers = fixture_task()
    if neutral:
        task._refresh_avatar_planner_obstacles = lambda tag: clouds.append(np.ones((1, 3)))
    with block_neighbor_obstacles(task, "target", "right"):
        refresh_block_obstacles(task, "right")
        assert set(objects) == {"unrelated_prop", "block_ranking_neighbor_neighbor"}
        np.testing.assert_allclose(objects["block_ranking_neighbor_neighbor"].get_pose().p, centers["neighbor"])
        assert len(clouds) == 1
        np.testing.assert_equal(clouds[0], np.ones((1, 3)) if neutral else np.empty((0, 3)))
    assert set(objects) == {"unrelated_prop"}
    assert task._block_neighbor_scene is None


def test_refresh_tracks_live_neighbor_pose_without_touching_entity():
    task, objects, clouds, centers = fixture_task()
    with block_neighbor_obstacles(task, "target", "right"):
        refresh_block_obstacles(task, "right")
        centers["neighbor"] = np.array([.2, -.1, .825])
        refresh_block_obstacles(task, "right")
        np.testing.assert_allclose(objects["block_ranking_neighbor_neighbor"].get_pose().p, centers["neighbor"])
    assert len(clouds) == 2


def test_original_execution_error_survives_cleanup():
    task, objects, _, _ = fixture_task()
    error = ValueError("physical execution failed")
    with pytest.raises(ValueError) as caught:
        with block_neighbor_obstacles(task, "target", "right"):
            refresh_block_obstacles(task, "right")
            raise error
    assert caught.value is error
    assert set(objects) == {"unrelated_prop"}
    assert task._block_neighbor_scene is None


@pytest.mark.parametrize("bad", [[np.nan, 0., .8], [0., .8]])
def test_invalid_geometry_fails_before_world_or_cloud_changes(bad):
    task, objects, clouds, centers = fixture_task()
    centers["neighbor"] = np.array(bad)
    with pytest.raises(RuntimeError, match="invalid block neighbor"):
        with block_neighbor_obstacles(task, "target", "right"):
            refresh_block_obstacles(task, "right")
    assert set(objects) == {"unrelated_prop"} and not clouds


def test_wrong_arm_and_nested_context_cannot_replace_active_scene():
    task, objects, _, _ = fixture_task()
    with block_neighbor_obstacles(task, "target", "right"):
        state = task._block_neighbor_scene
        with pytest.raises(RuntimeError, match="different arm"):
            refresh_block_obstacles(task, "left")
        with pytest.raises(RuntimeError, match="nested"):
            with block_neighbor_obstacles(task, "neighbor", "right"):
                pytest.fail("nested scene entered")
        assert task._block_neighbor_scene is state
    assert set(objects) == {"unrelated_prop"}
