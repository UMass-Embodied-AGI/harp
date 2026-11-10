import numpy as np

from envs.grasp import GraspPose
from envs.task_bases import dump_bin_milk
from envs.utils import Pose


def test_side_turns_keep_the_original_side_first():
    assert dump_bin_milk.SIDE_TURNS[0] == 0.
    assert len(set(np.round(dump_bin_milk.SIDE_TURNS, 6))) == 4


def test_turned_grasp_keeps_distance_height_and_tilt():
    object_pose = Pose([.3, -.1, .80], [1., 0., 0., 0.])
    center = np.array([.3, -.1, .82])
    tcp = Pose([.35, -.1, .83], [np.cos(.35), 0., np.sin(.35), 0.])
    local = object_pose.inv() * tcp
    grasp = GraspPose('g', local.p, local.q, pre_distance=.05, source='manual', scale_frame='runtime')
    for turn in dump_bin_milk.SIDE_TURNS:
        turned = dump_bin_milk.turn_about_vertical(grasp, object_pose, 1., center, turn).to_world(object_pose, 1.)
        before, after = tcp.to_matrix()[:3, :3], turned.to_matrix()[:3, :3]
        assert np.isclose(np.linalg.norm(turned.p[:2] - center[:2]), np.linalg.norm(tcp.p[:2] - center[:2]))
        assert np.isclose(turned.p[2], tcp.p[2])
        assert np.isclose(after[2, 2], before[2, 2])       # same approach tilt
