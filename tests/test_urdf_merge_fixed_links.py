"""NYX-safe merge_fixed_links default for URDFs without fixed joints."""

from pathlib import Path

from envs.utils import urdf_merge_fixed_links_kwargs

REPO = Path(__file__).resolve().parents[1]
BOWL_URDF = REPO / "envs/task_bases/stack_bowls_collision/bowl_002_model1/bowl.urdf"


def _write(tmp_path, joints):
    body = "".join(
        f'<joint name="j{i}" type="{kind}"><parent link="base"/><child link="l{i}"/></joint>'
        f'<link name="l{i}"/>'
        for i, kind in enumerate(joints)
    )
    path = tmp_path / "robot.urdf"
    path.write_text(f'<robot name="r"><link name="base"/>{body}</robot>')
    return path


def test_single_body_bowl_disables_merge():
    assert urdf_merge_fixed_links_kwargs(BOWL_URDF) == {"merge_fixed_links": False}


def test_movable_joints_only_disable_merge(tmp_path):
    path = _write(tmp_path, ["revolute", "prismatic"])
    assert urdf_merge_fixed_links_kwargs(path) == {"merge_fixed_links": False}


def test_fixed_joint_keeps_genesis_default(tmp_path):
    path = _write(tmp_path, ["revolute", "fixed"])
    assert urdf_merge_fixed_links_kwargs(path) == {}
