from pathlib import Path
from xml.etree import ElementTree

import pytest

from envs.robot.xarm7_robot import (
    XARM7_BUNDLED_DESCRIPTION_DIR,
    XARM7_ARM_JOINTS,
    XArm7Arm,
    resolve_xarm7_planner_description,
)


def _touch_pair(root: Path, urdf_rel: str, srdf_rel: str) -> tuple[Path, Path]:
    urdf = root / urdf_rel
    srdf = root / srdf_rel
    urdf.parent.mkdir(parents=True, exist_ok=True)
    srdf.parent.mkdir(parents=True, exist_ok=True)
    urdf.write_text("<robot name='test'/>")
    srdf.write_text("<robot name='test'/>")
    return urdf, srdf


def test_planner_description_prefers_bundled_pair(tmp_path):
    external_urdf, external_srdf = _touch_pair(
        tmp_path,
        "example-robot-data/robots/xarm_description/urdf/xarm7.urdf",
        "example-robot-data/robots/xarm_description/srdf/xarm7.srdf",
    )
    bundled_urdf, bundled_srdf = _touch_pair(
        tmp_path / "bundled",
        "xarm7.urdf",
        "xarm7.srdf",
    )

    resolved = resolve_xarm7_planner_description(
        assets_path=tmp_path,
        bundled_dir=bundled_urdf.parent,
    )
    assert resolved == (str(bundled_urdf), str(bundled_srdf))
    assert resolved != (str(external_urdf), str(external_srdf))


def test_planner_description_falls_back_without_mixing_pairs(tmp_path):
    external_urdf, external_srdf = _touch_pair(
        tmp_path,
        "example-robot-data/robots/xarm_description/urdf/xarm7.urdf",
        "example-robot-data/robots/xarm_description/srdf/xarm7.srdf",
    )
    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir()
    (bundled_dir / "xarm7.urdf").write_text("<robot name='incomplete'/>")

    resolved = resolve_xarm7_planner_description(
        assets_path=tmp_path,
        bundled_dir=bundled_dir,
    )
    assert resolved == (str(external_urdf), str(external_srdf))


def test_planner_description_failure_lists_expected_pairs(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_xarm7_planner_description(
            assets_path=tmp_path / "assets",
            bundled_dir=tmp_path / "bundled",
        )
    message = str(exc_info.value)
    assert "xarm7.urdf" in message
    assert "xarm7.srdf" in message
    assert "example-robot-data" in message


def test_bundled_planner_description_has_complete_arm_chain():
    urdf_path, srdf_path = resolve_xarm7_planner_description(
        assets_path=Path("/definitely/missing/assets"),
        bundled_dir=XARM7_BUNDLED_DESCRIPTION_DIR,
    )
    urdf_root = ElementTree.parse(urdf_path).getroot()
    srdf_root = ElementTree.parse(srdf_path).getroot()

    links = {node.attrib["name"] for node in urdf_root.findall("link")}
    joint_nodes = {
        node.attrib["name"]: node for node in urdf_root.findall("joint")
    }
    joints = set(joint_nodes)
    assert {"world", "link_base", "link7", "link_eef"} <= links
    assert set(XARM7_ARM_JOINTS) <= joints
    assert all(
        joint_nodes[name].attrib["type"] == "revolute"
        for name in XARM7_ARM_JOINTS
    )
    for joint in joint_nodes.values():
        assert joint.find("parent").attrib["link"] in links
        assert joint.find("child").attrib["link"] in links

    for mesh in urdf_root.findall(".//mesh"):
        filename = Path(mesh.attrib["filename"])
        assert not filename.is_absolute()
        assert (Path(urdf_path).parent / filename).resolve().is_file()

    group = srdf_root.find("./group[@name='xarm7']")
    assert group is not None
    srdf_joints = {
        node.attrib["name"] for node in group.findall("joint")
    }
    assert set(XARM7_ARM_JOINTS) <= srdf_joints
    assert srdf_joints <= joints
    for collision_pair in srdf_root.findall("disable_collisions"):
        assert collision_pair.attrib["link1"] in links
        assert collision_pair.attrib["link2"] in links


def test_relative_planner_urdf_is_not_rewritten(tmp_path):
    urdf = tmp_path / "xarm7.urdf"
    urdf.write_text(
        "<robot name='xarm7'><link name='base'><visual><geometry>"
        "<mesh filename='../meshes/base.stl'/></geometry></visual></link></robot>"
    )

    resolved = XArm7Arm._materialize_relative_path_urdf(str(urdf))
    assert resolved == str(urdf)
    assert not (tmp_path / "xarm7_relpaths.urdf").exists()
