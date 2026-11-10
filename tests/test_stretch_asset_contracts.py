from pathlib import Path
from xml.etree import ElementTree

import pytest

from envs.robot.stretch_robot import (
    STRETCH_ARM_JOINTS,
    STRETCH_BUNDLED_URDF,
    STRETCH_EE_LINK,
    STRETCH_FINGER_JOINTS,
    STRETCH_URDF_REL,
    STRETCH_WHEEL_JOINTS,
    resolve_stretch_urdf,
)


def _write_urdf(path: Path, mesh_filename: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = ""
    if mesh_filename is not None:
        geometry = (
            "<visual><geometry>"
            f"<mesh filename='{mesh_filename}'/>"
            "</geometry></visual>"
        )
    path.write_text(f"<robot name='test'><link name='base'>{geometry}</link></robot>")
    return path


def test_resolver_prefers_complete_external_description(tmp_path):
    bundled = _write_urdf(tmp_path / "bundled.urdf")
    external = _write_urdf(
        tmp_path / "assets" / STRETCH_URDF_REL,
        "meshes/base.stl",
    )
    mesh = external.parent / "meshes" / "base.stl"
    mesh.parent.mkdir()
    mesh.write_bytes(b"complete fixture")

    assert resolve_stretch_urdf(
        assets_path=tmp_path / "assets",
        bundled_urdf=bundled,
    ) == str(external)


def test_resolver_falls_back_when_external_mesh_bundle_is_partial(tmp_path):
    bundled = _write_urdf(tmp_path / "bundled.urdf")
    _write_urdf(
        tmp_path / "assets" / STRETCH_URDF_REL,
        "meshes/missing.stl",
    )

    assert resolve_stretch_urdf(
        assets_path=tmp_path / "assets",
        bundled_urdf=bundled,
    ) == str(bundled)


def test_resolver_failure_identifies_both_install_locations(tmp_path):
    assets = tmp_path / "assets"
    bundled = tmp_path / "missing-bundled.urdf"
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_stretch_urdf(assets_path=assets, bundled_urdf=bundled)

    message = str(exc_info.value)
    assert str(assets / STRETCH_URDF_REL) in message
    assert str(bundled) in message


def test_bundled_description_preserves_stretch_control_interface():
    root = ElementTree.parse(STRETCH_BUNDLED_URDF).getroot()
    links = {node.attrib["name"] for node in root.findall("link")}
    joint_nodes = {
        node.attrib["name"]: node for node in root.findall("joint")
    }

    required_joints = (
        set(STRETCH_ARM_JOINTS)
        | set(STRETCH_FINGER_JOINTS)
        | set(STRETCH_WHEEL_JOINTS)
    )
    assert required_joints <= joint_nodes.keys()
    assert {"base_link", STRETCH_EE_LINK, "link_gripper_s3_body"} <= links

    for joint in joint_nodes.values():
        assert joint.find("parent").attrib["link"] in links
        assert joint.find("child").attrib["link"] in links

    for name in STRETCH_ARM_JOINTS:
        expected_type = "prismatic" if name.startswith("joint_arm_") or name == "joint_lift" else "revolute"
        assert joint_nodes[name].attrib["type"] == expected_type
        assert joint_nodes[name].find("limit") is not None

    grasp_joint = joint_nodes["joint_grasp_center"]
    assert grasp_joint.find("parent").attrib["link"] == "link_gripper_s3_body"
    assert grasp_joint.find("origin").attrib["xyz"] == "0 0 0.23"

    # The fallback is intentionally self-contained primitive geometry.
    assert not root.findall(".//mesh")
    assert STRETCH_BUNDLED_URDF.with_name("stretch.LICENSE").is_file()
