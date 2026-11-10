"""Generated shoebox collisions must agree with the actual inner visual floor."""
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import trimesh

from envs.task_bases import place_dual_shoes as module


def collision_bounds(path):
    result = {}
    for node in ET.parse(path).findall(".//collision"):
        centre = np.fromstring(node.find("origin").get("xyz"), sep=" ")
        size = np.fromstring(node.find("geometry/box").get("size"), sep=" ")
        result[node.get("name")] = np.array([centre - size / 2, centre + size / 2])
    return result


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "generated"
    monkeypatch.setattr(module, "_SHOEBOX_CACHE_ROOT", root)
    return root


@pytest.mark.parametrize("scale", ([.47, .43, .4], [.6, .7, .5], .43))
def test_collision_floor_matches_actual_inner_mesh_and_preserves_rim(cache, scale):
    path, half, bbox_min = module._ensure_shoebox_urdf(scale)
    bounds = collision_bounds(path)
    assert set(bounds) == {"floor", "left", "right", "front", "back"}
    floor = bounds["floor"]
    assert floor[1, 1] - floor[0, 1] == pytest.approx(.01)
    np.testing.assert_allclose(floor[:, [0, 2]], np.array([bbox_min, bbox_min + 2 * half])[:, [0, 2]])

    scale_vec = np.broadcast_to(scale, (3,))
    mesh = trimesh.load(module.ASSETS_PATH / "objects/007_shoe-box/visual/base0.glb", force="mesh", process=False)
    mesh.apply_scale(scale_vec)
    # Independent measurements across the interior, well clear of the walls.
    for x in (-.1, 0., .1):
        for z in (-.2, 0., .2):
            hits, _, _ = mesh.ray.intersects_location(
                [[x * scale_vec[0], -1., z * scale_vec[2]]], [[0., 1., 0.]], multiple_hits=True)
            levels = hits[:, 1][hits[:, 1] < .02 * scale_vec[1]]
            assert len(levels) > 0
            assert abs(floor[1, 1] - max(levels)) < .00005

    for name in ("left", "right", "front", "back"):
        assert bounds[name][0, 1] == pytest.approx(floor[1, 1])
        assert bounds[name][1, 1] == pytest.approx(bbox_min[1] + 2 * half[1])
    for name, axis in (("left", 0), ("right", 0), ("front", 2), ("back", 2)):
        assert bounds[name][1, axis] - bounds[name][0, axis] == pytest.approx(.01)

    # The visual is the existing asset at the requested scale, not repositioned.
    visual = trimesh.load(path.parent / "shoebox_visual.obj", force="mesh", process=False)
    np.testing.assert_allclose(visual.bounds, mesh.bounds, atol=1e-8, rtol=0)


def test_generated_cache_distinguishes_scale_and_reuses_equivalent_scale(cache):
    first, _, _ = module._ensure_shoebox_urdf(.43)
    contents = first.read_bytes()
    mtime = first.stat().st_mtime_ns
    equivalent, _, _ = module._ensure_shoebox_urdf([.43, .43, .43])
    second, _, _ = module._ensure_shoebox_urdf([.47, .43, .4])
    assert first == equivalent and second != first
    assert first.read_bytes() == contents and first.stat().st_mtime_ns == mtime
    assert not np.array_equal(collision_bounds(first)["floor"], collision_bounds(second)["floor"])


@pytest.mark.parametrize("scale", (0., -1., float("nan"), float("inf"), [1., 0., 1.], [1., float("nan"), 1.]))
def test_invalid_scale_rejected_before_writing_cache(cache, scale):
    with pytest.raises(ValueError, match="finite positive"):
        module._ensure_shoebox_urdf(scale)
    assert not cache.exists()
