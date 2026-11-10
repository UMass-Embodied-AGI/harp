"""Failure-specific guards for native physical bowl stacking."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from envs.base_task import BaseTask
from envs.planning.base import PlanResult
from envs.task_bases.stack_bowls_motion import _BowlMotionRejected, _pose_matches
from envs.task_bases.stack_bowls_three import StackBowlsThree
from envs.utils import Pose


@pytest.fixture
def native():
    state = {"tcp": Pose([0., 0., .9]), "collision": False, "executed": []}
    backend = SimpleNamespace(check_for_env_collision=lambda _q: ["table"] if state["collision"] else [])
    arm = SimpleNamespace(n_arm=7, tcp_offset=Pose([0, 0, .1034]),
                          get_arm_qpos=lambda: np.zeros(7),
                          get_ee_pose=lambda: state["tcp"].to_pose7(),
                          planner=SimpleNamespace(_mplib=backend, _pad_qpos=lambda q: np.r_[q, 0., 0.]))
    task = StackBowlsThree.__new__(StackBowlsThree)
    task.robot = SimpleNamespace(get_arm=lambda _tag: arm)
    task.plan_success = True
    task._bowl_fk_tcp = lambda *_a: state["tcp"]
    task._bowl_refresh_avatar = lambda _tag: None
    task.execute_plan = lambda plan, _tag: state["executed"].append(plan)
    task.step_sim = lambda: None
    return task, arm, state


@pytest.mark.parametrize("fault", ["nan", "jump", "branch", "velocity", "collision", "fk", "orientation"])
def test_unsafe_native_path_never_executes(native, fault):
    task, arm, state = native
    target = Pose([0., 0., .9])
    pos = np.stack([np.zeros(7), np.full(7, .01)])
    velocity = np.zeros_like(pos)
    if fault == "nan": pos[1, 0] = np.nan
    if fault == "jump": pos[1, 0] = .31
    if fault == "branch": pos = np.linspace(np.zeros(7), np.full(7, 1.4), 20); velocity = np.zeros_like(pos)
    if fault == "velocity": velocity = np.zeros((1, 7))
    if fault == "collision": state["collision"] = True
    if fault == "fk": state["tcp"] = Pose([0, 0, .92])
    if fault == "orientation": state["tcp"] = Pose([0, 0, .9], [0, 1, 0, 0])
    plan = PlanResult(True, pos, velocity)
    arm.planner.plan_screw_path = lambda *_a, **_k: plan
    with pytest.raises(_BowlMotionRejected):
        task._bowl_move(target, "right", "test")
    assert not state["executed"] and not task.plan_success


def test_measured_lag_rejected_after_valid_native_execution(native):
    task, arm, state = native
    target = Pose([0, 0, .9])
    task._bowl_fk_tcp = lambda *_a: target
    state["tcp"] = Pose([0, 0, .87])
    arm.planner.plan_screw_path = lambda *_a, **_k: PlanResult(True, np.zeros((2, 7)), np.zeros((2, 7)))
    with pytest.raises(_BowlMotionRejected, match="arrival"):
        task._bowl_move(target, "right", "lift")
    assert len(state["executed"]) == 1


def test_tiny_lift_has_tighter_fk_gate():
    target = Pose([0, 0, .9])
    actual = Pose([0, 0, .912])
    assert _pose_matches(actual, target)
    assert not _pose_matches(actual, target, tiny=True)


def test_execution_preserves_native_duration_and_gripper_timeline(monkeypatch):
    task = StackBowlsThree.__new__(StackBowlsThree)
    task._bowl_motion_active = True
    task.scene = SimpleNamespace(sim=SimpleNamespace(dt=.002))
    captured = []
    monkeypatch.setattr(BaseTask, "execute_plan", lambda _self, result, tag, gripper_result=None:
                        captured.append((result, gripper_result)))
    pos = np.arange(21).reshape(3, 7) * .001
    vel = np.ones_like(pos)
    grip = np.array([1., .6, .025])
    task.execute_plan(PlanResult(True, pos, vel), "right", grip)
    np.testing.assert_array_equal(captured[0][0].position, np.repeat(pos, 2, axis=0))
    np.testing.assert_array_equal(captured[0][1], np.repeat(grip, 2))
    assert len(captured[0][0].position) * .002 == len(pos) * .004


def _physical_stub(*, miss=False, carry_drop=False, held_release=False,
                   tiny_rise=None, tiny_slip=None):
    task = StackBowlsThree.__new__(StackBowlsThree)
    state = {"tcp": Pose([.04465, 0., .78474]), "bowl": np.array([0., 0., .765]),
             "grip": 1., "held": False, "calls": [], "public": None, "move_targets": []}
    arm = SimpleNamespace(tcp_offset=Pose(), get_ee_pose=lambda: state["tcp"].to_pose7())
    task.robot = SimpleNamespace(get_arm=lambda _: arm)
    task.bowls = [SimpleNamespace(get_pose=lambda: Pose(state["bowl"], [2**-.5, 2**-.5, 0, 0]))]
    task.TABLE_TOP_Z = .765
    task._bowl_scale = .047
    task.plan_success = True
    task._bowl_motion_scene = lambda _: nullcontext()
    task._bowl_xyz = lambda _: state["bowl"].copy()
    task.step_sim = lambda: None

    def grasp(**kwargs):
        state["public"] = kwargs
        target = state["tcp"]
        if miss: state["tcp"] = Pose(target.p + [0, 0, .03], target.q)
        return target, target, SimpleNamespace(kinematic_pin=True)

    def set_grip(value, _tag):
        state["calls"].append(("grip", value))
        state["grip"] = value
        if value == task.GRIPPER_HOLD_VALUE:
            state["held"] = True
        elif state["held"] and not held_release:
            state["held"] = False
            state["bowl"][2] = .810

    def move(target, _tag, label, **_kwargs):
        state["calls"].append(("move", label))
        state["move_targets"].append((label, target.p-state["tcp"].p, _kwargs))
        before_z = state["bowl"][2]
        if state["held"]:
            state["bowl"] += target.p - state["tcp"].p
        if label == "tiny lift":
            if tiny_rise is not None:
                state["bowl"][2] = before_z + tiny_rise
            if tiny_slip is not None:
                state["bowl"] += tiny_slip
        state["tcp"] = target
        if label == "carry" and carry_drop: state["bowl"][2] -= .10
        return target

    task.select_and_execute_grasp = grasp
    task.open_gripper = lambda _tag: set_grip(1., _tag)
    task.set_gripper = set_grip
    task._bowl_move = move
    return task, state


def test_public_scaled_rim_grasp_opens_before_withdrawal():
    task, state = _physical_stub()
    assert task._pick_and_stack_one(0, (.15, -.1, .850))
    assert state["public"]["object_scale"] == .047
    assert state["public"]["table_z"] == .765
    assert len(state["public"]["candidate_grasps"]) == 8
    assert state["public"]["max_candidates"] == 10
    assert all(g.scale_frame == "mesh_unit" for g in state["public"]["candidate_grasps"])
    assert callable(state["public"]["post_plan_check"])
    assert [v for k, v in state["calls"] if k == "move"] == ["tiny lift", "lift", "carry", "descent", "withdrawal"]
    assert [v for k,v in state["calls"] if k=="grip"] == [1.,.0125,.35,1.]
    release=state['calls'].index(('grip',.35))
    retreat=state['calls'].index(('move','withdrawal'))
    assert state['calls'][release:retreat]==[('grip',.35),('grip',1.)]
    # The actor deliberately has no entity/set_pos/set_quat API. Even an
    # annotation requesting kinematic pinning must use physical stepping.


def test_missed_grasp_does_not_close():
    task, state = _physical_stub(miss=True)
    assert not task._pick_and_stack_one(0, (.15, -.1, .850))
    assert all(kind != "grip" or value == task.GRIPPER_ENTRY_VALUE for kind, value in state["calls"])


def test_initial_clearance_still_uses_native_tiny_limits():
    task, state = _physical_stub(tiny_rise=.035)
    assert task._pick_and_stack_one(0, (.15, -.1, .850))
    label, delta, options = state["move_targets"][0]
    assert label == "tiny lift" and options["tiny"] is True
    np.testing.assert_allclose(delta, [0., 0., .050])


@pytest.mark.parametrize("fault", [dict(tiny_rise=.019), dict(tiny_slip=np.array([.031, 0., 0.]))])
def test_initial_clearance_rejects_failed_body_rise_or_retention(fault):
    task, state = _physical_stub(**fault)
    assert not task._pick_and_stack_one(0, (.15, -.1, .850))
    assert [v for k, v in state["calls"] if k == "move"] == ["tiny lift"]
    assert [v for k, v in state["calls"] if k == "grip"] == [task.GRIPPER_ENTRY_VALUE, task.GRIPPER_HOLD_VALUE]


def test_carry_drop_aborts_before_release():
    task, state = _physical_stub(carry_drop=True)
    assert not task._pick_and_stack_one(0, (.15, -.1, .850))
    assert ("move", "descent") not in state["calls"]
    assert ("grip", .4) not in state["calls"]


def test_held_bowl_after_withdrawal_is_failure():
    task, state = _physical_stub(held_release=True)
    assert not task._pick_and_stack_one(0, (.15, -.1, .850))
    assert ("move", "withdrawal") in state["calls"]


def test_open_grasp_refinement_corrects_measured_pd_bias_with_native_tiny_motion():
    task, state = _physical_stub()
    target = state["tcp"]
    bias = np.array([.001, .004, .0054])
    state["tcp"] = Pose(target.p - bias, target.q)
    moves = []

    def move(command, tag, label, *, tiny=False, check_arrival=True):
        assert tag == "right" and tiny
        # The sag-biased command is not itself an arrival target.
        assert check_arrival is False
        assert state["grip"] == 1.
        moves.append(command)
        state["tcp"] = Pose(command.p - bias, command.q)
        return state["tcp"]

    task._bowl_move = move
    actual = task._bowl_refine_open_grasp(target, "right", lambda: state["bowl"])
    assert len(moves) == 1
    np.testing.assert_allclose(moves[0].p, target.p + bias)
    np.testing.assert_allclose(actual.p, target.p)
    assert state["calls"] == []


@pytest.mark.parametrize("fault", ["large_bias", "rotation", "body_motion", "nonfinite_body", "stalled", "total_bound"])
def test_open_grasp_refinement_stops_on_unreliable_feedback(fault):
    task, state = _physical_stub()
    target = state["tcp"]
    distance = (task.BOWL_REFINE_MAX_ERROR_M + .0001 if fault == "large_bias"
                else (.007 if fault == "total_bound" else .003))
    state["tcp"] = Pose(target.p - [distance, 0, 0], target.q)
    if fault == "rotation":
        state["tcp"] = Pose(state["tcp"].p, [np.cos(.03), 0, 0, np.sin(.03)])
    moves = []

    def move(command, tag, label, *, tiny=False, check_arrival=True):
        assert tiny
        moves.append(command)
        if fault == "body_motion":
            state["bowl"][0] += .0031
        elif fault == "nonfinite_body":
            state["bowl"][0] = np.nan
        return state["tcp"]

    task._bowl_move = move
    with pytest.raises(_BowlMotionRejected):
        task._bowl_refine_open_grasp(target, "right", lambda: state["bowl"])
    assert not task.plan_success
    assert len(moves) <= 3
    if fault in ("large_bias", "rotation"):
        assert moves == []
    assert state["grip"] == 1.


def test_unresolved_public_grasp_alignment_aborts_before_expansion():
    task, state = _physical_stub()
    public = task.select_and_execute_grasp

    def biased_public(**kwargs):
        result = public(**kwargs)
        state["tcp"] = Pose(result[0].p - [task.BOWL_REFINE_MAX_ERROR_M + .001, 0, 0], result[0].q)
        return result

    task.select_and_execute_grasp = biased_public
    assert not task._pick_and_stack_one(0, (.15, -.1, .850))
    assert all(kind != "grip" or value == task.GRIPPER_ENTRY_VALUE for kind, value in state["calls"])


@pytest.mark.parametrize("scale", [.047, .060])
@pytest.mark.parametrize("yaw", [-.7, 0., .7])
def test_lip_candidates_match_wall_slope_and_apply_scale_once(scale, yaw):
    """Pinch tilt equals the measured 4 degree wall flare, band centred at 38 mm
    on the straight wall below the rim curl (wall_profile_v1); 38 mm failed the 30 mm palm guard."""
    from envs.task_bases import stack_bowls_three as module
    assert module._LIP_TILT_DEG == 4.0 and module._LIP_HEIGHT_M == .041
    task = StackBowlsThree.__new__(StackBowlsThree)
    obj = Pose([.1, -.2, .765], [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]) * Pose(
        [0, 0, 0], [2**-.5, 2**-.5, 0, 0])
    candidates = task._bowl_grasp_candidates()
    assert len(candidates) == 8
    for grasp in candidates:
        tcp = grasp.to_world(obj, scale)
        offset = tcp.p - obj.p
        ratio = scale/.047
        assert .052*ratio < np.linalg.norm(offset[:2]) < .056*ratio
        assert abs(offset[2] - .041*ratio) < .0004*ratio
        rotation = tcp.to_matrix()[:3, :3]
        assert rotation[2,2] == pytest.approx(-np.cos(np.deg2rad(4)))
        assert rotation[2,1] == pytest.approx(-np.sin(np.deg2rad(4)))
        assert np.dot(rotation[:2,2],offset[:2]) < 0
        assert grasp.kinematic_pin is False


def test_lip_grasps_center_actual_pad_surfaces_and_clear_open_hand():
    import itertools
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import genesis as gs
    import trimesh
    from envs.robot.franka_robot import FrankaArm
    from envs.grasp import tcp_to_link_pose
    from envs.utils import ASSETS_PATH
    task = StackBowlsThree.__new__(StackBowlsThree)
    mesh = trimesh.load(ASSETS_PATH/'objects/002_bowl/visual/base1.glb',force='mesh',process=False)
    mesh.vertices *= .047
    assets=Path(gs.__file__).parent/'assets/xml/franka_emika_panda'
    xml=ET.parse(assets/'panda.xml').getroot()
    points=[np.asarray(trimesh.load(assets/'assets/finger_0.obj',force='mesh',process=False).vertices)]
    palm=trimesh.load(assets/'assets/hand.stl',force='mesh',process=False)
    for default in xml.findall('.//default'):
        if not default.get('class','').startswith('fingertip_pad_collision_'):continue
        geom=default.find('geom');half=np.fromstring(geom.get('size'),sep=' ');center=np.fromstring(geom.get('pos'),sep=' ')
        points.append(center+np.array(list(itertools.product(*[[-h,0,h] for h in half]))))
    points=np.concatenate(points)
    arm=FrankaArm(None,Pose())
    for grasp in task._bowl_grasp_candidates():
        tcp = grasp.to_world(Pose(),.047)
        rotation = tcp.to_matrix()[:3,:3]
        direction = rotation[:,1]
        offsets=np.array([[x,0,z] for x,z in itertools.product(
            np.linspace(-.0055,.0055,11),np.linspace(-.0085,.0085,11))])
        centers=offsets@rotation.T+tcp.p
        hits,_,triangles=mesh.ray.intersects_location(
            centers-.12*direction,np.tile(direction,(len(centers),1)),multiple_hits=True)
        local=(hits-tcp.p)@rotation
        near=np.abs(local[:,1])<.04
        left,right=local[near,1].min(),local[near,1].max()
        assert -left == pytest.approx(right,abs=1e-6)
        # Half wall thickness at 41 mm is about 1.5 mm (2.9-3.0 mm wall, 4 degree slice).
        assert .0012 < right < .0020
        normals=[]
        for sign in (-1,1):
            sel=near & (sign*local[:,1]>=right-.001)
            n=mesh.face_normals[triangles[sel]].mean(0)
            normals.append(n/np.linalg.norm(n))
        assert np.dot(*normals)<-.8
        link=tcp_to_link_pose(tcp,arm.tcp_offset)
        palm_world=palm.vertices@link.to_matrix()[:3,:3].T+link.p
        in_hand=(mesh.vertices-link.p)@link.to_matrix()[:3,:3]
        mask=np.all((in_hand[:,:2]>=palm.bounds[0,:2])&(in_hand[:,:2]<=palm.bounds[1,:2]),axis=1)
        assert mask.any() and in_hand[mask,2].min()-palm.bounds[1,2]>.030
        fingers=[]
        for sign in (-1,1):
            pts=points.copy()
            if sign<0:pts[:,:2]*=-1
            pts+=np.array([0,sign*.040,.0584])
            fingers.append(pts@link.to_matrix()[:3,:3].T+link.p)
        world=np.concatenate(fingers+[palm_world])
        assert world[:,1].min()-mesh.bounds[0,1]>.007
        # Finite mesh evidence does not certify collision-free native paths,
        # contact forces, retained lift or release; physical guards remain.


def _friction_loads(monkeypatch, module):
    loads = []

    class _Entity:
        def __init__(self, pos, quat):
            self._pos, self._quat = np.asarray(pos, float), np.asarray(quat, float)

        def get_pos(self):
            return self._pos.copy()

        def get_quat(self):
            return self._quat.copy()

    def load_object(_scene, pose, object_id, **kwargs):
        loads.append((object_id, dict(kwargs)))
        return SimpleNamespace(entity=_Entity(pose.p, pose.q))

    monkeypatch.setattr(module, "load_object", load_object)
    if hasattr(module, "load_fixed_collision_bowl"):
        monkeypatch.setattr(module, "load_fixed_collision_bowl", load_object)
    monkeypatch.setattr(module, "_read_scale", lambda *_a, **_k: .047)
    monkeypatch.setattr(module, "_read_extents", lambda *_a, **_k: np.array([3., 1., 3.]))
    return loads


@pytest.mark.parametrize("assist", [False, True])
def test_every_bowl_is_built_with_stiff_pair_friction(monkeypatch, assist):
    """V9 rim pinches held ~0.02 N per pad at pair friction 5 (325x softer rows);
    the corrected constant must reach every bowl in both base and assist layouts."""
    import envs.task_bases.stack_bowls_three as base_module
    import envs.tasks.stack_three_bowls_contribution as assist_module
    module = assist_module if assist else base_module
    loads = _friction_loads(monkeypatch, module)
    cls = assist_module.StackThreeBowlsContribution if assist else StackBowlsThree
    task = cls({"hard": True})
    task.scene = object()
    task.TABLE_TOP_Z = .765
    np.random.seed(3)
    task.load_actors()
    assert StackBowlsThree.BOWL_FRICTION == 1.0
    assert cls.BOWL_FRICTION == 1.0 and cls.BOWL_DENSITY == 10.0
    assert len(loads) == task._num_bowls >= 3
    for object_id, kwargs in loads:
        assert object_id == base_module._BOWL_ID
        assert kwargs["friction"] == 1.0
        assert kwargs["density"] == 10.0
        assert kwargs["convex"] is True and kwargs["is_static"] is False


def test_avatar_variants_opt_into_fixed_collision_bowls(monkeypatch):
    """Assist, interrupt and neutral load every bowl through the fixed
    collision package; the no-avatar base keeps runtime CoACD byte-for-byte."""
    import envs.task_bases.stack_bowls_three as base_module
    import envs.tasks.stack_three_bowls_contribution as assist_module
    from envs.tasks.stack_three_bowls_interruption import StackThreeBowlsInterruption
    from envs.tasks.stack_three_bowls_copresence import StackThreeBowlsCopresence
    assert StackBowlsThree.BOWL_FIXED_COLLISION is False
    assert StackThreeBowlsInterruption.BOWL_FIXED_COLLISION is True
    assert StackThreeBowlsCopresence.BOWL_FIXED_COLLISION is True
    assert assist_module.StackThreeBowlsContribution.BOWL_FIXED_COLLISION is True
    calls = []

    def fixed(_scene, pose, object_id, **kwargs):
        calls.append(("fixed", object_id, kwargs))
        return SimpleNamespace(entity=SimpleNamespace(get_pos=lambda: np.asarray(pose.p),
                                                      get_quat=lambda: np.asarray(pose.q)))

    def legacy(*_a, **_k):
        raise AssertionError("assist must not load bowls through runtime CoACD")

    monkeypatch.setattr(assist_module, "load_fixed_collision_bowl", fixed)
    monkeypatch.setattr(assist_module, "load_object", legacy)
    monkeypatch.setattr(assist_module, "_read_scale", lambda *_a, **_k: .047)
    monkeypatch.setattr(assist_module, "_read_extents", lambda *_a, **_k: np.array([3., 1., 3.]))
    task = assist_module.StackThreeBowlsContribution({"hard": True})
    task.scene = object()
    task.TABLE_TOP_Z = .765
    np.random.seed(5)
    task.load_actors()
    assert len(calls) == task._num_bowls >= 3
    for _kind, object_id, kwargs in calls:
        assert object_id == base_module._BOWL_ID
        assert kwargs == {"model_id": base_module._BOWL_MODEL_ID, "convex": True,
                          "is_static": False, "friction": 1.0, "density": 10.0}


def test_fixed_collision_loader_rejects_other_objects():
    import envs.task_bases.stack_bowls_three as base_module
    with pytest.raises(ValueError):
        base_module.load_fixed_collision_bowl(object(), SimpleNamespace(p=[0, 0, 0], q=[1, 0, 0, 0]),
                                              "003_plate", model_id=1)


def test_fixed_collision_package_integrity_and_thin_rim_wall():
    """The URDF names the textured visual and exactly the frozen hulls at the
    catalog scale, and those hulls give a rim wall within .5 mm of the visual
    3.0 mm wall at every 15 degree sector (runtime CoACD gave 5-9 mm)."""
    import hashlib
    import json
    import xml.etree.ElementTree as ET
    import trimesh
    import envs.task_bases.stack_bowls_three as base_module
    urdf = base_module._FIXED_COLLISION_URDF
    package = urdf.parent
    manifest = json.loads((package / "package_manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest, name
    scale = base_module._read_scale(base_module._BOWL_ID, base_module._BOWL_MODEL_ID)
    assert manifest["scale"] == pytest.approx(scale)
    root = ET.parse(urdf).getroot()
    links = root.findall("link")
    assert len(links) == 1
    visuals = links[0].findall("visual/geometry/mesh")
    assert len(visuals) == 1
    assert (package / visuals[0].get("filename")).resolve() == (
        base_module.ASSETS_PATH / "objects" / base_module._BOWL_ID / "visual" / "base1.glb").resolve()
    meshes = links[0].findall("collision/geometry/mesh")
    names = sorted(m.get("filename") for m in meshes)
    assert names == sorted(p.name for p in package.glob("hull_*.obj")) and len(names) == manifest["n_hulls"]
    for mesh in meshes + visuals:
        assert [float(v) for v in mesh.get("scale").split()] == pytest.approx([scale] * 3, rel=1e-5)
    hulls = [trimesh.load(package / name, force="mesh", process=False) for name in names]
    # Genesis convexifies each collision mesh; each frozen piece must already be
    # its own convex hull so that step changes nothing.
    for hull in hulls:
        assert hull.convex_hull.volume == pytest.approx(abs(hull.volume), rel=1e-3)
    visual = trimesh.load(base_module.ASSETS_PATH / "objects" / base_module._BOWL_ID / "visual" / "base1.glb",
                          force="mesh", process=False)

    def wall(meshes, degrees, height):
        radial = np.array([np.cos(np.deg2rad(degrees)), 0., np.sin(np.deg2rad(degrees))])
        origin = (np.array([0., height, 0.]) + .15 * radial) / scale
        spans = []
        for mesh in meshes:
            hits, _, _ = mesh.ray.intersects_location([origin], [-radial], multiple_hits=True)
            radii = sorted(float(p @ radial) * scale for p in hits if p @ radial > 0)
            if radii:
                spans.append((radii[0], radii[-1]))
        spans.sort(key=lambda s: -s[1])
        outer, inner = spans[0][1], spans[0][0]
        for low, high in spans[1:]:
            if high >= inner - 1e-4:
                inner = min(inner, low)
        return outer, inner

    for degrees in range(0, 360, 15):
        for height in (.038, .041, .044):
            vo, vi = wall([visual], degrees, height)
            ho, hi = wall(hulls, degrees, height)
            assert abs(ho - vo) <= .0005 and abs(hi - vi) <= .0005, (degrees, height, ho - vo, hi - vi)


def test_assist_offers_finger_swapped_rim_pinch_after_each_original():
    """Carry/lift rejections of the original rim pinches come from joint 7
    crossing its limit; the same pinch with swapped fingers (180 deg about
    the approach axis) sits a half turn away on the wrist. Siblings keep the
    eight originals."""
    import envs.tasks.stack_three_bowls_contribution as assist_module
    from envs.tasks.stack_three_bowls_interruption import StackThreeBowlsInterruption
    from envs.tasks.stack_three_bowls_copresence import StackThreeBowlsCopresence
    assert StackBowlsThree.BOWL_WRIST_FLIP is False
    # The interrupt and neutral experts use the same
    # finger-swapped candidates (expert-side only; scenes are unchanged).
    assert StackThreeBowlsInterruption.BOWL_WRIST_FLIP is True
    assert StackThreeBowlsCopresence.BOWL_WRIST_FLIP is True
    cls = assist_module.StackThreeBowlsContribution
    assert cls.BOWL_WRIST_FLIP is True and cls.BOWL_WRIST_HEADROOM == .75
    base = StackBowlsThree.__new__(StackBowlsThree)._bowl_grasp_candidates()
    assist = cls.__new__(cls)._bowl_grasp_candidates()
    assert len(base) == 8 and len(assist) == 16
    for i, original in enumerate(base):
        same, flipped = assist[2 * i], assist[2 * i + 1]
        assert same.name == original.name and flipped.name == original.name + "_flip"
        np.testing.assert_allclose(same.pose.p, original.pose.p)
        np.testing.assert_allclose(same.pose.q, original.pose.q)
        np.testing.assert_allclose(flipped.pose.p, original.pose.p)
        a = original.pose.to_matrix()[:3, :3]
        b = flipped.pose.to_matrix()[:3, :3]
        np.testing.assert_allclose(b[:, 2], a[:, 2], atol=1e-9)   # same approach
        np.testing.assert_allclose(b[:, 1], -a[:, 1], atol=1e-9)  # swapped closing
        assert flipped.pre_distance == original.pre_distance and flipped.source == "manual"
        assert flipped.scale_frame == "mesh_unit" and flipped.kinematic_pin is False


@pytest.mark.parametrize("wrist, ok", [(2.14, True), (2.15, False), (2.6, False), (-.5, True), (-2.95, False)])
def test_wrist_headroom_keeps_measured_carry_travel(wrist, ok):
    import envs.tasks.stack_three_bowls_contribution as assist_module
    task = assist_module.StackThreeBowlsContribution.__new__(assist_module.StackThreeBowlsContribution)
    limits = [[-2.9, 2.9]] * 6 + [[-2.8973, 2.8973]]
    task.robot = SimpleNamespace(get_arm=lambda _t: SimpleNamespace(
        planner=SimpleNamespace(_mplib=SimpleNamespace(joint_limits=limits))))
    grasp = SimpleNamespace(position=np.array([[0.] * 6 + [0.], [0.] * 6 + [wrist]]))
    assert task._bowl_wrist_headroom((None, grasp, None, None), "right") is ok
    assert task._bowl_wrist_headroom((None, SimpleNamespace(position=np.zeros((0, 7))), None, None), "right") is False
    base = StackBowlsThree.__new__(StackBowlsThree)
    assert base._bowl_wrist_headroom((None, grasp, None, None), "right") is True
