"""Co-presence scenario for the categorize family (``categorize_objects_copresence``)."""

import numpy as np

from ..base_task import BaseTask
from ..interaction.eval_mode import CategorizeNeutralController
from ..task_bases.categorize import CategorizeTaskCore, PickSpec, PlaceSpec
from ..task_bases.categorize_support.motion import stop_on_motion_rejection
from ..task_bases.neutral_avatar_table_work import (
    NeutralAvatarTableWorkMixin,
    NeutralTaskSpec,
)


class CategorizeObjectsCopresence(NeutralAvatarTableWorkMixin, CategorizeTaskCore):
    """Robot categorizes objects while the human performs unrelated table work."""

    # Plan around the human's objects as exact boxes.
    NEUTRAL_OBJECT_PLANNER_BOXES = True

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        # Match the tested training candidate without requiring special
        # collector flags; callers can still disable either fix explicitly.
        cfg.setdefault("categorize_payload_friction_v1", True)
        cfg.setdefault("categorize_split_screw_v1", True)
        super().__init__(cfg)

    _CATEGORIZE_COLUMN_LAYOUT = True
    _CATEGORIZE_ROBOT_Y = -0.5
    _CATEGORIZE_FINGER_KV = 200.0
    _CATEGORIZE_OPEN_LIDS = False
    _USE_OBSTACLE_INFLATION = True

    table_offset = np.array([0.0, -0.15])
    TABLE_HALF_SIZE_Y = 0.50
    TABLE_LEGS_HALF_Y = 0.45
    INTERRUPTED_IDX = 1
    INTERRUPT_OBJECT_Y = 0.18
    EVAL_TRIGGER_STEP_MIN = 80
    EVAL_TRIGGER_STEP_MAX = 180

    PREFERRED_GRASP_NAMES = {
        "035_apple": (
            "manual_center_p0_yaw0",
            "manual_center_p0_yaw2",
            "manual_underhand_p0_yaw0",
            "manual_underhand_p0_yaw2",
        ),
        "086_woodenblock": (
            "manual_topdown_yaw45_center",
            "manual_topdown_yaw45_higher",
        ),
        "100_seal": (
            "manual_topdown_X_center_m2",
            "manual_topdown_Y_center_m2",
            "manual_grasp_015_yaw90_deeper_upper",
        ),
    }

    static_camera_list = [{
        "name": "head_camera",
        "position": [0.0, -0.145, 2.85],
        "forward": [0.0, -0.005, -1.35],
    }]
    recording_camera_pos = [1.7, 0.05, 1.85]
    recording_camera_lookat = [0.0, 0.0, 1.55]
    side_camera_pos = [0.0, -0.25, 2.7]
    side_camera_lookat = [0.0, -0.25, 1.30]

    NEUTRAL_TASK_SPEC = NeutralTaskSpec(
        # Categorize uses a larger table; keep neutral work on the right/back
        # edge, away from the basket/object columns.
        work_xy=(0.50, 0.24),
        mouse_work_xy=(0.50, 0.30),
        sponge_work_region=(0.46, 0.66, 0.22, 0.34),
        table_xy_bounds=(-0.80, 0.80, -0.55, 0.25),
        work_candidates=(
            (0.50, 0.24),
            (0.58, 0.24),
            (0.50, 0.32),
            (0.58, 0.32),
            (-0.50, 0.24),
            (-0.58, 0.24),
        ),
        mouse_work_candidates=(
            (0.50, 0.30),
            (0.58, 0.30),
            (0.50, 0.34),
            (0.58, 0.34),
            (-0.50, 0.30),
        ),
    )

    INSTRUCTION = (
        "sort the objects into matching baskets while the human works nearby "
        "on the same table"
    )

    @stop_on_motion_rejection
    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_arm_pd()
        self._start_neutral_avatar_table_work(add_start_delay=True)
        self._open_basket_lids()
        self._move_to_safe(arm_tag)

        ok_any = False
        order = list(range(len(self._sort_targets)))
        order.sort(reverse=True)
        for idx in order:
            actor, obj_name, model_id, basket_idx = self._sort_targets[idx]
            self._neutral_debug(
                f"[categorize_neutral] robot sorting {idx}: {obj_name} -> basket {basket_idx}"
            )
            pick = PickSpec(
                get_center=lambda a=actor, n=obj_name, m=model_id: self._get_object_world_center(a, n, m),
                radius=self._get_object_cross_section_radius(obj_name, model_id),
                label=obj_name,
                close_target_m=self._get_grasp_close_target(obj_name, model_id),
                actor=actor,
            )
            basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
            drop_from_z = basket_pos[2] + (0.10 if obj_name == "100_seal" else 0.18)
            place = PlaceSpec(
                pos=basket_pos,
                label=f"basket {basket_idx}",
                basket_idx=basket_idx,
                drop_from_z=float(drop_from_z),
            )
            ok = self.pick_and_place(pick, place, arm_tag)
            self._move_to_safe(arm_tag)
            ok_any = bool(ok) or ok_any

        return ok_any

    def _robot_basket_metrics(self) -> dict:
        center_threshold = 0.10
        footprint_half = self._BASKET_INNER_HALF_XY + self._BASKET_FOOTPRINT_MARGIN
        per_object = []
        all_correct = True
        for actor, obj_name, model_id, basket_idx in self._sort_targets:
            obj_center = self._get_object_world_center(actor, obj_name, model_id)
            basket_pos = np.array(self.basket_poses[basket_idx].p, dtype=float)
            dist_xy = float(np.linalg.norm(obj_center[:2] - basket_pos[:2]))
            lo_xy, hi_xy = self._actor_footprint_aabb(actor, obj_name, model_id)
            rel_lo = lo_xy - basket_pos[:2]
            rel_hi = hi_xy - basket_pos[:2]
            center_ok = bool(dist_xy <= center_threshold)
            footprint_ok = bool(
                np.all(rel_lo >= -footprint_half)
                and np.all(rel_hi <= footprint_half)
            )
            resting_ok = self._object_resting_in_basket(
                actor, obj_center, basket_pos,
            )
            ok = bool(center_ok and footprint_ok and resting_ok)
            all_correct = all_correct and ok
            per_object.append({
                "side": "robot",
                "object": obj_name,
                "model_id": int(model_id),
                "basket_idx": int(basket_idx),
                "object_pos": obj_center.tolist(),
                "basket_pos": basket_pos.tolist(),
                "dist_xy": dist_xy,
                "rel_aabb_lo_xy": rel_lo.tolist(),
                "rel_aabb_hi_xy": rel_hi.tolist(),
                "center_success": center_ok,
                "footprint_success": footprint_ok,
                "resting_success": resting_ok,
                "success": ok,
            })
        return {
            "basket_center_threshold": center_threshold,
            "basket_footprint_half_xy": float(footprint_half),
            "basket_all_correct": bool(all_correct),
            "basket_objects": per_object,
        }

    def check_success(self) -> bool:
        if not self.plan_success or getattr(self, "_neutral_prop_disturbed", None):
            return False
        metrics = self._robot_basket_metrics()
        if self.avatar_collision_checker is not None and self.avatar_collided:
            summary = self.avatar_collision_summary()
            self._neutral_debug(
                "[categorize_neutral] avatar collision detected: "
                f"first_frame={summary.get('first_collision_frame')} "
                f"deepest={summary.get('deepest_depth_m')}"
            )
            return False
        return bool(metrics["basket_all_correct"])

    def evaluate(self) -> dict:
        metrics = BaseTask.evaluate(self)
        metrics.update(self._robot_basket_metrics())
        # The mixin's neutral metrics (neutral_object_collided,
        # neutral_eval_avatar_failed, trigger step, ...), as the other
        # neutral tasks report them.
        metrics.update(self.interaction_controller.interaction_metrics(self))
        metrics.update(self.neutral_layout_metrics())
        metrics.update(self.neutral_aware_metrics())
        metrics["neutral_prop_disturbed"] = getattr(self, "_neutral_prop_disturbed", None)
        metrics["neutral_robot_only_grading"] = True
        return metrics

    def create_interaction_controller(self) -> CategorizeNeutralController:
        return CategorizeNeutralController()

    def play_blind_once(self) -> bool:
        return self.play_once()
