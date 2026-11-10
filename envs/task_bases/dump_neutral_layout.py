"""Experimental parked-robot clearance for dump-neutral candidate layouts."""

import numpy as np

from ..avatar.utils import ActionStatus, AvatarState


def initial_avatar_pose_is_clear(task, job, xy, body_pos):
    """Query the existing checker at frame zero, restoring the preview pose.

    This runs only during reset-time generic layout selection, before props
    attach. It does not step physics, move the robot, advance any motion clock,
    alter collision history, or approximate the checker's skin/contact rules.
    The separate seated/laptop/phone placement routes never call this helper.
    """
    avatar = task.avatar
    collider = task.avatar_collider
    checker = task.avatar_collision_checker
    if (avatar is None or collider is None or checker is None
            or checker.collider is not collider):
        raise RuntimeError('dump neutral initial clearance requires its collision checker')
    robot = avatar.robot
    motion = avatar.motion_modules.get(job['motion'])
    if motion is None or len(getattr(motion, 'data', ())) == 0:
        raise RuntimeError('dump neutral initial clearance requires a loaded motion')
    if (getattr(robot, 'skin', None) is None
            or any(item is not None for item in robot.attached_object)
            or robot.two_hand_attached_object is not None):
        raise RuntimeError('dump neutral initial clearance requires unattached avatar skin')

    position = np.asarray(body_pos, dtype=float).copy()
    # Layout recovery may measure a rotated candidate at the same work point.
    # Check that exact pose, not the original side heading.
    candidate_rotation = getattr(task, '_neutral_avatar_measure_body_rot', None)
    if candidate_rotation is None:
        candidate_rotation = task._neutral_avatar_body_rot_for_work_xy(job, xy)
    rotation = np.asarray(candidate_rotation, dtype=float).copy()
    if (position.shape != (3,) or rotation.shape != (3, 3)
            or not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation))):
        raise RuntimeError('invalid dump neutral candidate body pose')

    # Assign rather than mutate the original arrays: even their identities are
    # restored. Avoid avatar.reset()/play_animation(), which clear attachments
    # and alter controller and animation state. This is the same skin frame
    # applied by _neutral_avatar_hold_motion_frame_idle after layout selection.
    fields = ('global_trans', 'global_rot', 'pose', 'node_trans', 'global_mat',
              'global_mat_inv', 'base_state', 'action_state', 'action_status')
    saved = {name: getattr(robot, name) for name in fields}
    try:
        robot.global_trans = position
        robot.global_rot = rotation
        robot.pose = np.asarray(motion.data[0]).copy()
        robot.node_trans = np.asarray(motion.node_data[0]).copy()
        robot.global_mat = np.asarray(motion.global_mat).copy()
        robot.global_mat_inv = np.asarray(motion.global_mat_inv).copy()
        robot.base_state = AvatarState.STANDING
        robot.action_state = AvatarState.NO_ACTION
        robot.action_status = ActionStatus.INIT
        robot.update()
        # Analytic capsules, palm boxes and mesh refinement query the updated
        # skin directly; no collider rigid-entity updates or simulation steps.
        result = checker.check()
        if not isinstance(result, dict) or 'collided' not in result:
            raise RuntimeError('invalid dump neutral initial clearance result')
        return not bool(result['collided'])
    finally:
        for name, value in saved.items():
            setattr(robot, name, value)
        robot.update()
