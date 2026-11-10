"""Base motion module."""
from ..robot import AvatarRobot


class BaseMotionModule:
    def __init__(self, motion_name: str, robot: AvatarRobot):
        self.motion_name = motion_name
        self.robot = robot
