"""Avatar module: minimal avatar simulation for Genesis."""
from .controller import AvatarController
from .collider import AvatarCollider
from .collision_checker import AvatarCollisionChecker

__all__ = ["AvatarController", "AvatarCollider", "AvatarCollisionChecker"]
