"""scene.add_entity must not consume the global np.random stream."""

import numpy as np

from envs.base_task_parts.scene import isolate_add_entity_rng


class _Scene:
    def __init__(self):
        self.calls = 0

    def add_entity(self, morph, **kwargs):
        self.calls += 1
        np.random.random(17)          # e.g. a renderer-specific mesh pass
        return ("entity", morph, kwargs.get("tag"))


def test_add_entity_restores_the_global_stream():
    scene = _Scene()
    isolate_add_entity_rng(scene)
    isolate_add_entity_rng(scene)      # idempotent
    np.random.seed(9000095)
    ref = np.random.RandomState(9000095).randint(0, 100, size=3).tolist()
    assert scene.add_entity("urdf", tag="cabinet") == ("entity", "urdf", "cabinet")
    assert scene.calls == 1
    assert np.random.randint(0, 100, size=3).tolist() == ref


def test_state_is_restored_even_when_loading_raises():
    class Boom(_Scene):
        def add_entity(self, morph, **kwargs):
            np.random.random(3)
            raise RuntimeError("bad mesh")

    scene = Boom()
    isolate_add_entity_rng(scene)
    np.random.seed(1)
    ref = np.random.RandomState(1).random()
    try:
        scene.add_entity("x")
    except RuntimeError:
        pass
    assert np.random.random() == ref
