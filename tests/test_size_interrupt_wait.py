"""Size-interrupt expert must yield through the inspection hand withdrawal."""
from types import SimpleNamespace

from envs.tasks.rank_blocks_by_size_interruption import RankBlocksBySizeInterruption
from envs.tasks.rank_blocks_by_color_interruption import RankBlocksByColorInterruption


def test_size_waits_after_attach_until_withdrawal_finishes():
    task = RankBlocksBySizeInterruption({})
    frames = {"remaining": 3}
    events = []
    task.avatar = SimpleNamespace(spare=lambda: frames["remaining"] == 0)
    task.interaction_controller.wait_until_avatar_attach = lambda owner: events.append("attach")
    task.avatar_collided = False
    task.plan_success = True

    def step():
        frames["remaining"] -= 1
        events.append("step")
        # Waiting must preserve a real contact/failure if one occurs there.
        task.avatar_collided = True
        task.plan_success = False

    task.step_sim = step
    task._wait_until_avatar_attach()
    assert events == ["attach", "step", "step", "step"]
    assert task.avatar.spare()
    assert task.avatar_collided and not task.plan_success


def test_finished_inspection_adds_no_simulation_steps():
    task = RankBlocksBySizeInterruption({})
    task.avatar = SimpleNamespace(spare=lambda: True)
    calls = []
    task.interaction_controller.wait_until_avatar_attach = lambda owner: calls.append("attach")
    task.step_sim = lambda: calls.append("unexpected step")
    task._wait_until_avatar_attach()
    assert calls == ["attach"]


def test_no_avatar_wait_is_empty_and_rgb_keeps_original_hook():
    task = RankBlocksBySizeInterruption({})
    task.avatar = None
    task.step_sim = lambda: (_ for _ in ()).throw(AssertionError("unexpected step"))
    task._wait_until_avatar_attach()
    assert RankBlocksByColorInterruption._wait_until_avatar_attach is not RankBlocksBySizeInterruption._wait_until_avatar_attach
