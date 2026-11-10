"""RGB-interrupt expert must yield through the inspection hand withdrawal."""
from types import SimpleNamespace

from envs.tasks.rank_blocks_by_color_interruption import RankBlocksByColorInterruption


def test_waits_through_withdrawal_and_preserves_failures():
    task = RankBlocksByColorInterruption({})
    state = {"remaining": 3}
    events = []
    task.avatar = SimpleNamespace(spare=lambda: state["remaining"] == 0)
    task.interaction_controller.wait_until_avatar_attach = lambda owner: events.append("attach")
    task.avatar_collided = False
    task.plan_success = True

    def step():
        state["remaining"] -= 1
        events.append("step")
        task.avatar_collided = True
        task.plan_success = False

    task.step_sim = step
    task._wait_until_avatar_attach()
    assert events == ["attach", "step", "step", "step"]
    assert task.avatar.spare()
    assert task.avatar_collided and not task.plan_success


def test_finished_inspection_adds_no_steps():
    task = RankBlocksByColorInterruption({})
    task.avatar = SimpleNamespace(spare=lambda: True)
    calls = []
    task.interaction_controller.wait_until_avatar_attach = lambda owner: calls.append("attach")
    task.step_sim = lambda: calls.append("unexpected step")
    task._wait_until_avatar_attach()
    assert calls == ["attach"]


def test_absent_avatar_does_not_call_controller_or_step():
    task = RankBlocksByColorInterruption({})
    task.avatar = None
    calls = []
    task.interaction_controller.wait_until_avatar_attach = lambda owner: calls.append("unexpected attach")
    task.step_sim = lambda: calls.append("unexpected step")
    task._wait_until_avatar_attach()
    assert calls == []
