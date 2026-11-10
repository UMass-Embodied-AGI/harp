"""Every motion delegator dump_bin_interrupt overrides must keep its base signature.

dump_bin_interrupt wraps a few motion entry points purely to refresh the avatar
obstacle cloud before the base plans. A wrapper that silently narrows the
signature raises TypeError in the middle of a rollout rather than failing at
import, which is how ``_move_screw(..., tcp_yaw=...)`` -- the call
dump_bin_grasp makes for every oriented top-down pick -- crashed two of the five
episodes in the original [220,225) group.
"""
import inspect

import pytest

from envs.tasks.dump_bin_interruption import DumpBinInterruption

WRAPPED = ['_move_screw', '_move_seeded']


def base_impl(name):
    for klass in DumpBinInterruption.__mro__[1:]:
        if name in vars(klass):
            return klass, vars(klass)[name]
    return None, None


@pytest.mark.parametrize('name', WRAPPED)
def test_override_accepts_everything_the_base_accepts(name):
    if name not in vars(DumpBinInterruption):
        pytest.skip(f'{name} is not overridden')
    klass, base = base_impl(name)
    assert base is not None, f'no base implementation of {name}'
    over_sig, base_sig = inspect.signature(vars(DumpBinInterruption)[name]), inspect.signature(base)
    missing = [p for p in base_sig.parameters if p not in over_sig.parameters]
    assert not missing, f'{name} drops base parameter(s) {missing}: {over_sig} vs {base_sig} (base in {klass.__name__})'


def test_move_screw_forwards_tcp_yaw_rather_than_swallowing_it(monkeypatch):
    klass, _base = base_impl('_move_screw')
    seen = {}
    monkeypatch.setattr(klass, '_move_screw',
                        lambda self, target_pos, arm_tag, tcp_yaw=0.0: seen.update(tcp_yaw=tcp_yaw, arm=arm_tag) or True,
                        raising=True)
    task = DumpBinInterruption.__new__(DumpBinInterruption)
    task._refresh_avatar_planner_obstacles = lambda arm_tag: seen.update(refreshed=True)
    assert DumpBinInterruption._move_screw(task, [0., 0., 1.], 'right', tcp_yaw=1.25) is True
    assert seen['refreshed'] is True and seen['arm'] == 'right'
    assert seen['tcp_yaw'] == pytest.approx(1.25)
