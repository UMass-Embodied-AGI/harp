from envs.task_bases.dump_bin import DumpBin


def test_held_creep_objects_spawn_at_friction_one():
    assert DumpBin.HELD_CREEP_FRICTION == {"031_jam-jar": 1.0, "035_apple": 1.0}
    assert DumpBin.OBJECT_FRICTION == 4.0
