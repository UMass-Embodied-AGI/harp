from envs.task_bases import dump_bin_grasp


def test_apple_and_jar_wait_for_rest_before_acquisition():
    assert dump_bin_grasp.REST_BEFORE_ACQUISITION == ('031_jam-jar', '035_apple')
