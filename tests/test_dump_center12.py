from envs.task_bases.dump_bin_motion import DumpBinMotionMixin


def test_centering_correction_covers_the_measured_pd_sag():
    assert DumpBinMotionMixin.CENTER_CORRECTION_CEILING == .012
    assert DumpBinMotionMixin.CENTER_CORRECTION_BUDGET == .014
