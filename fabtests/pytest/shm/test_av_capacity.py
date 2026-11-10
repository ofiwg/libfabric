import os

import pytest


@pytest.mark.pr_ci
@pytest.mark.unit
def test_av_capacity(cmdline_args):
    from common import UnitTest
    min_addrs = max(256, os.cpu_count() or 0)
    test = UnitTest(cmdline_args, "fi_av_capacity_test -n " + str(min_addrs))
    test.run()
