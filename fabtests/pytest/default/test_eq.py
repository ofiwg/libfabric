import pytest

@pytest.mark.pr_ci
@pytest.mark.unit
def test_eq(cmdline_args):
    from common import UnitTest
    test = UnitTest(cmdline_args, "fi_eq_test")
    test.run()
