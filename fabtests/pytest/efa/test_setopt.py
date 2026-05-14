import pytest

@pytest.mark.pr_ci
@pytest.mark.fabric(params=["efa", "efa-direct"])
@pytest.mark.unit
def test_setopt(cmdline_args, fabric):
    from common import UnitTest
    test = UnitTest(cmdline_args, f"fi_setopt_test -f {fabric}")
    test.run()

