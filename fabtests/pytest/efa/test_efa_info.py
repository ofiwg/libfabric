import pytest

@pytest.mark.unit
def test_efa_info(cmdline_args):
    from common import UnitTest
    test = UnitTest(cmdline_args, "fi_efa_info_test")
    test.run()
