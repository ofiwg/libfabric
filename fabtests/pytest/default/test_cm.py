import pytest

@pytest.mark.pr_ci
@pytest.mark.functional
def test_cm_data(cmdline_args):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args, "fi_cm_data")
    test.run()
