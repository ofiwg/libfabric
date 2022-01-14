import pytest

@pytest.mark.parametrize("iteration_type",
                         [pytest.param("short", marks=pytest.mark.short),
                          pytest.param("standard", marks=pytest.mark.standard)])
def test_dgram_pingpong(cmdline_args, iteration_type):
    from common import ClientServerTest
    # efa's dgram endpoint requires prefix therefore must always test with prefix mode on
    test = ClientServerTest(cmdline_args, "fi_dgram_pingpong", iteration_type,
                            prefix_type="with_prefix")
    test.run()

