import pytest

@pytest.mark.functional
def test_flood_peer(cmdline_args, fabric):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args, f"fi_flood -e rdm -W 6400 -S 512 -T 5",
                            timeout=300, fabric=fabric)
    test.run()
