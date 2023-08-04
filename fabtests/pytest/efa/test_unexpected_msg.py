import pytest

@pytest.mark.functional
@pytest.mark.parametrize("msg_size", ["1", "512", "9000", "1048576"]) # cover various switch points of shm/efa protocols
@pytest.mark.parametrize("msg_count", ["1", "1024", "2048"]) # below and above efa/shm's default rx size
def test_unexpected_msg(cmdline_args, msg_size, msg_count):
    from common import ClientServerTest
    test = ClientServerTest(cmdline_args, f"fi_unexpected_msg -e rdm -I 10 -S {msg_size} -M {msg_count}")
    test.run()
