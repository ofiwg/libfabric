import pytest

@pytest.mark.multinode
@pytest.mark.parametrize("C", ["msg", "rma"])
def test_multinode(cmdline_args, C):
    from common import MultinodeTest
    command = "fi_multinode -C " + C
    numproc = 3
    test = MultinodeTest(cmdline_args, command, 3)
    test.run()

@pytest.mark.multinode
def test_multinode_coll(cmdline_args):
    from common import MultinodeTest
    numproc = 3
    test = MultinodeTest(cmdline_args, "fi_multinode_coll", 3)
    test.run()
