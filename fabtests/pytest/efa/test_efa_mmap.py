import pytest
from common import UnitTest


@pytest.mark.unit
def test_efa_mmap(cmdline_args):
    test = UnitTest(cmdline_args, "fi_efa_mmap_test")
    test.run()
