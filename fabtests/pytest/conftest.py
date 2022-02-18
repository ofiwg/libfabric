import pytest

def pytest_addoption(parser):
    parser.addoption("--provider", dest="provider", help="libfabric provider")
    parser.addoption("--client_id", dest="client_id", help="client IP address or hostname")
    parser.addoption("--server_id", dest="server_id", help="server IP address or hostname")
    parser.addoption("--good_address", dest="good_address",
                     help="good address from host's perspective (default $GOOD_ADDR)")
    parser.addoption("--exclusion_list", dest="exclusion_list", help="a list of regex patterns")
    parser.addoption("--environments", dest="environments",
                     help="export provided variable name and value to ssh client and server processes.")
    parser.addoption("--exclusion_file", dest="exclusion_file",
                     help="a file that contains a list of regex patterns (one per line)")
    parser.addoption("--exclude_negative_tests", dest="exclude_negative_tests", action="store_true",
                     help="exclude negative unit tests")
    parser.addoption("--binpath", dest="binpath", help="path to test bins (default PATH)") 
    parser.addoption("--client_interface", dest="client_interface", type=str, help="client interface")
    parser.addoption("--server_interface", dest="server_interface", type=str, help="server interface")
    parser.addoption("--ubertest_config_file", dest="ubertest_config_file", type=str,
                     help="configure option for ubertest tests")
    parser.addoption("--timeout", dest="timeout", default="120",
                     help="timeout value for each test, default to 120 seconds")
    parser.addoption("--pin-core", dest="core_list", type=str, help="Specify cores to pin when running standard tests. Cores can specified via a comma-delimited list, like 0,2-4")
    parser.addoption("--strict_fabtests_mode", dest="strict_fabtests_more", action="store_true",
                     help="strict mode. -FI_ENODATA and -FI_NOSYS treated as failure instead of skip/notrun") 
    parser.addoption("--additional_server_arguments", dest="additional_server_arguments", type=str,
                     help="addtional arguments passed to server programs")
    parser.addoption("--additional_client_arguments", dest="additional_client_arguments", type=str,
                     help="addtional arguments passed to client programs")
    parser.addoption("--oob_address_exchange", dest="oob_address_exchange", action="store_true",
                     help="use out of band address exchange")

# base ssh command
bssh = "ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=2 -o BatchMode=yes"

class CmdlineArgs:

    def __init__(self, request):
        self.provider = request.config.getoption("--provider")
        if self.provider is None:
            raise RuntimeError("Error: libfabric provider is not specified")

        self.server_id = request.config.getoption("--server_id")
        if self.server_id is None:
            raise RuntimeError("Error: server is not specified")

        self.client_id = request.config.getoption("--client_id")
        if self.client_id is None:
            raise RuntimeError("Error: client is not specified")

        self.good_address = request.config.getoption("--good_address")
        self.environments = request.config.getoption("--environments")

        self._exclusion_patterns = []
        exclusion_list = request.config.getoption("--exclusion_list")
        if exclusion_list:
            self._add_exclusion_patterns_from_list(exclusion_list)

        exclusion_file = request.config.getoption("--exclusion_file")
        if exclusion_file:
            self._add_exclusion_patterns_from_file(exclusion_file)

        self.exclude_negative_tests = request.config.getoption("--exclude_negative_tests")

        self.binpath = request.config.getoption("--binpath")

        self.client_interface = request.config.getoption("--client_interface")
        if self.client_interface is None:
            self.client_interface = self.client_id

        self.server_interface = request.config.getoption("--server_interface")
        if self.server_interface is None:
            self.server_interface = self.server_id

        self.ubertest_config_file = request.config.getoption("--ubertest_config_file")
        self.timeout = int(request.config.getoption("--timeout"))
        self.core_list = request.config.getoption("--pin-core")
        self.strict_fabtests_mode = request.config.getoption("--strict_fabtests_mode")
        self.additional_server_arguments = request.config.getoption("--additional_server_arguments")
        self.additional_client_arguments = request.config.getoption("--additional_client_arguments")
        self.oob_address_exchange = request.config.getoption("--oob_address_exchange")

    def populate_command(self, base_command, host_type):
        '''
            populate base command with informations in command line: provider, environments, etc
        '''
        command = base_command
        # use binpath if specified
        if not (self.binpath is None):
            command = self.binpath + "/" + command

        command = "timeout " + str(self.timeout) + " " + command

        # set environment variables if specified
        if not (self.environments is None):
            command = self.environments + " " + command

        if command.find("fi_ubertest") == -1:
            command = self._populate_normal_command(command, host_type)
        else:
            command = self._populate_ubertest_command(command, host_type)

        if host_type == "host" or host_type == "server":
            command = bssh + " " + self.server_id + " " + command
        else:
            assert host_type == "client"
            command = bssh + " " + self.client_id + " " + command

        return command

    def is_test_excluded(self, test_base_command, test_is_negative=False):
        if test_is_negative and self.exclude_negative_tests:
            return True

        for pattern in self._exclusion_patterns:
            if pattern.search(test_base_command):
                return True

        return False

    def _add_exclusion_patterns_from_list(self, exclusion_list):
        import re
        pattern_strs = exclusion_list.split(",")
        for pattern_str in pattern_strs:
            self._exclusion_patterns.append(re.compile(pattern_str))

    def _add_exclusion_patterns_from_file(self, exclusion_file):
        import re
        
        ifs = open(exclusion_file)
        line = ifs.readline()
        while len(line) > 0:
            line = line.strip()
            if len(line)>0 and line[0] != '#':
                self._exclusion_patterns.append(re.compile(line))
            line = ifs.readline()

    def _populate_normal_command(self, command, host_type):
        # setup provider
        assert self.provider
        command = command + " -p " + self.provider

        if host_type == "host":
            return command

        if host_type == "server":
            if self.oob_address_exchange:
                command += " -E"
            else:
                command += " -s " + self.server_interface

            if self.additional_server_arguments:
                command += " " + self.additional_server_arguments

            return command

        assert host_type == "client"
        if self.oob_address_exchange:
            command += " -E " + self.server_id
        else:
            command += " -s " + self.client_interface + " " + self.server_interface

        if self.additional_client_arguments:
            command += " " + self.additional_client_arguments

        return command

    def _populate_ubertest_command(self, command, host_type):
        assert command.find("ubertest") != -1
        if host_type == "server":
            return command + " -x"

        assert host_type == "client"
        assert self.ubertest_config_file
        assert self.client_id
        return command + " -u " + self.ubertest_config_file + " " + self.client_id

@pytest.fixture
def cmdline_args(request):
    return CmdlineArgs(request)

@pytest.fixture
def good_address(cmdline_args):
    import os

    if cmdline_args.good_address:
        return cmdline_args.good_address

    if "GOOD_ADDR" in os.environ:
        return os.environ["GOOD_ADDR"]

    if cmdline_args.server_interface:
        return cmdline_args.server_interface

    return cmdline_args.server_id

@pytest.fixture
def server_address(cmdline_args, good_address):
    if cmdline_args.oob_address_exchange:
        return good_address

    return cmdline_args.server_id

@pytest.fixture(scope="module", params=["transmit_complete", "delivery_complete"])
def completion_type(request):
    return request.param

@pytest.fixture(scope="module", params=["with_prefix", "wout_prefix"])
def prefix_type(request):
    return request.param

@pytest.fixture(scope="module", params=["with_datacheck", "wout_datacheck"])
def datacheck_type(request):
    return request.param
