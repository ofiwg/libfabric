import pytest

def has_cuda(ip):
    from subprocess import run
    proc = run(["ssh", ip, "nvidia-smi", "-L"])
    return proc.returncode == 0

def check_returncode(returncode, strict):
    import errno

    if returncode == 0:
        return

    if not strict:
        if returncode == errno.ENODATA:
            pytest.skip("ENODATA")
            return

        if returncode == errno.ENOSYS:
            pytest.skip("ENOSYS")
            return

    error_msg = "returncode {}".format(returncode)
    # all tests are run under the timeout command
    # which will return 124 when timeout expired.
    if returncode == 124:
        error_msg += ", timeout"

    pytest.fail(error_msg)

class UnitTest:

    def __init__(self, cmdline_args, base_command, is_negative=False):
        self._cmdline_args = cmdline_args
        self._base_command = base_command
        self._is_negative = is_negative
        self._command = cmdline_args.populate_command(base_command, "host")

    def run(self):
        import os
        from tempfile import NamedTemporaryFile
        from subprocess import Popen, TimeoutExpired

        if self._cmdline_args.is_test_excluded(self._base_command, self._is_negative):
            pytest.skip("excluded")
            return

        # start running
        outfile = NamedTemporaryFile(prefix="fabtests_server.out.").name
        process = Popen(self._command + "> " + outfile + " 2>&1", shell=True)

        timeout = False
        try:
            process.wait(timeout=self._cmdline_args.timeout)
        except TimeoutExpired:
            process.terminate()
            timeout = True

        print("")
        print("command: " + self._command)
        print("stdout: ")
        print(open(outfile).read())
        os.unlink(outfile)

        assert not timeout, "timed out"
        check_returncode(process.returncode, self._cmdline_args.strict_fabtests_mode)

class ClientServerTest:

    def __init__(self, cmdline_args, executable, iteration_type=None, completion_type="transmit_complete",
                 prefix_type="wout_prefix", datacheck_type="wout_datacheck", message_size=None,
                 memory_type="host_to_host", timeout=None):

        self._cmdline_args = cmdline_args
        self._server_base_command = self.prepare_base_command("server", executable, iteration_type,
                                                              completion_type, prefix_type,
                                                              datacheck_type, message_size,
                                                              memory_type)
        self._client_base_command = self.prepare_base_command("client", executable, iteration_type,
                                                              completion_type, prefix_type,
                                                              datacheck_type, message_size,
                                                              memory_type)

        if timeout:
            self._timeout = timeout
        else:
            self._timeout = cmdline_args.timeout

        self._server_command = cmdline_args.populate_command(self._server_base_command, "server", self._timeout)
        self._client_command = cmdline_args.populate_command(self._client_base_command, "client", self._timeout)

    def prepare_base_command(self, command_type, executable, iteration_type=None, completion_type="transmit_complete",
                             prefix_type="wout_prefix", datacheck_type="wout_datacheck", message_size=None,
                             memory_type="host_to_host"):
        if executable == "fi_ubertest":
            return "fi_ubertest"

        '''
            all execuables in fabtests (except fi_ubertest) accept a common set of arguments:
                -I: number of iteration
                -U: delivery complete (transmit complete if not specified)
                -k: force prefix mode (not force prefix mode if not specified)
                -v: data verification (no data verification if not specified)
                -S: message size
            this function will construct a command with these options
        '''

        command = executable[:]
        if iteration_type == "short":
            command += " -I 5"
        elif iteration_type == "standard":
            if not (self._cmdline_args.core_list is None):
                command += " --pin-core " + self._cmdline_args.core_list
            pass
        elif iteration_type is None:
            pass
        else:
            command += " -I " + str(iteration_type)

        if completion_type == "delivery_complete":
            command += " -U"
        else:
            assert completion_type == "transmit_complete"

        if datacheck_type == "with_datacheck":
            command += " -v"
        else:
            if datacheck_type != "wout_datacheck":
                print("datacheck_type: " + datacheck_type)
            assert datacheck_type == "wout_datacheck"

        if prefix_type == "with_prefix":
            command += " -k"
        else:
            assert prefix_type == "wout_prefix"

        if message_size:
            command += " -S " + str(message_size)

        # in communication test, client is sender, server is receiver
        client_memory_type,server_memory_type = memory_type.split("_to_")
        if command_type == "server" and server_memory_type == "cuda":
            if not has_cuda(self._cmdline_args.server_id):
                pytest.skip("no cuda device")
                return

            return command + " -D cuda"

        if command_type == "client" and client_memory_type == "cuda":
            if not has_cuda(self._cmdline_args.client_id):
                pytest.skip("no cuda device")
                return

            return command + " -D cuda"

        return command

    def run(self):
        import os
        from time import sleep
        from tempfile import NamedTemporaryFile
        from subprocess import Popen, TimeoutExpired

        if self._cmdline_args.is_test_excluded(self._server_base_command):
            pytest.skip("excluded")
            return

        if self._cmdline_args.is_test_excluded(self._client_base_command):
            pytest.skip("excluded")
            return

        # start running
        server_outfile = NamedTemporaryFile(prefix="fabtests_server.out.").name
        server_process = Popen(self._server_command + " > " + server_outfile + " 2>&1", shell=True)
        sleep(1)
        client_outfile = NamedTemporaryFile(prefix="fabtests_client.out.").name
        client_process = Popen(self._client_command + " > " + client_outfile + " 2>&1", shell=True)

        server_timed_out = False
        try:
            server_process.wait(timeout=self._timeout)
        except TimeoutExpired:
            server_process.terminate()
            server_timed_out = True

        client_timed_out = False
        try:
            client_process.wait(timeout=self._timeout)
        except TimeoutExpired:
            client_process.terminate()
            client_timed_out = True

        print("")
        print("server_command: " + self._server_command)
        print("server_stdout:")
        print(open(server_outfile).read())
        os.unlink(server_outfile)
        print("client_command: " + self._client_command)
        print("client_stdout:")
        print(open(client_outfile).read())
        os.unlink(client_outfile)

        assert not server_timed_out, "server timed out"
        assert not client_timed_out, "client timed out"

        strict = self._cmdline_args.strict_fabtests_mode
        check_returncode(server_process.returncode, strict)
        check_returncode(client_process.returncode, strict)

class MultinodeTest:

    def __init__(self, cmdline_args, base_command, numproc):
        self._cmdline_args = cmdline_args
        self._base_command = base_command
        self._numproc = numproc
        self._timeout = self._cmdline_args.timeout

        multinode_command = self._base_command + " -n {}".format(self._numproc)
        self._server_command = cmdline_args.populate_command(multinode_command, "server", self._timeout)
        self._client_command = cmdline_args.populate_command(multinode_command, "client", self._timeout)

    def run(self):
        import os
        from time import sleep
        from tempfile import NamedTemporaryFile
        from subprocess import Popen, TimeoutExpired

        if self._cmdline_args.is_test_excluded(self._base_command):
            pytest.skip("excluded")
            return

        server_outfile = NamedTemporaryFile(prefix="fabtests_server.out.").name

        # start running
        server_process = Popen(self._server_command + "> " + server_outfile + " 2>&1", shell=True)
        sleep(1)

        numclient = self._numproc - 1
        client_process_list = [None] * numclient
        client_outfile_list = [None] * numclient
        for i in range(numclient):
            client_outfile_list[i] = NamedTemporaryFile(prefix="fabtests_client_{}.out.".format(i)).name
            client_process_list[i] = Popen(self._client_command + "> " + client_outfile_list[i] + " 2>&1", shell=True)

        server_timed_out = False
        try:
            server_process.wait(timeout=self._timeout)
        except TimeoutExpired:
            server_process.terminate()
            server_timed_out = True

        client_timed_out = False
        for i in range(numclient):
            try:
                client_process_list[i].wait(timeout=self._timeout)
            except TimeoutExpired:
                client_process_list[i].terminate()
                client_timed_out = True

        print("")
        print("server_command: " + self._server_command)
        print("server_stdout:")
        print(open(server_outfile).read())
        os.unlink(server_outfile)

        print("client_command: " + self._client_command)
        for i in range(numclient):
            print("client_{}_stdout:".format(i))
            print(open(client_outfile_list[i]).read())
            os.unlink(client_outfile_list[i])

        assert not server_timed_out, "server timed out"
        assert not client_timed_out, "client timed out"

        strict = self._cmdline_args.strict_fabtests_mode
        check_returncode(server_process.returncode, strict)
        for i in range(numclient):
            check_returncode(client_process_list[i].returncode, strict)

