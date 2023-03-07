import copy
import errno
import os
import subprocess
from subprocess import Popen, TimeoutExpired, run
from tempfile import NamedTemporaryFile
from time import sleep

from retrying import retry

import pytest

SERVER_RESTART_DELAY_MS = 10_1000
CLIENT_RETRY_INTERVAL_MS = 1_000
class SshConnectionError(Exception):

    def __init__(self):
        super().__init__(self, "Ssh connection failed")


def is_ssh_connection_error(exception):
    return isinstance(exception, SshConnectionError)


def has_ssh_connection_err_msg(output):
    err_msgs = ["Connection closed by remote host",
                "Connection reset by peer",
                "Connection refused"]

    for msg in err_msgs:
        if output.find(msg) != -1:
            return True

    return False


@retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
def has_cuda(ip):
    outfile = NamedTemporaryFile(prefix="nvidia_smi.").name
    proc = run("ssh {} nvidia-smi -L > {} 2>&1".format(ip, outfile), shell=True)
    output = open(outfile).read()
    os.unlink(outfile)
    if has_ssh_connection_err_msg(output):
        raise SshConnectionError()

    return proc.returncode == 0


@retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
def has_neuron(ip):
    proc = run("ssh {} neuron-ls -j".format(ip),
               stdout=subprocess.PIPE,
               stderr=subprocess.STDOUT,
               shell=True,
               universal_newlines=True)
    if has_ssh_connection_err_msg(proc.stdout):
        raise SshConnectionError()

    return proc.returncode == 0


@retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
def has_hmem_support(cmdline_args, ip):
    binpath = cmdline_args.binpath or ""
    cmd = "timeout " + str(cmdline_args.timeout) \
          + " " + os.path.join(binpath, "check_hmem") \
          + " " + "-p " + cmdline_args.provider
    if cmdline_args.environments:
        cmd = cmdline_args.environments + " " + cmd
    proc = run("ssh {} {}".format(ip, cmd),
               stdout=subprocess.PIPE,
               stderr=subprocess.STDOUT,
               shell=True,
               universal_newlines=True)
    if has_ssh_connection_err_msg(proc.stdout):
        raise SshConnectionError()

    return proc.returncode == 0


PASS = 1
SKIP = 2
FAIL = 3

def check_returncode(returncode, strict):
    """
    check one return code
    @param returncode: input
    @param strict: whether to use strict mode, which treat all error as failure.
                   In none strict mode, ENODATA and ENOSYS is treated as pass
    @return: a tuple with return type (PASS, SKIP and FAIL), and a messge.
             when return type is PASs, message will be None
    """
    if returncode == 0:
        return PASS, None

    if not strict:
        if returncode == errno.ENODATA:
            return SKIP, "ENODATA"

        if returncode == errno.ENOSYS:
            return SKIP, "ENOSYS"

    error_msg = "returncode {}".format(returncode)
    # all tests are run under the timeout command
    # which will return 124 when timeout expired.
    if returncode == 124:
        error_msg += ", timeout"

    return FAIL, error_msg

def check_returncode_list(returncode_list, strict):
    """
    check a list of returncode, and call pytest's handler accordingly.
        If there is failure in return, call pytest.fail()
        If there is no failure, but there is skip in return, call pytest.skip()
        If there is no failure or skip, do nothing
    @param resultcode_list: a list of return code
    @param strict: a boolean indicating wether strict mode should be used.
    @return: no return
    """
    result = PASS
    reason = None
    for returncode in returncode_list:
        # note that failure has higher priority than skip, therefore:
        #
        #     if a failure is encoutered, we break out immediately
        #     if a skip is encountered, we record it and continue
        #
        # this ensures skip can be overwritten by failure
        cur_result,cur_reason = check_returncode(returncode, strict)

        if cur_result != PASS:
            result = cur_result
            reason = cur_reason

        if cur_result == FAIL:
            break

    if result == FAIL:
        pytest.fail(reason)

    if result == SKIP:
        pytest.skip(reason)


class UnitTest:

    def __init__(self, cmdline_args, base_command, is_negative=False, failing_warn_msgs=None):
        if isinstance(failing_warn_msgs, str):
            failing_warn_msgs = [failing_warn_msgs]

        if failing_warn_msgs:
            self._cmdline_args = copy.copy(cmdline_args)
            self._cmdline_args.append_environ("FI_LOG_LEVEL=warn")
        else:
            self._cmdline_args = cmdline_args

        self._failing_warn_msgs = failing_warn_msgs
        self._base_command = base_command
        self._is_negative = is_negative
        self._command = self._cmdline_args.populate_command(base_command, "host")

    @retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=5000)
    def run(self):
        if self._cmdline_args.is_test_excluded(self._base_command, self._is_negative):
            pytest.skip("excluded")

        # start running
        outfile = NamedTemporaryFile(prefix="fabtests_server.out.").name
        process = Popen(self._command + "> " + outfile + " 2>&1", shell=True)

        timeout = False
        try:
            process.wait(timeout=self._cmdline_args.timeout)
        except TimeoutExpired:
            process.terminate()
            timeout = True

        output = open(outfile).read()
        print("")
        print("command: " + self._command)
        if has_ssh_connection_err_msg(output):
            print("encountered ssh connection issue")
            raise SshConnectionError()

        print("stdout: ")
        print(output)
        os.unlink(outfile)

        assert not timeout, "timed out"
        check_returncode_list([process.returncode], self._cmdline_args.strict_fabtests_mode)

        if self._failing_warn_msgs:
            for msg in self._failing_warn_msgs:
                assert output.find(msg) == -1

class ClientServerTest:

    def __init__(self, cmdline_args, executable,
                 iteration_type=None,
                 completion_type="transmit_complete",
                 prefix_type="wout_prefix",
                 datacheck_type="wout_datacheck",
                 message_size=None,
                 memory_type="host_to_host",
                 timeout=None,
                 warmup_iteration_type=None):

        self._cmdline_args = cmdline_args
        self._timeout = timeout or cmdline_args.timeout
        self._server_base_command = self.prepare_base_command("server", executable, iteration_type,
                                                              completion_type, prefix_type,
                                                              datacheck_type, message_size,
                                                              memory_type, warmup_iteration_type)
        self._client_base_command = self.prepare_base_command("client", executable, iteration_type,
                                                              completion_type, prefix_type,
                                                              datacheck_type, message_size,
                                                              memory_type, warmup_iteration_type)


        self._server_command = self._cmdline_args.populate_command(self._server_base_command, "server", self._timeout)
        self._client_command = self._cmdline_args.populate_command(self._client_base_command, "client", self._timeout)

    def prepare_base_command(self, command_type, executable,
                             iteration_type=None,
                             completion_type="transmit_complete",
                             prefix_type="wout_prefix",
                             datacheck_type="wout_datacheck",
                             message_size=None,
                             memory_type="host_to_host",
                             warmup_iteration_type=None):
        if executable == "fi_ubertest":
            return "fi_ubertest"

        '''
            all execuables in fabtests (except fi_ubertest) accept a common set of arguments:
                -I: number of iteration
                -U: delivery complete (transmit complete if not specified)
                -k: force prefix mode (not force prefix mode if not specified)
                -v: data verification (no data verification if not specified)
                -S: message size
                -w: number of warmup iterations
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

        if warmup_iteration_type:
            command += " -w " + str(warmup_iteration_type)

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
        client_memory_type, server_memory_type = memory_type.split("_to_")
        host_memory_type, host_ip = (server_memory_type, self._cmdline_args.server_id) if command_type == "server" else (
            client_memory_type, self._cmdline_args.client_id)

        if host_memory_type != "host":
            if not has_hmem_support(self._cmdline_args, host_ip):
                pytest.skip("no hmem support")

            if host_memory_type == "cuda" and not has_cuda(host_ip):
                pytest.skip("no cuda device")
            elif host_memory_type == "neuron" and not has_neuron(host_ip):
                pytest.skip("no neuron device")
            elif (client_memory_type == server_memory_type == "neuron") and (
                    self._cmdline_args.server_id == self._cmdline_args.client_id):
                pytest.skip("Neuron to Neuron tests require 2 nodes")

            command = command + " -D " + host_memory_type

        return command

    def _run_client_command(self, server_process, client_command, client_output_file=None,
                            run_client_asynchronously=False):

        if server_process.poll():
            if client_output_file:
                with open(client_output_file, "w") as f:
                    f.write("")
            raise RuntimeError("Server has terminated")

        print("")
        print("client_command: " + client_command)

        process = Popen(client_command, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        if run_client_asynchronously:
            return process

        client_timed_out = False
        output = ""
        try:
            output, _ = process.communicate(timeout=self._timeout)
            if client_output_file:
                with open(client_output_file, "w") as f:
                    f.write(output)
        except TimeoutExpired:
            client_timed_out = True
            process.terminate()

        if has_ssh_connection_err_msg(output):
            print("client encountered ssh connection issue!")
            raise SshConnectionError()

        print("client_stdout:")
        print(output)
        print(f"client returncode: {process.returncode}")

        if client_timed_out:
            raise RuntimeError("Client timed out")

        return process

    @retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=SERVER_RESTART_DELAY_MS)
    def run(self):
        if self._cmdline_args.is_test_excluded(self._server_base_command):
            pytest.skip("excluded")

        if self._cmdline_args.is_test_excluded(self._client_base_command):
            pytest.skip("excluded")

        # Start server
        print("")
        print("server_command: " + self._server_command)
        server_process = Popen(self._server_command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        sleep(1)

        client_returncode = -1
        try:
            # Start client
            # Retry on SSH connection error until server timeout
            client_returncode = retry(
                retry_on_exception=is_ssh_connection_error,
                stop_max_delay=self._timeout * 1000,  # Convert to milliseconds
                wait_fixed=CLIENT_RETRY_INTERVAL_MS,
            )(self._run_client_command)(server_process, self._client_command).returncode
        except Exception as e:
            print("Client error: {}".format(e))
            # Clean up server if client is terminated unexpectedly
            server_process.terminate()

        server_output = ""
        server_timed_out = False
        try:
            server_output, _ = server_process.communicate(
                timeout=self._timeout)
        except TimeoutExpired:
            server_process.terminate()
            server_timed_out = True

        if has_ssh_connection_err_msg(server_output):
            print("encountered ssh connection issue!")
            raise SshConnectionError()

        print("server_stdout:")
        print(server_output)
        print(f"server returncode: {server_process.returncode}")

        if server_timed_out:
            raise RuntimeError("Server timed out")

        check_returncode_list([server_process.returncode, client_returncode],
                              self._cmdline_args.strict_fabtests_mode)


class MultinodeTest(ClientServerTest):

    def __init__(self, cmdline_args, server_base_command,
                 client_base_command, client_hostname_list,
                 run_client_asynchronously=True):
        self._cmdline_args = cmdline_args
        self.numclient = len(client_hostname_list)
        self._timeout = self._cmdline_args.timeout

        self._server_base_command = cmdline_args.populate_command(server_base_command, "server", self._timeout)
        self._client_base_command_list = []
        for client_hostname in client_hostname_list:
            cmdline_args_copy = copy.copy(cmdline_args)
            cmdline_args_copy.client_id = client_hostname
            self._client_base_command_list.append(cmdline_args_copy.populate_command(client_base_command, "client", self._timeout))
        self._run_client_asynchronously = run_client_asynchronously

    @retry(retry_on_exception=is_ssh_connection_error, stop_max_attempt_number=3, wait_fixed=SERVER_RESTART_DELAY_MS)
    def run(self):
        if self._cmdline_args.is_test_excluded(self._server_base_command):
            pytest.skip("excluded")

        # _client_base_command_list is populated from the same client_base_command,
        # so it should be safe to only check the first one.
        if self._cmdline_args.is_test_excluded(self._client_base_command_list[0]):
            pytest.skip("excluded")

        server_outfile = NamedTemporaryFile(prefix="fabtests_server.out.").name

        # start running
        server_process = Popen(self._server_base_command + "> " + server_outfile + " 2>&1", shell=True)
        sleep(1)

        client_process_list = [None] * self.numclient
        client_outfile_list = [None] * self.numclient
        for i in range(self.numclient):
            client_outfile_list[i] = NamedTemporaryFile(prefix="fabtests_client_{}.out.".format(i)).name
            try:
                # Start client
                # Retry on SSH connection error until server timeout
                client_process_list[i] = retry(
                    retry_on_exception=is_ssh_connection_error,
                    stop_max_delay=self._timeout * 1000,  # Convert to milliseconds
                    wait_fixed=CLIENT_RETRY_INTERVAL_MS,
                )(self._run_client_command)(server_process, self._client_base_command_list[i],
                                            client_outfile_list[i],
                                            self._run_client_asynchronously)
            except Exception as e:
                print("Client error: {}".format(e))
                # Clean up server if client is terminated unexpectedly
                server_process.terminate()

        server_timed_out = False
        try:
            server_process.wait(timeout=self._timeout)
        except TimeoutExpired:
            server_process.terminate()
            server_timed_out = True

        if has_ssh_connection_err_msg(open(server_outfile).read()):
            print("encountered ssh connection issue!")
            raise SshConnectionError()

        client_timed_out = False
        # for syncrhonous clients, this is already
        # handled in self._run_client_command
        if self._run_client_asynchronously:
            for i in range(self.numclient):
                try:
                    output, _ = client_process_list[i].communicate(timeout=self._timeout)
                    with open(client_outfile_list[i], "w") as f:
                        f.write(output)
                except TimeoutExpired:
                    client_process_list[i].terminate()
                    client_timed_out = True

        print("")
        print("server_command: " + self._server_base_command)
        print("server_stdout:")
        print(open(server_outfile).read())
        os.unlink(server_outfile)

        for i in range(self.numclient):
            print("client_{}_command: ".format(i) + self._client_base_command_list[i])
            print("client_{}_stdout:".format(i))
            print(open(client_outfile_list[i]).read())
            os.unlink(client_outfile_list[i])

        if server_timed_out:
            raise RuntimeError("Server timed out")

        if client_timed_out:
            raise RuntimeError("Client timed out")

        strict = self._cmdline_args.strict_fabtests_mode

        returncode_list = [server_process.returncode]
        for i in range(self.numclient):
            returncode_list.append(client_process_list[i].returncode)

        check_returncode_list(returncode_list, strict)
