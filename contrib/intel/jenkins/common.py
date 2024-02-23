import collections
import subprocess
import sys
import os
from subprocess import Popen, TimeoutExpired
from time import sleep

def get_node_name(host, interface):
   return '%s-%s' % (host, interface)

def run_command(command):
    print(" ".join(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    print(p.returncode)
    while True:
        out = p.stdout.read(1)
        if (out == '' and p.poll() != None):
            break
        if (out != ''):
            sys.stdout.write(out)
            sys.stdout.flush()

    print(f"Return code is {p.returncode}")
    if (p.returncode != 0):
        print("exiting with " + str(p.poll()))
        sys.exit(p.returncode)

def run_logging_command(command, log_file):
    print("filename: ".format(log_file))
    f = open(log_file, 'a')
    print(" ".join(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    print(p.returncode)
    f.write(" ".join(command) + '\n')
    while True:
        out = p.stdout.read(1)
        f.write(out)
        if (out == '' and p.poll() != None):
            break
        if (out != ''):
            sys.stdout.write(out)
            sys.stdout.flush()

    print(f"Return code is {p.returncode}")
    if (p.returncode != 0):
        print("exiting with " + str(p.poll()))
        f.close()
        sys.exit(p.returncode)
    f.close()

def read_file(file_name):
    with open(file_name) as file_out:
        output = file_out.read()
    return output

class ClientServerTest:
    def __init__(self, server_cmd, client_cmd, server_log, client_log,
                timeout=None):
        self.server_cmd = server_cmd
        self.client_cmd = client_cmd
        self.server_log = server_log
        self.client_log = client_log
        self._timeout = timeout

    def run(self):
        server_process = Popen(
            f"{self.server_cmd} > {self.server_log} 2>&1",
            shell=True, close_fds=True
        )
        sleep(1)
        client_process = Popen(
            f"{self.client_cmd} > {self.client_log} 2>&1",
            shell=True, close_fds=True
        )

        try:
            server_process.wait(timeout=self._timeout)
        except TimeoutExpired:
            server_process.terminate()

        try:
            client_process.wait(timeout=self._timeout)
        except TimeoutExpired:
            client_process.terminate()

        server_output = read_file(self.server_log)
        client_output = read_file(self.client_log)

        print("")
        print(f"server_command: {self.server_cmd}")
        print('server_stdout:')
        print(server_output)
        print(f"client_command: {self.client_cmd}")
        print('client_stdout:')
        print(client_output)

        return (server_process.returncode, client_process.returncode)

Prov = collections.namedtuple('Prov', 'core util')
prov_list = [
   Prov('psm3', None),
   Prov('verbs', None),
   Prov('verbs', 'rxd'),
   Prov('verbs', 'rxm'),
   Prov('sockets', None),
   Prov('tcp', None),
   Prov('udp', None),
   Prov('udp', 'rxd'),
   Prov('shm', None),
   Prov('ucx', None)
]

providers = {
    'daos'      : {
                    'enable'     : ['verbs', 'tcp'],
                    'disable'    : []
                  },
    'gpu'       : {
                    'enable'     : ['verbs', 'shm'],
                    'disable'    : ['psm3']
                  },
    'dsa'       : {
                    'enable'     : ['shm'],
                    'disable'    : []
                  },
    'ucx'       : {
                    'enable'    : ['ucx'],
                    'disable'   : []
                  },
    'water'     : {
                    'enable'     : ['tcp', 'verbs', 'psm3', 'sockets'],
                    'disable'    : []
                  },
    'grass'     : {
                    'enable'     : ['tcp', 'sockets', 'udp', 'shm'],
                    'disable'    : []
                  },
    'fire'      : {
                    'enable'     : ['shm'],
                    'disable'    : []
                  },
    'cyndaquil' : {
                    'enable'     : ['shm'],
                    'disable'    : []
                  },
    'quilava'   : {
                    'enable'     : ['shm'],
                    'disable'    : []
                  },
    'ivysaur': {
                    'enable'    : ['tcp'],
                    'disable'   : []
                  },
    'electric'  : {
                    'enable'     : ['shm'],
                    'disable'    : []
                  }
}

common_disable_list = [
    'efa',
    'perf',
    'hook_debug',
    'mrail',
    'opx'
]

cloudbees_log_start_string = "Begin Cloudbees Test Output"
