import collections
import subprocess
import sys
import os

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
    if (p.returncode != 0):
        print("exiting with " + str(p.poll()))
        sys.exit(p.returncode)

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
default_prov_list = [
    'verbs',
    'tcp',
    'sockets',
    'udp',
    'shm',
    'psm3',
    'ucx'
]
daos_prov_list = [
    'verbs',
    'tcp'
]
dsa_prov_list = [
    'shm'
]
gpu_prov_list = [
    'verbs',
    'shm'
]
common_disable_list = [
    'usnic',
    'efa',
    'perf',
    'rstream',
    'hook_debug',
    'bgq',
    'mrail',
    'opx'
]
default_enable_list = [
    'ze-dlopen'
]
