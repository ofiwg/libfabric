import collections
import ci_site_config
import subprocess
import sys
import os

def get_node_name(host, interface):
   return '%s-%s' % (host, interface)

def run_command(command, logdir=None, test_type=None, ofi_build_mode=None):
    stage_name = os.environ['STAGE_NAME']
    if (test_type and ('tcp-rxm' in stage_name)):
        filename = f'{logdir}/MPI_tcp-rxm_{test_type}_{ofi_build_mode}'
    elif (test_type and ('MPI_net' in stage_name)):
        filename = f'{logdir}/MPI_net_{test_type}_{ofi_build_mode}'
    elif (test_type and ofi_build_mode):
        filename = f'{logdir}/{stage_name}_{test_type}_{ofi_build_mode}'
    else:
        filename = f'{logdir}/{stage_name}'
    print("filename: ".format(filename))
    if (logdir):
        f = open(filename, 'a')
    print(" ".join(command))
    p = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    print(p.returncode)
    if (logdir):
        f.write(" ".join(command) + '\n')
    while True:
        out = p.stdout.read(1)
        if (logdir):
            f.write(out)
        if (out == '' and p.poll() != None):
            break
        if (out != ''):
            sys.stdout.write(out)
            sys.stdout.flush()
    if (p.returncode != 0):
        print("exiting with " + str(p.poll()))
        if (logdir):
            f.close()
        sys.exit(p.returncode)
    if (logdir):
        f.close()


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
]
default_prov_list = [
    'verbs',
    'tcp',
    'sockets',
    'udp',
    'shm',
    'psm3'
]
daos_prov_list = [
    'verbs',
    'tcp'
]
dsa_prov_list = [
    'shm'
]
common_disable_list = [
    'usnic',
    'psm',
    'efa',
    'perf',
    'rstream',
    'hook_debug',
    'bgq',
    'mrail',
    'opx'
]
default_enable_list = [
    'ze_dlopen'
]
