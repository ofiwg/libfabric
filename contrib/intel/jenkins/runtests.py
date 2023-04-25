import argparse
import os
import sys
sys.path.append(os.environ['CI_SITE_CONFIG'])
import ci_site_config
import run
import common

parser = argparse.ArgumentParser()

parser.add_argument('--prov', help="core provider", choices=['verbs', \
                     'tcp', 'udp', 'sockets', 'shm', 'psm3'])
parser.add_argument('--util', help="utility provider", choices=['rxd', 'rxm'])
parser.add_argument('--ofi_build_mode', help="specify the build configuration", \
                    choices = ['dbg', 'dl'], default='reg')
parser.add_argument('--test', help="specify test to execute", \
                    choices = ['all', 'shmem', 'IMB', 'osu', 'oneccl', \
                               'mpichtestsuite', 'fabtests', 'onecclgpu', \
                               'fi_info', 'daos', 'multinode'])

parser.add_argument('--imb_grp', help="IMB test group {1:[MPI1, P2P], \
                    2:[EXT, IO], 3:[NBC, RMA, MT]", choices=['1', '2', '3'])
parser.add_argument('--device', help="optional gpu device", choices=['ze'])
parser.add_argument('--user_env', help="Run with additional environment variables", \
                    default='{}')

args = parser.parse_args()
args_core = args.prov

args_util = args.util
args_device = args.device
user_env = args.user_env

if (args.ofi_build_mode):
    ofi_build_mode = args.ofi_build_mode
else:
    ofi_build_mode='reg'

if (args.test):
    run_test = args.test
else:
    run_test = 'all'

if (args.imb_grp):
    imb_group = args.imb_grp
else:
    imb_group = '1'

node = (os.environ['NODE_NAME']).split('_')[0]
hosts = [node]

mpilist = ['impi', 'mpich', 'ompi']

#this script is executed from /tmp
#this is done since some mpi tests
#look for a valid location before running
# the test on the secondary host(client)
# but jenkins only creates a valid path on
# the primary host (server/test node)

os.chdir('/tmp/')

if(args_core):
    for host in ci_site_config.node_map[node]:
        hosts.append(host)

        if (args.device != 'ze'):
            if (run_test == 'all' or run_test == 'fi_info'):
                run.fi_info_test(args_core, hosts, ofi_build_mode, user_env, run_test,
                                 util=args.util)

            if (run_test == 'all' or run_test == 'fabtests'):
                run.fabtests(args_core, hosts, ofi_build_mode, user_env, run_test, args_util)

            if (run_test == 'all' or run_test == 'shmem'):
                run.shmemtest(args_core, hosts, ofi_build_mode, user_env, run_test, args_util)

            if (run_test == 'all' or run_test == 'oneccl'):
                run.oneccltest(args_core, hosts, ofi_build_mode, user_env, run_test, args_util)

            if (run_test == 'all' or run_test == 'onecclgpu'):
                run.oneccltestgpu(args_core, hosts, ofi_build_mode, user_env, run_test, args_util)

            if (run_test == 'all' or run_test == 'daos'):
                run.daos_cart_tests(args_core, hosts, ofi_build_mode, user_env, run_test, args_util)

            if (run_test == 'all' or run_test == 'multinode'):
                run.multinodetest(args_core, hosts, ofi_build_mode, user_env, run_test, args_util)

            for mpi in mpilist:
                if (run_test == 'all' or run_test == 'mpichtestsuite'):
                    run.mpich_test_suite(args_core, hosts, mpi,
                                         ofi_build_mode, user_env, run_test, args_util)
                if (run_test == 'all' or run_test == 'IMB'):
                    run.intel_mpi_benchmark(args_core, hosts, mpi,
                                            ofi_build_mode, imb_group,
                                            user_env, run_test, args_util)
                if (run_test == 'all' or run_test == 'osu'):
                    run.osu_benchmark(args_core, hosts, mpi,
                                      ofi_build_mode, user_env, run_test, args_util)
        else:
            run.ze_fabtests(args_core, hosts, ofi_build_mode, user_env, run_test, args_util)

else:
    print("Error : Specify a core provider to run tests")
