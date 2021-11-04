import argparse
import os
import sys
sys.path.append(os.environ['CI_SITE_CONFIG'])
import ci_site_config
import run
import common

parser = argparse.ArgumentParser()

parser.add_argument("--prov", help="core provider", choices=["verbs", \
                     "tcp", "udp", "sockets", "shm"])
parser.add_argument("--util", help="utility provider", choices=["rxd", "rxm"])
parser.add_argument("--ofi_build_mode", help="specify the build configuration", \
                    choices = ["dbg", "dl"])
parser.add_argument("--test", help="specify test to execute", \
                    choices = ["all", "unit", "shmem", "mpi", "oneccl"])
parser.add_argument("--device", help="optional gpu device", choices=["ze"])

args = parser.parse_args()
args_core = args.prov

args_util = args.util
args_device = args.device

if (args.ofi_build_mode):
    ofi_build_mode = args.ofi_build_mode
else:
    ofi_build_mode='reg'

if (args.test):
    run_test = args.test
else:
    run_test = 'all'

node = (os.environ['NODE_NAME']).split('-')[0]
hosts = [node]
# Note: Temporarily disabling all mpich testing
# due to mpich options issues which is causing
# multiple tests to fail.
#mpilist = ['impi', 'mpich', 'ompi']
mpilist = ['impi', 'ompi']

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

    if (args_util == None):
        if (args.device != 'ze'):
            if (run_test == 'all' or run_test == 'unit'):
                run.fi_info_test(args_core, hosts, ofi_build_mode)
                run.fabtests(args_core, hosts, ofi_build_mode)

            if (run_test == 'all' or run_test == 'shmem'):
                run.shmemtest(args_core, hosts, ofi_build_mode)

            if (run_test == 'all' or run_test == 'oneccl'):
                run.oneccltest(args_core, hosts, ofi_build_mode)

            if (run_test == 'all' or run_test == 'all'):
                for mpi in mpilist:
                    run.mpich_test_suite(args_core, hosts, mpi, ofi_build_mode)
                    run.intel_mpi_benchmark(args_core, hosts, mpi, ofi_build_mode)
                    run.osu_benchmark(args_core, hosts, mpi, ofi_build_mode)
        else:
            run.ze_fabtests(args_core, hosts, ofi_build_mode)
    else:
        if (run_test == 'all' or run_test == 'unit'):
            run.fi_info_test(args_core, hosts, ofi_build_mode, util=args_util)
            run.fabtests(args_core, hosts, ofi_build_mode, util=args_util)

        if (run_test == 'all' or run_test == 'shmem'):
            run.shmemtest(args_core, hosts, ofi_build_mode, util=args_util)

        if (run_test == 'all' or run_test == 'oneccl'):
            run.oneccltest(args_core, hosts, ofi_build_mode, util=args_util)

        if (run_test == 'all' or run_test == 'all'):
            for mpi in mpilist:
                run.mpich_test_suite(args_core, hosts, mpi, ofi_build_mode, \
                                     util=args_util)
                run.intel_mpi_benchmark(args_core, hosts, mpi, ofi_build_mode, \
                                        util=args_util)
                run.osu_benchmark(args_core, hosts, mpi, ofi_build_mode, \
                                  util=args_util)
else:
    print("Error : Specify a core provider to run tests")
