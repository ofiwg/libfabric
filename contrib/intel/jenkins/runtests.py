import argparse
import os
import sys
sys.path.append(os.environ['CI_SITE_CONFIG'])
import ci_site_config
import run
import common

parser = argparse.ArgumentParser()
parser.add_argument("--prov", help="core provider", choices=["psm2", "verbs", \
                     "tcp", "udp", "sockets", "shm"])
parser.add_argument("--ofi_build_mode", help="specify the build configuration", \
                     choices = ["dbg", "dl"])

args = parser.parse_args()
args_prov = args.prov


if (args.ofi_build_mode):
    ofi_build_mode = args.ofi_build_mode
else:
    ofi_build_mode='reg'

node = (os.environ['NODE_NAME']).split('-')[0]
hosts = [node]
mpilist = ['impi', 'mpich', 'ompi']

#this script is executed from /tmp
#this is done since some mpi tests
#look for a valid location before running
# the test on the secondary host(client)
# but jenkins only creates a valid path on 
# the primary host (server/test node)

os.chdir('/tmp/')

if(args_prov):
    for host in ci_site_config.node_map[node]:
        hosts.append(host)

    for prov in common.prov_list:
        if (prov.core == args_prov):
            if (prov.util == None):
                run.fi_info_test(prov.core, hosts, ofi_build_mode)
                run.fabtests(prov.core, hosts, ofi_build_mode)
                for mpi in mpilist:
                    run.intel_mpi_benchmark(prov.core, hosts, mpi, ofi_build_mode)   
                    run.mpistress_benchmark(prov.core, hosts, mpi, ofi_build_mode)
                    run.osu_benchmark(prov.core, hosts, mpi, ofi_build_mode)  
            else:
                run.fi_info_test(prov.core, hosts, ofi_build_mode, util=prov.util)
                run.fabtests(prov.core, hosts, ofi_build_mode, util=prov.util)
                for mpi in mpilist:
                    run.intel_mpi_benchmark(prov.core, hosts, mpi, ofi_build_mode, \
                                           util=prov.util,)
                    run.mpistress_benchmark(prov.core, hosts, mpi, ofi_build_mode, \
                                            util=prov.util)
                    run.osu_benchmark(prov.core, hosts, mpi, ofi_build_mode, \
                                             util=prov.util)
else:
    print("Error : Specify a core provider to run tests")
    
