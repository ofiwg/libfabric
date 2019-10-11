import tests
import subprocess
import sys
import argparse
import os
import common

sys.path.append(os.environ['CI_SITE_CONFIG'])
import ci_site_config

fab = os.environ['FABRIC']#args.fabric
brname = os.environ['BRANCH_NAME']#args.branchname
bno = os.environ['BUILD_NUMBER']#args.buildno


#run fi_info test
def fi_info_test(core, hosts, mode,util=None):
    
    fi_info_test = tests.FiInfoTest(branchname=brname,buildno=bno,\
                    testname="fi_info", core_prov=core, fabric=fab,\
                         hosts=hosts, ofi_build_mode=mode, util_prov=util)
    print("running fi_info test for {}-{}-{}".format(core, util, fab))
    fi_info_test.execute_cmd()
        

#runfabtests
def fabtests(core, hosts, mode, util=None):
       
       runfabtest = tests.Fabtest(branchname=brname,buildno=bno,\
                    testname="runfabtests", core_prov=core, fabric=fab,\
                         hosts=hosts, ofi_build_mode=mode, util_prov=util)

       if (runfabtest.execute_condn):
            print("running fabtests for {}-{}-{}".format(core, util, fab))
            runfabtest.execute_cmd()
       else:
            print("skipping {} as execute condition fails"\
                  .format(runfabtest.testname))
       print("----------------------------------------------------------------------------------------\n")
    

#imb-tests
def intel_mpi_benchmark(core, hosts, mpi, mode, util=None):

    imb_test = tests.MpiTestIMB(branchname=brname,buildno=bno,\
               testname="IntelMPIbenchmark",core_prov=core, fabric=fab,\
               hosts=hosts, mpitype=mpi, ofi_build_mode=mode, util_prov=util)
    
    if (imb_test.execute_condn == True  and imb_test.mpi_gen_execute_condn == True):
        print("running imb-test for {}-{}-{}-{}".format(core, util, fab, mpi))
        imb_test.execute_cmd()
    else:
        print("skipping {} as execute condition fails"\
                    .format(imb_test.testname))
    print("----------------------------------------------------------------------------------------\n")
    
#mpi_stress benchmark tests
def mpistress_benchmark(core, hosts, mpi, mode, util=None):

    stress_test = tests.MpiTestStress(branchname=brname,buildno=bno,\
                  testname="stress",core_prov=core, fabric=fab, mpitype=mpi,\
                  hosts=hosts, ofi_build_mode=mode, util_prov=util)
 
    if (stress_test.execute_condn == True and stress_test.mpi_gen_execute_condn == True):
        print("running mpistress-test for {}-{}-{}-{}".format(core, util, fab, mpi))
        stress_test.execute_cmd()
    else:
        print("skipping {} as execute condition fails" \
                    .format(stress_test.testname))
    print("----------------------------------------------------------------------------------------\n")

#osu benchmark tests    
def osu_benchmark(core, hosts, mpi, mode, util=None):

    osu_test = tests.MpiTestOSU(branchname=brname, buildno=bno, \
               testname="osu-benchmarks",core_prov=core, fabric=fab, mpitype=mpi, \
               hosts=hosts, ofi_build_mode=mode, util_prov=util)
    
    if (osu_test.execute_condn == True and osu_test.mpi_gen_execute_condn == True):
        print("running osu-test for {}-{}-{}-{}".format(core, util, fab, mpi))
        osu_test.execute_cmd()
    else:
        print("skipping {} as execute condition fails" \
                .format(osu_test.testname))
    print("----------------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    pass


