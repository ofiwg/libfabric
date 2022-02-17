import tests
import subprocess
import sys
import argparse
import os
import common

sys.path.append(os.environ['CI_SITE_CONFIG'])
import ci_site_config

# read Jenkins environment variables
# In Jenkins, JOB_NAME = 'ofi_libfabric/master' vs BRANCH_NAME = 'master'
# job name is better to use to distinguish between builds of different
# jobs but with the same branch name.
fab = os.environ['FABRIC']#args.fabric
jbname = os.environ['JOB_NAME']#args.jobname
bno = os.environ['BUILD_NUMBER']#args.buildno

def fi_info_test(core, hosts, mode,util=None):

    fi_info_test = tests.FiInfoTest(jobname=jbname,buildno=bno,
                                    testname='fi_info', core_prov=core,
                                    fabric=fab, hosts=hosts,
                                    ofi_build_mode=mode, util_prov=util)
    print("-------------------------------------------------------------------")
    print("Running fi_info test for {}-{}-{}".format(core, util, fab))
    fi_info_test.execute_cmd()
    print("-------------------------------------------------------------------")

def fabtests(core, hosts, mode, util=None):

    runfabtest = tests.Fabtest(jobname=jbname,buildno=bno,
                               testname='runfabtests', core_prov=core,
                               fabric=fab, hosts=hosts, ofi_build_mode=mode,
                               util_prov=util)

    print("-------------------------------------------------------------------")
    if (runfabtest.execute_condn):
        print("Running Fabtests for {}-{}-{}".format(core, util, fab))
        runfabtest.execute_cmd()
    else:
        print("Skipping {} {} as execute condition fails" \
              .format(core, runfabtest.testname))
    print("-------------------------------------------------------------------")

def shmemtest(core, hosts, mode, util=None):

    runshmemtest = tests.ShmemTest(jobname=jbname,buildno=bno,\
                 testname="shmem test", core_prov=core, fabric=fab,\
                 hosts=hosts, ofi_build_mode=mode, util_prov=util)
    if (runshmemtest.execute_condn):
        print("running shmem unit test for {}-{}-{}".format(core, util, fab))
        runshmemtest.execute_cmd("unit")
        print("running shmem PRK test for {}-{}-{}".format(core, util, fab))
        runshmemtest.execute_cmd("prk")
        print("running shmem ISx test for {}-{}-{}".format(core, util, fab))
        runshmemtest.execute_cmd("isx")
        print("running shmem uh test for {}-{}-{}".format(core, util, fab))
        runshmemtest.execute_cmd("uh")
    else:
        print("skipping {} as execute condition fails"\
              .format(runshmemtest.testname))
    print("----------------------------------------------------------------------------------------\n")

def ze_fabtests(core, hosts, mode, util=None):
    runzefabtests = tests.ZeFabtests(jobname=jbname,buildno=bno,\
                   testname="ze test", core_prov=core, fabric=fab,\
                   hosts=hosts, ofi_build_mode=mode, util_prov=util)
    if (runzefabtests.execute_condn):
        print("running ze tests for {}-{}-{}".format(core, util, fab))
        runzefabtests.execute_cmd()
    else:
        print("skipping {} as execute condition fails"\
              .format(runfabzetests.testname))
    print("----------------------------------------------------------------------------------------\n")

def intel_mpi_benchmark(core, hosts, mpi, mode, group, util=None):

    imb_test = tests.MpiTestIMB(jobname=jbname,buildno=bno,
                                testname='IntelMPIbenchmark', core_prov=core,
                                fabric=fab, hosts=hosts, mpitype=mpi,
                                ofi_build_mode=mode, test_group=group,
                                util_prov=util)

    print("-------------------------------------------------------------------")
    if (imb_test.execute_condn == True and \
        imb_test.mpi_gen_execute_condn == True):
        print("Running IMB-tests for {}-{}-{}-{}".format(core, util, fab, mpi))
        imb_test.execute_cmd()
    else:
        print("Skipping {} {} as execute condition fails" \
              .format(mpi.upper(), imb_test.testname))
    print("-------------------------------------------------------------------")


#mpich_test_suite
def mpich_test_suite(core, hosts, mpi, mode, util=None):

    mpich_tests = tests.MpichTestSuite(jobname=jbname,buildno=bno,\
                  testname="MpichTestSuite",core_prov=core, fabric=fab,\
                  mpitype=mpi, hosts=hosts, ofi_build_mode=mode, \
                  util_prov=util)
    if (mpich_tests.execute_condn == True and \
        mpich_tests.mpi_gen_execute_condn == True):
        print("Running mpich test suite: Spawn coll, comm, dt Tests for {}-{}-{}-{}".format(core, util, fab, mpi))
        os.environ["MPITEST_RETURN_WITH_CODE"] = "1"
        mpich_tests.execute_cmd("spawn")
    print("----------------------------------------------------------------------------------------\n")

def osu_benchmark(core, hosts, mpi, mode, util=None):

    osu_test = tests.MpiTestOSU(jobname=jbname, buildno=bno,
                                testname='osu-benchmarks', core_prov=core,
                                fabric=fab, mpitype=mpi, hosts=hosts,
                                ofi_build_mode=mode, util_prov=util)

    print("-------------------------------------------------------------------")
    if (osu_test.execute_condn == True and \
        osu_test.mpi_gen_execute_condn == True):
        print("Running OSU-Test for {}-{}-{}-{}".format(core, util, fab, mpi))
        osu_test.execute_cmd()
    else:
        print("Skipping {} {} as execute condition fails" \
              .format(mpi.upper(), osu_test.testname))
    print("-------------------------------------------------------------------")

#OneCCL examples and functional tests
def oneccltest(core, hosts, mode, util=None):

    runoneccltest = tests.OneCCLTests(jobname=jbname,buildno=bno, \
                    testname="oneccl test", core_prov=core, fabric=fab, \
                    hosts=hosts, ofi_build_mode=mode, util_prov=util)
    if (runoneccltest.execute_condn):
        print("running oneCCL examples test for {}-{}-{}" \
              .format(core, util, fab))
        runoneccltest.execute_cmd("examples")
        print("running oneCCL functional test for {}-{}-{}" \
              .format(core, util, fab))
        runoneccltest.execute_cmd("functional")
    else:
        print("skipping {} as execute condition fails" \
              .format(runoneccltest.testname))
    print("----------------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    pass
