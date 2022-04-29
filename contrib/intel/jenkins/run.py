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

def fi_info_test(core, hosts, mode, util):

    fi_info_test = tests.FiInfoTest(jobname=jbname,buildno=bno,
                                    testname='fi_info', core_prov=core,
                                    fabric=fab, hosts=hosts,
                                    ofi_build_mode=mode, util_prov=util)
    print("-------------------------------------------------------------------")
    print("Running fi_info test for {}-{}-{}".format(core, util, fab))
    fi_info_test.execute_cmd()
    print("-------------------------------------------------------------------")

def fabtests(core, hosts, mode, util):

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

def shmemtest(core, hosts, mode, util):

    runshmemtest = tests.ShmemTest(jobname=jbname,buildno=bno,
                                   testname="shmem test", core_prov=core,
                                   fabric=fab, hosts=hosts,
                                   ofi_build_mode=mode, util_prov=util)

    print("-------------------------------------------------------------------")
    if (runshmemtest.execute_condn):
#        skip unit because it is failing shmem_team_split_2d
#        print("running shmem unit test for {}-{}-{}".format(core, util, fab))
#        runshmemtest.execute_cmd("unit")
        print("Running shmem PRK test for {}-{}-{}".format(core, util, fab))
        runshmemtest.execute_cmd("prk")

        print("---------------------------------------------------------------")
        print("Running shmem ISx test for {}-{}-{}".format(core, util, fab))
        runshmemtest.execute_cmd("isx")

        print("---------------------------------------------------------------")
        print("Running shmem uh test for {}-{}-{}".format(core, util, fab))
        runshmemtest.execute_cmd("uh")
    else:
        print("Skipping {} {} as execute condition fails"\
              .format(core, runshmemtest.testname))
    print("-------------------------------------------------------------------")

def ze_fabtests(core, hosts, mode, util):
    runzefabtests = tests.ZeFabtests(jobname=jbname,buildno=bno,
                                     testname="ze test", core_prov=core,
                                     fabric=fab, hosts=hosts,
                                     ofi_build_mode=mode, util_prov=util)

    print("-------------------------------------------------------------------")
    if (runzefabtests.execute_condn):
        print("Running ze tests for {}-{}-{}".format(core, util, fab))
        runzefabtests.execute_cmd()
    else:
        print("Skipping {} {} as execute condition fails"\
              .format(core, runfabzetests.testname))
    print("-------------------------------------------------------------------")

def intel_mpi_benchmark(core, hosts, mpi, mode, group, util):

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

def mpich_test_suite(core, hosts, mpi, mode, util):

    mpich_tests = tests.MpichTestSuite(jobname=jbname,buildno=bno,
                                       testname="MpichTestSuite",core_prov=core,
                                       fabric=fab, mpitype=mpi, hosts=hosts,
                                       ofi_build_mode=mode, util_prov=util)

    print("-------------------------------------------------------------------")
    if (mpich_tests.execute_condn == True and \
        mpich_tests.mpi_gen_execute_condn == True):
        print("Running mpichtestsuite: Spawn Tests " \
              "for {}-{}-{}-{}".format(core, util, fab, mpi))
        mpich_tests.execute_cmd("spawn")
    else:
        print("Skipping {} {} as execute condition fails" \
              .format(mpi.upper(), mpich_tests.testname))
    print("-------------------------------------------------------------------")

def osu_benchmark(core, hosts, mpi, mode, util):

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

def oneccltest(core, hosts, mode, util):

    runoneccltest = tests.OneCCLTests(jobname=jbname,buildno=bno,
                                      testname="oneccl test", core_prov=core,
                                      fabric=fab, hosts=hosts,
                                      ofi_build_mode=mode, util_prov=util)

    print("-------------------------------------------------------------------")
    if (runoneccltest.execute_condn):
        print("Running oneCCL examples test for {}-{}-{}" \
              .format(core, util, fab))
        runoneccltest.execute_cmd("examples")

        print("---------------------------------------------------------------")
        print("Running oneCCL functional test for {}-{}-{}" \
              .format(core, util, fab))
        runoneccltest.execute_cmd("functional")
    else:
        print("Skipping {} as execute condition fails" \
              .format(runoneccltest.testname))
    print("-------------------------------------------------------------------")

if __name__ == "__main__":
    pass
