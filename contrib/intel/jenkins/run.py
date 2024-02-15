import tests
import subprocess
import sys
import argparse
import os
import common

sys.path.append(f"{os.environ['WORKSPACE']}/ci_resources/configs/{os.environ['CLUSTER']}")
import cloudbees_config

# read Jenkins environment variables
# In Jenkins, JOB_NAME = 'ofi_libfabric/master' vs BRANCH_NAME = 'master'
# job name is better to use to distinguish between builds of different
# jobs but with the same branch name.
fab = os.environ['FABRIC']
if 'slurm' in fab:
    fab = cloudbees_config.fabric_map[f"{os.environ['SLURM_JOB_PARTITION']}"]

jbname = os.environ['JOB_NAME']#args.jobname
bno = os.environ['BUILD_NUMBER']#args.buildno

def fi_info_test(hw, core, hosts, mode, user_env, log_file, util):

    fi_info_test = tests.FiInfoTest(jobname=jbname,buildno=bno,
                                    testname='fi_info', hw=hw, core_prov=core,
                                    fabric=fab, hosts=hosts,
                                    ofi_build_mode=mode, user_env=user_env,
                                    log_file=log_file, util_prov=util)
    print('-------------------------------------------------------------------')
    print(f"Running fi_info test for {core}-{util}-{fab}")
    fi_info_test.execute_cmd()
    print('-------------------------------------------------------------------')

def fabtests(hw, core, hosts, mode, user_env, log_file, util, way):

    runfabtest = tests.Fabtest(jobname=jbname,buildno=bno,
                               testname='runfabtests', hw=hw, core_prov=core,
                               fabric=fab, hosts=hosts, ofi_build_mode=mode,
                               user_env=user_env, log_file=log_file,
                               util_prov=util, way=way)

    print('-------------------------------------------------------------------')
    if (runfabtest.execute_condn):
        print(f"Running Fabtests for {core}-{util}-{fab}")
        runfabtest.execute_cmd()
    else:
        print(f"Skipping {core} {runfabtest.testname} as exec condn fails")
    print('-------------------------------------------------------------------')

def shmemtest(hw, core, hosts, mode, user_env, log_file, util, weekly):
    runshmemtest = tests.ShmemTest(jobname=jbname,buildno=bno,
                                   testname="shmem test", hw=hw, core_prov=core,
                                   fabric=fab, hosts=hosts,
                                   ofi_build_mode=mode, user_env=user_env,
                                   log_file=log_file, util_prov=util,
                                   weekly=weekly)

    print('-------------------------------------------------------------------')
    if (runshmemtest.execute_condn):
        print(f"Running shmem SOS test for {core}-{util}-{fab}")
        runshmemtest.execute_cmd("sos")

        print('--------------------------------------------------------------')
        print(f"Running shmem PRK test for {core}-{util}-{fab}")
        runshmemtest.execute_cmd("prk")

        print('--------------------------------------------------------------')
        print(f"Running shmem ISx test for {core}-{util}-{fab}")
        runshmemtest.execute_cmd("isx")
    else:
        print(f"Skipping {core} {runshmemtest.testname} as exec condn fails")
    print('-------------------------------------------------------------------')

def multinodetest(hw, core, hosts, mode, user_env, log_file, util):

    runmultinodetest = tests.MultinodeTests(jobname=jbname,buildno=bno,
                                      testname="multinode performance test",
                                      hw=hw, core_prov=core, fabric=fab,
                                      hosts=hosts, ofi_build_mode=mode,
                                      user_env=user_env, log_file=log_file,
                                      util_prov=util)

    print("-------------------------------------------------------------------")
    if (runmultinodetest.execute_condn):
        print("Running multinode performance test for {}-{}-{}" \
              .format(core, util, fab))
        runmultinodetest.execute_cmd()

        print("---------------------------------------------------------------")
    else:
        print("Skipping {} as execute condition fails" \
              .format(runmultinodetest.testname))
    print("-------------------------------------------------------------------")

def intel_mpi_benchmark(hw, core, hosts, mpi, mode, group, user_env, log_file,
                        util):

    imb = tests.IMBtests(jobname=jbname, buildno=bno,
                         testname='IntelMPIbenchmark', core_prov=core, hw=hw,
                         fabric=fab, hosts=hosts, mpitype=mpi,
                         ofi_build_mode=mode, user_env=user_env,
                         log_file=log_file, test_group=group, util_prov=util)

    print('-------------------------------------------------------------------')
    if (imb.execute_condn == True):
        print(f"Running IMB-tests for {core}-{util}-{fab}-{mpi}")
        imb.execute_cmd()
    else:
        print(f"Skipping {mpi.upper} {imb.testname} as execute condition fails")
    print('-------------------------------------------------------------------')

def mpich_test_suite(hw, core, hosts, mpi, mode, user_env, log_file, util,
                     weekly=None):

    mpich_tests = tests.MpichTestSuite(jobname=jbname,buildno=bno,
                                       testname="MpichTestSuite",core_prov=core,
                                       hw=hw, fabric=fab, mpitype=mpi,
                                       hosts=hosts, ofi_build_mode=mode,
                                       user_env=user_env, log_file=log_file,
                                       util_prov=util, weekly=weekly)

    print('-------------------------------------------------------------------')
    if (mpich_tests.execute_condn == True):
        print(f"Running mpichtestsuite for {core}-{util}-{fab}-{mpi}")
        mpich_tests.execute_cmd()
    else:
        print(f"Skipping {mpi.upper()} {mpich_tests.testname} exec condn fails")
    print('-------------------------------------------------------------------')

def osu_benchmark(hw, core, hosts, mpi, mode, user_env, log_file, util):

    osu_test = tests.OSUtests(jobname=jbname, buildno=bno,
                                testname='osu-benchmarks', core_prov=core,
                                hw=hw, fabric=fab, mpitype=mpi, hosts=hosts,
                                ofi_build_mode=mode, user_env=user_env,
                                log_file=log_file, util_prov=util)

    print('-------------------------------------------------------------------')
    if (osu_test.execute_condn == True):
        print(f"Running OSU-Test for {core}-{util}-{fab}-{mpi}")
        osu_test.execute_cmd()
    else:
        print(f"Skipping {mpi.upper()} {osu_test.testname} as exec condn fails")
    print('-------------------------------------------------------------------')

def oneccltest(hw, core, hosts, mode, user_env, log_file, util):

    runoneccltest = tests.OneCCLTests(jobname=jbname,buildno=bno,
                                      testname="oneccl test", core_prov=core,
                                      hw=hw, fabric=fab, hosts=hosts,
                                      ofi_build_mode=mode, user_env=user_env,
                                      log_file=log_file, util_prov=util)

    print('-------------------------------------------------------------------')
    if (runoneccltest.execute_condn):
        print(f"Running oneCCL cpu tests for {core}-{util}-{fab}")
        runoneccltest.execute_cmd()
    else:
        print(f"Skipping {runoneccltest.testname} as execute condition fails")
    print('-------------------------------------------------------------------')

def oneccltestgpu(hw, core, hosts, mode, user_env, log_file, util):

    runoneccltestgpu = tests.OneCCLTestsGPU(jobname=jbname,buildno=bno,
                                         testname="oneccl GPU test",
                                         core_prov=core, hw=hw, fabric=fab,
                                         hosts=hosts, ofi_build_mode=mode,
                                         user_env=user_env, log_file=log_file,
                                         util_prov=util)

    print('-------------------------------------------------------------------')
    if (runoneccltestgpu.execute_condn):
        print(f"Running oneCCL GPU examples test for {core}-{util}-{fab}")
        runoneccltestgpu.execute_cmd('examples')

        print('---------------------------------------------------------------')
        print(f"Running oneCCL GPU functional test for {core}-{util}-{fab}")
        runoneccltestgpu.execute_cmd('functional')
    else:
        print(f"Skipping {runoneccltestgpu.testname} as execute condition fails")
    print('-------------------------------------------------------------------')

def daos_cart_tests(hw, core, hosts, mode, user_env, log_file, util):

    runcarttests = tests.DaosCartTest(jobname=jbname, buildno=bno,
                                      testname="Daos Cart Test", core_prov=core,
                                      hw=hw, fabric=fab, hosts=hosts,
                                      ofi_build_mode=mode, user_env=user_env,
                                      log_file=log_file, util_prov=util)

    print('-------------------------------------------------------------------')
    if (runcarttests.execute_condn):
        print(f"Running cart test for {core}-{util}-{fab}")
        runcarttests.execute_cmd()
    print('-------------------------------------------------------------------')

def dmabuftests(hw, core, hosts, mode, user_env, log_file, util):

    rundmabuftests = tests.DMABUFTest(jobname=jbname,buildno=bno,
                                      testname="DMABUF Tests", core_prov=core,
                                      hw=hw, fabric=fab, hosts=hosts,
                                      ofi_build_mode=mode, user_env=user_env,
                                      log_file=log_file, util_prov=util)

    print('-------------------------------------------------------------------')
    if (rundmabuftests.execute_condn):
        print(f"Running dmabuf H->H tests for {core}-{util}-{fab}")
        rundmabuftests.execute_cmd('H2H')

        print('---------------------------------------------------------------')
        print(f"Running dmabuf H->D tests for {core}-{util}-{fab}")
        rundmabuftests.execute_cmd('H2D')

        print('---------------------------------------------------------------')
        print(f"Running dmabuf D->H tests for {core}-{util}-{fab}")
        rundmabuftests.execute_cmd('D2H')

        print('---------------------------------------------------------------')
        print(f"Running dmabuf D->D tests for {core}-{util}-{fab}")
        rundmabuftests.execute_cmd('D2D')

        print('---------------------------------------------------------------')
    else:
        print(f"Skipping {rundmabuftests.testname} as execute condition fails")
    print('-------------------------------------------------------------------')

if __name__ == "__main__":
    pass
