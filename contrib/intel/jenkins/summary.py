import os
from pickle import FALSE
import sys

# add jenkins config location to PATH
sys.path.append(os.environ['CI_SITE_CONFIG'])

import ci_site_config
import argparse
import subprocess
import shlex
import common
import re
import shutil

verbose = False

def print_results(stage_name, passes, fails, failed_tests, excludes,
                  excluded_tests):
    total = passes + fails
    percent = passes/total * 100
    print(f"{stage_name}: {passes}/{total} = {percent:.2f} % Pass")
    if fails:
        print(f"\tFailed tests: {fails}")
        for test in failed_tests:
                print(f'\t\t{test}')
    if (verbose):
        if excludes:
            print(f"\tExcluded/Notrun tests: {excludes} ")
            for test in excluded_tests:
                print(f'\t\t{test}')


def summarize_fabtests(log_dir, prov, build_mode=None):
    file_name = f'{prov}_fabtests_{build_mode}'
    if not os.path.exists(f'{log_dir}/{file_name}'):
        return

    log = open(f'{log_dir}/{file_name}', 'r')
    line = log.readline()
    passes = 0
    fails = 0
    excludes = 0
    failed_tests = []
    excluded_tests = []
    test_name_string='no_test'
    while line:
        # don't double count ubertest output
        if 'ubertest' in line and 'client_cmd:' in line:
            while 'name:' not in line: # skip past client output in ubertest
                line = log.readline()

        if 'name:' in line:
            test_name = line.split()[2:]
            test_name_string = ' '.join(test_name)

        if 'result:' in line:
            result_line = line.split()
            # lines can look like 'result: Pass' or
            # 'Ending test 1 result: Success'
            result = (result_line[result_line.index('result:') + 1]).lower()
            if result == 'pass' or result == 'success':
                    passes += 1

            if result == 'fail':
                fails += 1
                if 'ubertest' in test_name_string:
                    idx = (result_line.index('result:') - 1)
                    ubertest_number = int((result_line[idx].split(',')[0]))
                    failed_tests.append(f"{test_name_string}: " \
                                        f"{ubertest_number}")
                else:
                    failed_tests.append(test_name_string)

            if result == 'excluded' or result == 'notrun':
                excludes += 1
                excluded_tests.append(test_name_string)

        if "exiting with" in line:
            fails += 1
            failed_tests.append(test_name_string)

        line = log.readline()

    print_results(f"{prov} fabtests {build_mode}", passes, fails, failed_tests,
                  excludes, excluded_tests)

    log.close()

def summarize_oneccl(log_dir, prov, build_mode=None):
    if 'GPU' in prov:
        file_name = f'{prov}_onecclgpu_{build_mode}'
    else:
        file_name = f'{prov}_oneccl_{build_mode}'

    if not os.path.exists(f'{log_dir}/{file_name}'):
        return

    log = open(f'{log_dir}/{file_name}', 'r')
    line = log.readline()
    passes = 0
    fails = 0
    failed_tests = []
    name = 'no_test'
    while line:
        #lines look like path/run_oneccl.sh ..... -test examples ..... test_name
        if " -test" in line:
            tokens = line.split()
            name = f"{tokens[tokens.index('-test') + 1]} " \
                   f"{tokens[len(tokens) - 1]}"

        if 'PASSED' in line:
            passes += 1

        if 'FAILED' in line or "exiting with" in line:
            fails += 1
            failed_tests.append(name)

        line = log.readline()

    print_results(f"{prov} oneccl {build_mode}", passes, fails, failed_tests, \
                  excludes=0, excluded_tests=[])

    log.close()


if __name__ == "__main__":
#read Jenkins environment variables
    # In Jenkins,  JOB_NAME  = 'ofi_libfabric/master' vs BRANCH_NAME = 'master'
    # job name is better to use to distinguish between builds of different
    # jobs but with same branch name.
    jobname = os.environ['JOB_NAME']
    buildno = os.environ['BUILD_NUMBER']
    workspace = os.environ['WORKSPACE']

    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_item', help="functional test to summarize",
                         choices=['fabtests', 'impi', 'ompi', 'mpichtestsuite'])
    parser.add_argument('--ofi_build_mode', help="select buildmode debug or dl",
                        choices=['dbg', 'dl'])
    parser.add_argument('-v', help="Verbose mode. Print excluded tests", \
                        action='store_true')

    args = parser.parse_args()
    verbose = args.v

    args = parser.parse_args()
    summary_item = args.summary_item

    if (args.ofi_build_mode):
        ofi_build_mode = args.ofi_build_mode
    else:
        ofi_build_mode = 'reg'

    log_dir = f'{ci_site_config.install_dir}/{jobname}/{buildno}/log_dir'

    if summary_item == 'fabtests':
        for prov,util in common.prov_list:
            if util:
                prov = f'{prov}-{util}'

            summarize_fabtests(log_dir, prov, ofi_build_mode)

    if summary_item == 'impi':
        print('impi')
    if summary_item == 'ompi':
        print('ompi')
    if summary_item == 'mpichtestsuite':
        print('mpichtestsuite')
