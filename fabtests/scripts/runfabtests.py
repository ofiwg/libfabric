#!/usr/bin/env python3
#
# Copyright (c) 2021-2022 Amazon.com, Inc. or its affiliates. All rights reserved.
#
# This software is available to you under a choice of one of two
# licenses.  You may choose to be licensed under the terms of the GNU
# General Public License (GPL) Version 2, available from the file
# COPYING in the main directory of this source tree, or the
# BSD license below:
#
#     Redistribution and use in source and binary forms, with or
#     without modification, are permitted provided that the following
#     conditions are met:
#
#      - Redistributions of source code must retain the above
#        copyright notice, this list of conditions and the following
#        disclaimer.
#
#      - Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials
#        provided with the distribution.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

def get_ubertest_test_type(fabtests_testsets):
    test_list = fabtests_testsets.split(",")

    for test in test_list:
        if test == "quick" or test == "ubertest_quick" or test == "ubertest":
            return "quick"

        if test == "all" or test == "ubertest_all":
            return "all"

        if test == "verify" or test == "ubertest_verify":
            return "verify"

    return None

def fabtests_testsets_to_pytest_markers(fabtests_testsets):
    test_set = set()
    test_list = fabtests_testsets.split(",")

    # use set() to remove duplicate test set
    for test in test_list:
        if test == "quick":
            test_set.add("unit")
            test_set.add("functional")
            test_set.add("short")
            test_set.add("ubertest_quick")
        elif test =="ubertest":
            test_set.add("ubertest_quick")
        elif test == "all":
            test_set.add("unit")
            test_set.add("functional")
            test_set.add("standard")
            test_set.add("multinode")
            test_set.add("ubertest_all")
        elif test == "verify":
            test_set.add("ubertest_verify")
        else:
            test_set.add(test)

    markers = None
    for test in test_set:
        if markers is None:
            markers = test[:]
        else:
            markers += " or " + test

    return markers

def get_default_exclusion_file(fabtests_args):
    import os
    test_configs_dir = os.path.abspath(os.path.join(get_pytest_root_dir(), "..", "test_configs"))
    exclusion_file = os.path.join(test_configs_dir, fabtests_args.provider,
                                  fabtests_args.provider + ".exclude")
    if not os.path.exists(exclusion_file):
        return None

    return exclusion_file

def get_default_ubertest_config_file(fabtests_args):
    import os
 
    test_configs_dir = os.path.abspath(os.path.join(get_pytest_root_dir(), "..", "test_configs"))
    provider = fabtests_args.provider
    if provider.find(";") != -1:
        core,util = fabtests_args.provider.split(";")
        cfg_file = os.path.join(test_configs_dir, util, core + ".test")
    else:
        core = fabtests_args.provider
        ubertest_test_type = get_ubertest_test_type(fabtests_args.testsets)
        if not ubertest_test_type:
            return None

        cfg_file = os.path.join(test_configs_dir, core, ubertest_test_type + ".test")

    if not os.path.exists(cfg_file):
        return None

    return cfg_file

def fabtests_args_to_pytest_args(fabtests_args):
    import os

    pytest_args = []

    pytest_args.append("--provider=" + fabtests_args.provider)
    pytest_args.append("--server_id=" + fabtests_args.server_id)
    pytest_args.append("--client_id=" + fabtests_args.client_id)

    # -v make pytest to print 1 line for each test
    pytest_args.append("-v")

    if fabtests_args.good_address:
        pytest_args.append("--good_address " + fabtests_args.good_address)

    pytest_verbose_options = {
            0 : "-rN",      # print no extra information
            1 : "-rfE",     # print extra information for failed test(s)
            2 : "-rfEsx",   # print extra information for failed/skipped test(s)
            3 : "-rA",      # print extra information for all test(s) (failed/skipped/passed)
        }

    print("fabtests_args.verbose: {}".format(fabtests_args.verbose))
    pytest_args.append(pytest_verbose_options[fabtests_args.verbose])

    verbose_fail = fabtests_args.verbose > 0
    if verbose_fail:
        # Use short python trace back because it show captured stdout of failed tests
        pytest_args.append("--tb=short")
    else:
        pytest_args.append("--tb=no")

    markers = fabtests_testsets_to_pytest_markers(fabtests_args.testsets)
    pytest_args.append("-m")
    pytest_args.append(markers)

    if fabtests_args.environments:
        pytest_args.append("--environments=" + fabtests_args.environments)

    if fabtests_args.exclusion_list:
        pytest_args.append("--exclusion_list=" + fabtests_args.exclusion_list)

    default_exclusion_file = get_default_exclusion_file(fabtests_args)
    if fabtests_args.exclusion_file:
        pytest_args.append("--exclusion_file=" + fabtests_args.exclusion_file)
    elif default_exclusion_file:
        pytest_args.append("--exclusion_file=" + default_exclusion_file)

    if fabtests_args.exclude_negative_tests:
        pytest_args.append("--exclude_negative_tests")

    if fabtests_args.binpath:
        pytest_args.append("--binpath=" + fabtests_args.binpath)

    if fabtests_args.client_interface:
        pytest_args.append("--client_interface=" + fabtests_args.client_interface)

    if fabtests_args.server_interface:
        pytest_args.append("--server_interface=" + fabtests_args.server_interface)

    default_ubertest_config_file = get_default_ubertest_config_file(fabtests_args)
    if fabtests_args.ubertest_config_file:
        pytest_args.append("--ubertest_config_file=" + fabtests_args.ubertest_config_file)
    elif default_ubertest_config_file:
        pytest_args.append("--ubertest_config_file=" + default_ubertest_config_file)

    pytest_args.append("--timeout={}".format(fabtests_args.timeout))

    if fabtests_args.core_list:
        pytest_args.append("--pin-core=" + fabtests_args.core_list)

    if fabtests_args.strict:
        pytest_args.append("--strict_fabtests_mode")

    if fabtests_args.additional_client_arguments:
        pytest_args.append("--additional_client_arguments=" + fabtests_args.additional_client_arguments)

    if fabtests_args.additional_server_arguments:
        pytest_args.append("--additional_server_arguments=" + fabtests_args.additional_server_arguments)

    if fabtests_args.oob_address_exchange:
        pytest_args.append("--oob_address_exchange")

    if fabtests_args.expression:
        pytest_args.append("-k")
        pytest_args.append(fabtests_args.expression)

    if fabtests_args.html:
        pytest_args.append("--html")
        pytest_args.append(os.path.abspath(fabtests_args.html))
        pytest_args.append("--self-contained-html")

    if fabtests_args.junit_xml:
        pytest_args.append("--junit-xml")
        pytest_args.append(os.path.abspath(fabtests_args.junit_xml))
        pytest_args.append("--self-contained-html")

    return pytest_args

def get_pytest_root_dir():
    '''
        find the pytest root directory according the location of runfabtests.py
    '''
    import os
    import sys
    script_path = os.path.abspath(sys.argv[0])
    script_dir = os.path.dirname(script_path)
    if os.path.basename(script_dir) == "bin":
        # runfabtests.py is part of a fabtests installation
        pytest_root_dir = os.path.abspath(os.path.join(script_dir, "..", "share", "fabtests", "pytest"))
    elif os.path.basename(script_dir) == "scripts":
        # runfabtests.py is part of a fabtests source code
        pytest_root_dir = os.path.abspath(os.path.join(script_dir, "..", "pytest"))
    else:
        raise RuntimeError("Error: runfabtests.py is under directory {}, "
                "which is neither part of fabtests installation "
                "nor part of fabetsts source code".format(script_dir))

    if not os.path.exists(pytest_root_dir):
        raise RuntimeError("Deduced pytest root directory {} does not exist!".format(pytest_root_dir))

    return pytest_root_dir

def get_pytest_relative_case_dir(fabtests_args, pytest_root_dir):
    '''
        the directory that contains test cases, relative to pytest_root_dir
    '''
    import os

    # provider's own test directory (if exists) overrides default
    pytest_case_dir = os.path.join(pytest_root_dir, fabtests_args.provider)
    if os.path.exists(pytest_case_dir):
        return fabtests_args.provider

    assert os.path.exists(os.path.join(pytest_root_dir, "default"))
    return "default"

def main():
    import os
    import sys
    import pytest
    import argparse

    parser = argparse.ArgumentParser(description="libfabric integration test runner")

    parser.add_argument("provider", type=str, help="libfabric provider")
    parser.add_argument("server_id", type=str, help="server ip or hostname")
    parser.add_argument("client_id", type=str, help="client ip or hostname")
    parser.add_argument("-g", dest="good_address",
                        help="good address from host's perspective (default $GOOD_ADDR)")
    parser.add_argument("-v", dest="verbose", action="count", default=0,
                        help="verbosity level"
                             "-v: print extra info for failed test(s)"
                             "-vv: print extra info of failed/skipped test(s)"
                             "-vvv: print extra info of failed/skipped/passed test(s)")
    parser.add_argument("-t", dest="testsets", type=str, default="quick",
                        help="test set(s): all,quick,unit,functional,standard,short,ubertest (default quick)")
    parser.add_argument("-E", dest="environments", type=str,
                        help="export provided variable name and value to ssh client and server processes.")
    parser.add_argument("-e", dest="exclusion_list", type=str,
                        help="exclude tests: comma delimited list of test names/regex patterns"
                             " e.g. \"dgram,rma.*write\"")
    parser.add_argument("-f", dest="exclusion_file", type=str,
                        help="exclude tests file: File containing list of test names/regex patterns"
                             " to exclude (one per line)")
    parser.add_argument("-N", dest="exclude_negative_tests", action="store_true", help="exlcude negative unit tests")
    parser.add_argument("-p", dest="binpath", type=str, help="path to test bins (default PATH)")
    parser.add_argument("-c", dest="client_interface", type=str, help="client interface")
    parser.add_argument("-s", dest="server_interface", type=str, help="server interface")
    parser.add_argument("-u", dest="ubertest_config_file", type=str, help="configure option for ubertest tests")
    parser.add_argument("-T", dest="timeout", type=int, default=120, help="timeout value in seconds")
    parser.add_argument("--pin-core", dest="core_list", type=str, help="Specify cores to pin when running standard tests. Cores can specified via a comma-delimited list, like 0,2-4")
    parser.add_argument("-S", dest="strict", action="store_true",
                        help="Strict mode: -FI_ENODATA, -FI_ENOSYS errors would be treated as failures"
                             " instead of skipped/notrun")
    parser.add_argument("-C", dest="additional_client_arguments", type=str,
                        help="Additional client test arguments: Parameters to pass to client fabtests")
    parser.add_argument("-L", dest="additional_server_arguments", type=str,
                        help="Additional server test arguments: Parameters to pass to server fabtests")
    parser.add_argument("-b", dest="oob_address_exchange", action="store_true",
                        help="out-of-band address exchange over the default port")
    parser.add_argument("--expression", dest="expression", type=str,
                        help="only run tests which match the given substring expression.")
    parser.add_argument("--html", dest="html", type=str,
                        help="path to generated html report")
    parser.add_argument("--junit_xml", dest="junit_xml", type=str,
                        help="path to generated junit xml report")

    fabtests_args = parser.parse_args()
    pytest_args = fabtests_args_to_pytest_args(fabtests_args)

    # find the directory that contains testing scripts
    pytest_root_dir = get_pytest_root_dir()
    os.chdir(pytest_root_dir)

    pytest_args.append(get_pytest_relative_case_dir(fabtests_args, pytest_root_dir))

    pytest_command = "cd " + pytest_root_dir + "; pytest"
    for arg in pytest_args:
        if arg.find(' ') != -1:
            arg = "'" + arg + "'"
        pytest_command += " " + arg
    print(pytest_command)

    # actually running tests
    exit(pytest.main(pytest_args))

main()
