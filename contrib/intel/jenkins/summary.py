from abc import ABC, abstractmethod
from typing import Tuple
import os
from pickle import FALSE
import sys

# add jenkins config location to PATH
sys.path.append(os.environ['CI_SITE_CONFIG'])

import ci_site_config
import argparse
import common

verbose = False

class Summarizer(ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "print_results")
            and callable(subclass.print_results)
            and hasattr(subclass, "check_name")
            and callable(subclass.check_name)
            and hasattr(subclass, "check_pass")
            and callable(subclass.check_pass)
            and hasattr(subclass, "check_fail")
            and callable(subclass.check_fail)
            and hasattr(subclass, "check_exclude")
            and callable(subclass.check_exclude)
            and hasattr(subclass, "read_file")
            and callable(subclass.read_file)
            and hasattr(subclass, "run")
            and callable(subclass.run)
            or NotImplemented
        )

    @abstractmethod
    def __init__(self, log_dir, prov, file_name, stage_name):
        self.log_dir = log_dir
        self.prov = prov
        self.file_name = file_name
        self.stage_name = stage_name
        self.file_path = os.path.join(self.log_dir, self.file_name)
        self.exists = os.path.exists(self.file_path)
        self.log = None
        self.passes = 0
        self.passed_tests = []
        self.fails = 0
        self.failed_tests = []
        self.excludes = 0
        self.excluded_tests = []
        self.test_name ='no_test'

    def print_results(self):
        spacing='\t'
        total = self.passes + self.fails
        # log was empty or not valid
        if not total:
            return

        percent = self.passes/total * 100
        if (verbose):
            print(f"{spacing}<>{self.stage_name}: ".ljust(40), end='')
        else:
            print(f"{spacing}{self.stage_name}: ".ljust(40), end='')
        print(f"{self.passes}/{total} ".ljust(10), end='')
        print(f"= {percent:.2f}%".ljust(12), end = '')
        print("Pass", end = '')
        if (self.excludes > 0):
            print(f"  |  {self.excludes:3.0f} Excluded/Notrun")
        else:
            print()

        if (verbose and self.passes):
            print(f"{spacing}\t Passed tests: {self.passes}")
            for test in self.passed_tests:
                    print(f'{spacing}\t\t{test}')
        if self.fails:
            print(f"{spacing}\tFailed tests: {self.fails}")
            for test in self.failed_tests:
                    print(f'{spacing}\t\t{test}')
        if (verbose):
            if self.excludes:
                print(f"{spacing}\tExcluded/Notrun tests: {self.excludes} ")
                for test in self.excluded_tests:
                    print(f'{spacing}\t\t{test}')

    def check_name(self, line):
        return
 
    def check_pass(self, line):
        return

    def check_fail(self, line):
        if "exiting with" in line:
            self.fails += 1

    def check_exclude(self, line):
        return

    def check_line(self, line):
        self.check_name(line)
        self.check_pass(line)
        self.check_fail(line)
        self.check_exclude(line)

    def read_file(self):
        with open(self.file_path, 'r') as log_file:
            for line in log_file:
                self.check_line(line.lower())

    def summarize(self):
        if not self.exists:
            return 0

        self.read_file()
        self.print_results()
        return int(self.fails)

class FiInfoSummarizer(Summarizer):
    def __init__(self, log_dir, prov, file_name, stage_name):
        super().__init__(log_dir, prov, file_name, stage_name)

    def check_fail(self, line):
        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(f"fi_info {self.prov}")

    def read_file(self):
        super().read_file()

        if not self.fails:
            self.passes += 1
            self.passed_tests.append(f"fi_info {self.prov}")

class FabtestsSummarizer(Summarizer):
    def __init__(self, log_dir, prov, file_name, stage_name):
        super().__init__(log_dir, prov, file_name, stage_name)

    def check_name(self, line):
        # don't double count ubertest output and don't count fi_ubertest's
        # invocation
        if 'ubertest' in line and 'client_cmd:' in line:
            self.test_name = 'no_test'
            if 'name:' not in line: # skip past client output in ubertest
                return

        test_name = line.split("name:")
        if len(test_name) > 1:
            self.test_name = test_name[-1].lower().strip()

    def get_result_line(self, line) -> Tuple[str,str]:
        result = line.split("result:")
        if len(result) > 1:
            return (result[-1].lower().strip(), line.split())
        return None, None

    def check_pass(self, line):
        result, result_line = self.get_result_line(line)
        if result == 'pass' or result == 'success' or result == 'passed':
            self.passes += 1
            if 'ubertest' in self.test_name:
                idx = (result_line.index('result:') - 1)
                ubertest_number = int((result_line[idx].split(',')[0]))
                self.passed_tests.append(f"{self.test_name}: "\
                                         f"{ubertest_number}")
            else:
                self.passed_tests.append(self.test_name)

    def check_fail(self, line):
        result, result_line = self.get_result_line(line)
        if result == 'fail':
            self.fails += 1
            if 'ubertest' in self.test_name:
                idx = (result_line.index('result:') - 1)
                ubertest_number = int((result_line[idx].split(',')[0]))
                self.failed_tests.append(f"{self.test_name}: " \
                                         f"{ubertest_number}")
            else:
                self.failed_tests.append(self.test_name)

        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(self.test_name)

    def check_exclude(self, line):
        result, _ = self.get_result_line(line)
        if result == 'excluded' or result == 'notrun':
            self.excludes += 1
            self.excluded_tests.append(self.test_name)

    def check_line(self, line):
        self.check_name(line)
        if (self.test_name != 'no_test'):
            self.check_pass(line)
            self.check_fail(line)
            self.check_exclude(line)

class MultinodePerformanceSummarizer(Summarizer):
    def __init__(self, log_dir, prov, file_name, stage_name):
        super().__init__(log_dir, prov, file_name, stage_name)

    def check_name(self, line):
        #name lines look like "starting <test_name>... <result>"
        if 'starting' in line and '...' in line:
            self.test_name = line.split()[1].split('.')[0]

    def check_pass(self, line):
        if 'pass' in line:
            self.passes += 1
            self.passed_tests.append(self.test_name)

    def check_fail(self, line):
        if 'fail' in line:
            self.fails += 1
            self.failed_tests.append(self.test_name)

        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(self.test_name)

class OnecclSummarizer(Summarizer):
    def __init__(self, log_dir, prov, file_name, stage_name):
        super().__init__(log_dir, prov, file_name, stage_name)
        self.file_path = os.path.join(self.log_dir, self.file_name)
        self.exists = os.path.exists(self.file_path)
        self.name = 'no_test'

    def check_name(self, line):
        #lines look like path/run_oneccl.sh ..... -test examples ..... test_name
        if " -test" in line:
            tokens = line.split()
            self.name = f"{tokens[tokens.index('-test') + 1]} " \
                   f"{tokens[len(tokens) - 1]}"

    def check_pass(self, line):
        if 'passed' in line:
            self.passes += 1
            self.passed_tests.append(self.name)

    def check_fail(self, line):
        if 'failed' in line or "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(self.name)

class ShmemSummarizer(Summarizer):
    def __init__(self, log_dir, prov, file_name, stage_name):
        super().__init__(log_dir, prov, file_name, stage_name)
        self.shmem_type = {
            'uh'    : { 'func' : self.check_uh,
                        'keyphrase' : 'summary'
                      },
            'isx'   : { 'func' : self.check_isx,
                        'keyphrase' : 'scaling'
                      },
            'prk'   : { 'func' : self.check_prk,
                        'keyphrase' : 'solution'
                      }
        }
        self.keyphrase = self.shmem_type[self.prov]['keyphrase']
        self.name = 'no_test'

    def check_uh(self, line, log_file):
        # (test_002) Running test_shmem_atomics.x: Test all atomics... OK
        # (test_003) Running test_shmem_barrier.x: Tests barrier ... Failed
        if "running test_" in line:
            tokens = line.split()
            for token in tokens:
                if 'test_' in token:
                    self.name = token
            if tokens[len(tokens) - 1] == 'ok':
                self.passes += 1
                self.passed_tests.append(self.name)
            else:
                self.fails += 1
                self.failed_tests.append(self.name)
        # Summary
        # x/z Passed.
        # y/z Failed.
        if self.keyphrase in line: #double check
            passed = log_file.readline().lower()
            failed = log_file.readline().lower()
            if self.passes != int(passed.split()[1].split('/')[0]):
                print(f"passes {self.passes} do not match log reported passes " \
                        f"{int(passed.split()[1].split('/')[0])}")
            if self.fails != int(failed.split()[1].split('/')[0]):
                print(f"fails {self.fails} does not match log fails " \
                        f"{int(failed.split()[1].split('/')[0])}")

    def check_prk(self, line, log_file=None):
        if self.keyphrase in line:
            self.passes += 1
        if 'error:' in line or "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(f"{self.prov} {self.passes + self.fails}")
        if 'test(s)' in line:
            if int(line.split()[0]) != self.fails:
                print(f"fails {self.fails} does not match log reported fails " \
                    f"{int(line.split()[0])}")

    def check_isx(self, line, log_file=None):
        if self.keyphrase in line:
            self.passes += 1
        if ('failed' in line and 'test(s)' not in line) or \
            "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(f"{self.prov} {self.passes + self.fails}")
        if 'test(s)' in line:
            if int(line.split()[0]) != self.fails:
                print(f"fails {self.fails} does not match log reported fails " \
                        f"{int(line.split()[0])}")

    def check_fails(self, line):
        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(f"{self.prov} {self.passes + self.fails}")

    def check_line(self, line, log_file):
        self.shmem_type[self.prov]['func'](line, log_file)
        self.check_fails(line)

    def read_file(self):
        with open(self.file_path, 'r') as log_file:
            for line in log_file:
                self.check_line(line.lower(), log_file)

class MpichTestSuiteSummarizer(Summarizer):
    def __init__(self, log_dir, prov, mpi, file_name, stage_name):
        super().__init__(log_dir, prov, file_name, stage_name)

        self.mpi = mpi
        if self.mpi == 'impi':
            self.run = '/mpiexec'
        else:
            self.run = '/mpirun'

    def check_name(self, line):
        if self.run in line:
            self.name = line.split()[len(line.split()) - 1].split('/')[1]
            #assume pass
            self.passes += 1
            self.passed_tests.append(self.name)

    def check_fail(self, line):
        # Fail cases take away assumed pass
        if "exiting with" in line:
            self.fails += 1
            self.passes -= 1
            self.failed_tests.append(f'{self.name}')
            #skip to next test
            while self.run not in line:
                line = self.log.readline().lower()

class ImbSummarizer(Summarizer):
    def __init__(self, log_dir, prov, mpi, file_name, stage_name):
        super().__init__(log_dir, prov, file_name, stage_name)

        self.mpi = mpi
        if self.mpi == 'impi':
            self.run = 'mpiexec'
        else:
            self.run = 'mpirun'
        self.test_type = ''

    def check_type(self, line):
        if 'part' in line:
            self.test_type = line.split()[len(line.split()) - 2]

    def check_name(self, line):
        if "benchmarking" in line:
            self.name = line.split()[len(line.split()) - 1]

    def check_pass(self, line):
        if "benchmarking" in line:
            self.passes += 1
            self.passed_tests.append(self.name)

    def check_fail(self, line):
        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(f"{self.test_type} {self.name}")
            self.passes -= 1

    def check_line(self, line):
        self.check_type(line)
        self.check_name(line)
        self.check_pass(line)
        self.check_fail(line)
        super().check_exclude(line)

class OsuSummarizer(Summarizer):
    def __init__(self, log_dir, prov, mpi, file_name, stage_name):
        super().__init__(log_dir, prov, file_name, stage_name)
        self.mpi = mpi
        if self.mpi == 'impi':
            self.run = 'mpiexec'
        else:
            self.run = 'mpirun'

        self.type = ''
        self.tokens = []

    def get_tokens(self, line):
        if "# osu" in line:
            self.tokens = line.split()
        else:
            self.tokens = []

    def check_name(self, line):
        if 'osu' in self.tokens:
            self.name = " ".join(self.tokens[self.tokens.index('osu') + \
                        1:self.tokens.index('test')])

    def check_type(self):
        if self.tokens:
            self.test_type = self.tokens[1]

    def check_pass(self, line):
        if 'osu' in self.tokens:
            # Assume pass
            self.passes += 1
            self.passed_tests.append(self.name)

    def check_fail(self, line):
        if "exiting with" in line:
            self.fails += 1
            self.failed_tests.append(f"{self.test_type} {self.name}")
            # Remove assumed pass
            self.passes -= 1

    def check_line(self, line):
        self.get_tokens(line)
        self.check_name(line)
        self.check_type()
        self.check_pass(line)
        self.check_fail(line)
        super().check_exclude(line)

class DaosSummarizer(Summarizer):
    def __init__(self, log_dir, prov, file_name, stage_name):
        super().__init__(log_dir, prov, file_name, stage_name)

    def check_name(self, line):
        if "reading ." in line:
            self.test_name = line.split('/')[len(line.split('/')) - 1] \
                             .rstrip('.yaml\n')

    def check_pass(self, line):
        res_string = line.lstrip("results    :").rstrip()
        res_list = res_string.split(' | ')
        for elem in res_list:
            if 'pass' in elem:
                self.passes += [int(s) for s in elem.split() if s.isdigit()][0]
                display_testname = self.test_name.ljust(20)
                self.passed_tests.append(f"{display_testname} : {res_string}")

    def check_fail(self, line):
        res_list = line.lstrip("results    :").rstrip().split('|')
        for elem in res_list:
            if 'pass' not in elem:
                self.fails += [int(s) for s in elem.split() if s.isdigit()][0]
                if self.fails != 0:
                    self.failed_tests.append(f'{self.test_name}')
        return (self.fails)

    def check_line(self, line):
        self.check_name(line)
        if "results    :" in line:
            self.check_pass(line)
            self.check_fail(line)


if __name__ == "__main__":
#read Jenkins environment variables
    # In Jenkins,  JOB_NAME  = 'ofi_libfabric/master' vs BRANCH_NAME = 'master'
    # job name is better to use to distinguish between builds of different
    # jobs but with same branch name.
    jobname = os.environ['JOB_NAME']
    buildno = os.environ['BUILD_NUMBER']

    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_item', help="functional test to summarize",
                         choices=['fabtests', 'imb', 'osu', 'mpichtestsuite',
                         'oneccl', 'shmem', 'ze', 'multinode', 'daos', 'all'])
    parser.add_argument('--ofi_build_mode', help="select buildmode debug or dl",
                        choices=['dbg', 'dl', 'reg'], default='all')
    parser.add_argument('-v', help="Verbose mode. Print all tests", \
                        action='store_true')
    parser.add_argument('--release', help="This job is testing a release."\
                        "It will be saved and checked into a git tree.",
                        action='store_true')

    args = parser.parse_args()
    verbose = args.v
    summary_item = args.summary_item
    release = args.release

    mpi_list = ['impi', 'mpich', 'ompi']

    if (args.ofi_build_mode):
        ofi_build_mode = args.ofi_build_mode
    else:
        ofi_build_mode = 'reg'

    log_dir = f'{ci_site_config.install_dir}/{jobname}/{buildno}/log_dir'
    ret = 0
    err = 0

    build_modes = ['reg', 'dbg', 'dl']
    for mode in build_modes:
        if ofi_build_mode != 'all' and mode != ofi_build_mode:
            continue
        print(f"Summarizing {mode} build mode")
        if ((summary_item == 'daos' or summary_item == 'all') 
             and mode == 'reg'):
            for prov in ['tcp', 'verbs']:       
                ret = DaosSummarizer(log_dir, prov,
                                     f'daos_{prov}_daos_{mode}',
                                     f"{prov} daos {mode}").summarize()
                err += ret if ret else 0

        if summary_item == 'fabtests' or summary_item == 'all':
            for prov,util in common.prov_list:
                if util:
                    prov = f'{prov}-{util}'
                ret = FabtestsSummarizer(log_dir, prov,
                                         f'{prov}_fabtests_{mode}',
                                         f"{prov} fabtests {mode}").summarize()
                err += ret if ret else 0
                ret = FiInfoSummarizer(log_dir, prov, f'{prov}_fi_info_{mode}',
                                       f"{prov} fi_info {mode}").summarize()
                err += ret if ret else 0

        if summary_item == 'imb' or summary_item == 'all':
            for mpi in mpi_list:
                for item in ['tcp-rxm', 'verbs-rxm', 'net']:
                    ret = ImbSummarizer(log_dir, item, mpi,
                                        f'MPI_{item}_{mpi}_IMB_{mode}',
                                        f"{item} {mpi} IMB {mode}").summarize()
                    err += ret if ret else 0

        if summary_item == 'osu' or summary_item == 'all':
            for mpi in mpi_list:
                    for item in ['tcp-rxm', 'verbs-rxm']:
                        ret = OsuSummarizer(log_dir, item, mpi,
                                            f'MPI_{item}_{mpi}_osu_{mode}',
                                            f"{item} {mpi} OSU {mode}"
                                           ).summarize()
                        err += ret if ret else 0

        if summary_item == 'mpichtestsuite' or summary_item == 'all':
            for mpi in mpi_list:
                    for item in ['tcp-rxm', 'verbs-rxm', 'sockets']:
                        ret = MpichTestSuiteSummarizer(log_dir, item, mpi,
                                    f'MPICH testsuite_{item}_{mpi}_'\
                                    f'mpichtestsuite_{mode}',
                                    f"{item} {mpi} mpichtestsuite "\
                                    f"{mode}").summarize()
                        err += ret if ret else 0
        if summary_item == 'multinode' or summary_item == 'all':
            for prov,util in common.prov_list:
                if util:
                    prov = f'{prov}-{util}'

                ret = MultinodePerformanceSummarizer(log_dir, prov,
                                        f'multinode_performance_{prov}_{mode}',
                                        f"multinode performance {prov} {mode}"
                                        ).summarize()
                err += ret if ret else 0

        if summary_item == 'oneccl' or summary_item == 'all':
            stage_name = f"{prov} {mode}"
            ret = OnecclSummarizer(log_dir, 'oneCCL',
                                   f'oneCCL_oneccl_{mode}',
                                   f'oneCCL {mode}').summarize()
            err += ret if ret else 0
            ret = OnecclSummarizer(log_dir, 'oneCCL-GPU',
                                   f'oneCCL-GPU_onecclgpu_{mode}',
                                   f'oneCCL-GPU {mode}').summarize()
            err += ret if ret else 0

        if summary_item == 'shmem' or summary_item == 'all':
            ret = ShmemSummarizer(log_dir, 'uh', f'SHMEM_uh_shmem_{mode}',
                                  f"shmem uh {mode}").summarize()
            err += ret if ret else 0
            ret = ShmemSummarizer(log_dir, 'prk', f'SHMEM_prk_shmem_{mode}',
                                  f"shmem prk {mode}").summarize()
            err += ret if ret else 0
            ret = ShmemSummarizer(log_dir, 'isx', f'SHMEM_isx_shmem_{mode}',
                                  f"shmem isx {mode}").summarize()
            err += ret if ret else 0

        if summary_item == 'ze' or summary_item == 'all':
            test_types = ['h2d', 'd2d', 'xd2d']
            for type in test_types:
                ret = FabtestsSummarizer(log_dir, 'shm',
                                         f'ze-{prov}_{type}_{mode}',
                                         f"ze {prov} {type} {mode}").summarize()
                err += ret if ret else 0

    exit(err)
